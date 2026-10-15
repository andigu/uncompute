/* fasteea.c — C implementation of the FastEEA walk model (the
 * integer-image twin of eea_common.EEABitState that gen_constants.py
 * fits its tables on): one Algorithm-3 step, per-step window-need
 * recording, and batch entry points that fold the two fit
 * reductions (base histogram / dev-tail) into the lane loop.
 *
 * The Python FastEEA in gen_constants.py is the authority; this
 * module is an accelerator.  gen_constants loads it fail-closed: the
 * backend is used only after a per-process bit-exactness gate (full
 * needs arrays + final state vs the Python model), and any mismatch
 * raises rather than falling back.
 *
 * Registers are 259-bit values as 5 x uint64, LSB-first (bit i =
 * position i+1 in the 1-based numbering of Luo et al.), the Python int layout.  Big-endian window
 * arithmetic reverses the extracted value through a byte table,
 * computes little-endian, and reverses back — the borrow/carry
 * direction is the only thing endianness changes.
 *
 * Thread-safe: no global mutable state; all lane scratch is on the
 * stack, so the batch entry points may be driven from concurrent
 * threads (ctypes releases the GIL) as well as processes.
 *
 * Stat order matches gen_constants.STATS and is asserted at load by
 * the Python side of the gate.
 *
 * Build (gen_constants.py does this automatically):
 *   cc -O2 -shared -fPIC -o fasteea.dylib fasteea.c
 */
#include <stdint.h>
#include <string.h>

#define N 256
#define W 259               /* n + 3 */
#define NW 5                /* 5 x 64 = 320 bits */
#define NSTATS 10
#define NEEDMAX 264
#define NMAX_CAP 2048       /* stack-buffer bound on nmax */

/* stat indices — gen_constants.STATS order */
enum { S_RK, S_RKK, S_SWK, S_SWKK, S_TK,
       S_LTK, S_LTKK, S_LRK, S_LRKK, S_NEWK };

typedef uint64_t u64;
typedef struct {
    u64 w1[NW], w2[NW];
    int lt, lq, ls, lrp, p1, p2, sign;
} State;

/* ------------------- multiword helpers (LSB-first) ------------------- */

static unsigned char RB[256];

__attribute__((constructor)) static void rb_init(void) {
    for (int i = 0; i < 256; i++) {
        int v = 0;
        for (int b = 0; b < 8; b++)
            if (i & (1 << b)) v |= 1 << (7 - b);
        RB[i] = (unsigned char)v;
    }
}

static inline u64 rev64(u64 x) {
#if __has_builtin(__builtin_bitreverse64)
    return __builtin_bitreverse64(x);
#endif
    u64 o = 0;
    for (int b = 0; b < 8; b++)
        o |= (u64)RB[(x >> (8 * b)) & 0xff] << (8 * (7 - b));
    return o;
}

static inline int getb(const u64 *r, int pos) {          /* pos 1-based */
    int i = pos - 1;
    return (int)((r[i >> 6] >> (i & 63)) & 1u);
}

static inline void flipb(u64 *r, int pos) {
    int i = pos - 1;
    r[i >> 6] ^= 1ULL << (i & 63);
}

static inline void shr_mw(u64 *v, int s) {               /* v >>= s */
    int q = s >> 6, r = s & 63;
    if (q) {
        for (int i = 0; i + q < NW; i++) v[i] = v[i + q];
        for (int i = NW - q; i < NW; i++) v[i] = 0;
    }
    if (r) {
        for (int i = 0; i < NW - 1; i++)
            v[i] = (v[i] >> r) | (v[i + 1] << (64 - r));
        v[NW - 1] >>= r;
    }
}

static inline void shl_mw(u64 *v, int s) {               /* v <<= s */
    int q = s >> 6, r = s & 63;
    if (q) {
        for (int i = NW - 1; i >= q; i--) v[i] = v[i - q];
        for (int i = 0; i < q; i++) v[i] = 0;
    }
    if (r) {
        for (int i = NW - 1; i > 0; i--)
            v[i] = (v[i] << r) | (v[i - 1] >> (64 - r));
        v[0] <<= r;
    }
}

static inline void mask_mw(u64 *v, int wd) {     /* keep low wd bits */
    for (int i = 0; i < NW; i++) {
        int lo = 64 * i;
        if (wd <= lo) v[i] = 0;
        else if (wd < lo + 64) v[i] &= (1ULL << (wd - lo)) - 1;
    }
}

/* u -= v (full multiword); returns the borrow.  For wd-bit operands
 * already masked below 2^wd the multiword borrow IS the window
 * borrow, since all higher words are zero. */
static inline int sub_mw(u64 *u, const u64 *v) {
    unsigned __int128 bw = 0;
    for (int i = 0; i < NW; i++) {
        unsigned __int128 d = (unsigned __int128)u[i] - v[i] - (u64)bw;
        u[i] = (u64)d;
        bw = (d >> 64) & 1;
    }
    return (int)bw;
}

/* u += v; returns the carry out of bit wd (operands < 2^wd);
 * masks u back to wd bits. */
static inline int add_mw(u64 *u, const u64 *v, int wd) {
    unsigned __int128 c = 0;
    for (int i = 0; i < NW; i++) {
        unsigned __int128 s = (unsigned __int128)u[i] + v[i] + (u64)c;
        u[i] = (u64)s;
        c = s >> 64;
    }
    int cw = wd >> 6, cb = wd & 63;
    int carry = (cw < NW) ? (int)((u[cw] >> cb) & 1) : (int)c;
    mask_mw(u, wd);
    return carry;
}

/* out = wd bits of reg from 1-based position pos */
static inline void extract(const u64 *reg, int pos, int wd, u64 *out) {
    memcpy(out, reg, sizeof(u64) * NW);
    shr_mw(out, pos - 1);
    mask_mw(out, wd);
}

/* write the low wd bits of val into reg at 1-based position pos */
static inline void deposit(u64 *reg, int pos, int wd, const u64 *val) {
    u64 m[NW], v[NW];
    for (int i = 0; i < NW; i++) m[i] = ~0ULL;
    mask_mw(m, wd);
    memcpy(v, val, sizeof(v));
    shl_mw(m, pos - 1);
    shl_mw(v, pos - 1);
    for (int i = 0; i < NW; i++) reg[i] = (reg[i] & ~m[i]) | v[i];
}

/* reverse a wd-bit value in place */
static inline void rev_val(u64 *v, int wd) {
    u64 t[NW];
    for (int i = 0; i < NW; i++) t[i] = rev64(v[NW - 1 - i]);
    shr_mw(t, 64 * NW - wd);
    memcpy(v, t, sizeof(t));
}

/* (reg & ((1 << B) - 1)).bit_length() */
static inline int bitlen_masked(const u64 *reg, int B) {
    u64 t[NW];
    memcpy(t, reg, sizeof(t));
    mask_mw(t, B);
    for (int i = NW - 1; i >= 0; i--)
        if (t[i]) return 64 * i + 64 - __builtin_clzll(t[i]);
    return 0;
}

/* lowest set 1-based position >= A; 0 if none */
static inline int lowest_set_geq(const u64 *reg, int A) {
    u64 t[NW];
    memcpy(t, reg, sizeof(t));
    shr_mw(t, A - 1);
    for (int i = 0; i < NW; i++)
        if (t[i]) return A + 64 * i + __builtin_ctzll(t[i]);
    return 0;
}

/* value >> 1 with bit 1 wrapped to bit W (the model's rot_left) */
static inline void rot_a(u64 *v) {
    int b0 = (int)(v[0] & 1);
    shr_mw(v, 1);
    if (b0) flipb(v, W);
}

/* value << 2 with the top two bits wrapped to the bottom */
static inline void rot_b(u64 *v) {
    int hi1 = getb(v, W - 1), hi2 = getb(v, W);
    shl_mw(v, 2);
    mask_mw(v, W);
    if (hi1) flipb(v, 1);
    if (hi2) flipb(v, 2);
}

/* ------------------- one Algorithm-3 step ------------------- */

/* Sparse needs log: step appends (stat, t, value) for every nonzero
 * need it would have written to the dense stat-major array.  Zero
 * values are dropped — every consumer (hist / devs / hold / the dense
 * reconstruction, which starts from a zeroed array) skips them. */
typedef struct { int16_t s, t, v; } Need;
#define LOGCAP (NMAX_CAP * 9)      /* max 9 writes per step */

#define ND(sid, val) do { \
        int16_t _v = (int16_t)(val); \
        if (_v) { lg[*pn].s = (sid); lg[*pn].t = (int16_t)i; \
                  lg[*pn].v = _v; (*pn)++; } \
    } while (0)

/* Verbatim transliteration of FastEEA.step (gen_constants.py); the
 * model-level asserts return T instead of raising. */
static int step(State *st, int T, Need *lg, int *pn) {
    int i = T - 1;
    int alive = st->lrp != 0;
    int p1 = st->p1, p2 = st->p2;
    u64 ub[NW], vb[NW];

    /* pre-shift */
    if (p1 == 0) {
        if (alive) rot_a(st->w2);
        st->ls += 1;
        if (p2 == 1) {
            if (alive) rot_b(st->w2);
            st->ls -= 2;
        }
    }
    if (st->ls < 0) return T;

    /* location-controlled ops on r's */
    if (alive && p1 == 0) {
        int Lp = st->lt + st->lq + 2;
        int R = W - st->ls;
        ND(S_RK, Lp);
        ND(S_RKK, R);
        int wd = R - Lp + 1;
        if (Lp <= R) {
            extract(st->w1, Lp, wd, ub); rev_val(ub, wd);
            extract(st->w2, Lp, wd, vb); rev_val(vb, wd);
            int borrow = sub_mw(ub, vb);
            mask_mw(ub, wd);
            rev_val(ub, wd);
            deposit(st->w1, Lp, wd, ub);
            st->sign ^= borrow;
        }
        if (p2 == 1) st->sign ^= 1;
        if (p2 == 0 || st->sign == 0) {
            if (Lp <= R) {
                extract(st->w1, Lp, wd, ub); rev_val(ub, wd);
                extract(st->w2, Lp, wd, vb); rev_val(vb, wd);
                add_mw(ub, vb, wd);
                rev_val(ub, wd);
                deposit(st->w1, Lp, wd, ub);
            }
        }
    }

    /* location-controlled swap */
    if ((p1 ^ p2) == 1) {
        int J = st->lt + st->lq + 1 + (p1 == 0 ? 1 : 0);
        ND(S_SWK, J);
        ND(S_SWKK, J);
        int bJ = getb(st->w1, J);
        if (st->sign != bJ) flipb(st->w1, J);
        st->sign = bJ;
        st->lq += (p1 == 1) ? -1 : 1;
        if (st->lq < 0) return T;
    }

    /* location-controlled ops on t's + post-shift */
    if (p1 == 1) {
        int R = (p2 == 0) ? st->lt + 1 : W - st->lrp - st->ls;
        ND(S_TK, R);
        if (R < 0) return T;
        if (p2 == 1 || st->sign == 0) {
            extract(st->w2, 1, R, ub);
            extract(st->w1, 1, R, vb);
            sub_mw(ub, vb);
            mask_mw(ub, R);
            deposit(st->w2, 1, R, ub);
        }
        st->sign ^= 1;
        extract(st->w2, 1, R, ub);
        extract(st->w1, 1, R, vb);
        int carry = add_mw(ub, vb, R);
        st->sign ^= carry;
        deposit(st->w2, 1, R, ub);
        rot_a(st->w2);
        st->ls += 1;
        if (p2 == 1) {
            rot_b(st->w2);
            st->ls -= 2;
        }
    }
    if (st->ls < 0) return T;

    /* phase update */
    if (st->lq == 0 && st->lrp > 0) {
        st->p2 ^= st->sign ^ st->p1;
        st->sign ^= st->p2;
    }
    if (st->ls == 0) {
        st->p1 ^= 1;
        st->p2 ^= 1;
    }

    /* end-of-iteration SWAP + length updates (4|T only) */
    if (T % 4 == 0 && st->lq == 0 && st->ls == 0) {
        u64 tmp[NW];
        memcpy(tmp, st->w1, sizeof(tmp));
        memcpy(st->w1, st->w2, sizeof(tmp));
        memcpy(st->w2, tmp, sizeof(tmp));
        int B = W - st->lrp;
        int old_lt = bitlen_masked(st->w2, B);
        int new_lt = bitlen_masked(st->w1, B);
        if (old_lt != st->lt) return T;
        ND(S_LTK, new_lt ? (new_lt < st->lt ? new_lt : st->lt) : st->lt);
        ND(S_LTKK, B);
        int lrp_old = st->lrp;
        st->lt = new_lt;
        int A = st->lt + 2;
        int m_old = lowest_set_geq(st->w1, A);
        int old_r = m_old ? (N + 4 - m_old) : 0;
        if (old_r != lrp_old) return T;
        ND(S_LRK, A);
        if (lrp_old) ND(S_LRKK, W - lrp_old);
        int m_new = lowest_set_geq(st->w2, A);
        int new_r = m_new ? (N + 4 - m_new) : 0;
        if (new_r) ND(S_NEWK, m_new);
        st->lrp = new_r;
    }
    return 0;
}

/* full walk; returns 0, or the step T where a model invariant broke.
 * Appends every nonzero need to lg (capacity LOGCAP); *pn is the
 * running entry count. */
static int run_lane(const u64 *w1i, const u64 *w2i, int lrp0, int nmax,
                    Need *lg, int *pn, State *out) {
    State st;
    memcpy(st.w1, w1i, sizeof(st.w1));
    memcpy(st.w2, w2i, sizeof(st.w2));
    st.lt = 1; st.lq = 0; st.ls = 0; st.lrp = lrp0;
    st.p1 = st.p2 = st.sign = 0;
    *pn = 0;
    for (int T = 1; T <= nmax; T++) {
        int e = step(&st, T, lg, pn);
        if (e) return e;
    }
    if (out) *out = st;
    return 0;
}

/* ------------------- exported entry points ------------------- */

/* One lane: needs (NSTATS x nmax int16) + optional final state
 * (5 w1 words, 5 w2 words, lt, lq, ls, lrp, p1, p2, sign as int64).
 * The dense array is reconstructed from the sparse log, with zeros
 * where nothing was recorded. */
int fasteea_run(const u64 *w1i, const u64 *w2i, int lrp0, int nmax,
                int16_t *needs, int64_t *state_out) {
    if (nmax > NMAX_CAP) return -1;
    Need lg[LOGCAP];
    int nlg;
    State st;
    int e = run_lane(w1i, w2i, lrp0, nmax, lg, &nlg, &st);
    if (e) return e;
    memset(needs, 0, sizeof(int16_t) * NSTATS * (size_t)nmax);
    for (int j = 0; j < nlg; j++)
        needs[(size_t)lg[j].s * nmax + lg[j].t] = lg[j].v;
    if (state_out) {
        memcpy(state_out, st.w1, sizeof(st.w1));
        memcpy(state_out + NW, st.w2, sizeof(st.w2));
        state_out[2 * NW + 0] = st.lt;
        state_out[2 * NW + 1] = st.lq;
        state_out[2 * NW + 2] = st.ls;
        state_out[2 * NW + 3] = st.lrp;
        state_out[2 * NW + 4] = st.p1;
        state_out[2 * NW + 5] = st.p2;
        state_out[2 * NW + 6] = st.sign;
    }
    return 0;
}

/* Base pass: H[s][T][v] += 1 for every nonzero need v.  Needs are
 * bounded by W < NEEDMAX, checked. */
int fasteea_hist(const u64 *w1i, const u64 *w2s, const int32_t *lrps,
                 int L, int nmax, uint32_t *H) {
    if (nmax > NMAX_CAP) return -1;
    Need lg[LOGCAP];
    int nlg;
    for (int k = 0; k < L; k++) {
        int e = run_lane(w1i, w2s + (size_t)k * NW, lrps[k], nmax, lg,
                         &nlg, 0);
        if (e) return -1000 - e;
        for (int j = 0; j < nlg; j++) {
            int v = lg[j].v;
            if (v < 0 || v >= NEEDMAX) return -2;
            H[((size_t)lg[j].s * nmax + lg[j].t) * NEEDMAX + v]++;
        }
    }
    return 0;
}

/* Dev pass: D[k] = the NSTATS + 3 per-lane worst deviations beyond
 * the base curves (sentinel -1000): the window stats (k-side low /
 * K-side high per kside flag), the t-core low side, and the r-core
 * L-high / R-low sides.  Bases use SENT = -1. */
int fasteea_devs(const u64 *w1i, const u64 *w2s, const int32_t *lrps,
                 int L, int nmax, int cut, const uint8_t *kside,
                 const int16_t *wl_base, const int16_t *t_base,
                 const int16_t *r_baseL, const int16_t *r_baseR,
                 int16_t *D) {
    if (nmax > NMAX_CAP || cut > nmax) return -1;
    Need lg[LOGCAP];
    int nlg;
    for (int k = 0; k < L; k++) {
        int e = run_lane(w1i, w2s + (size_t)k * NW, lrps[k], nmax, lg,
                         &nlg, 0);
        if (e) return -1000 - e;
        int16_t *row = D + (size_t)k * (NSTATS + 3);
        int best[NSTATS + 3];
        for (int s = 0; s < NSTATS + 3; s++) best[s] = -1000;
        for (int j = 0; j < nlg; j++) {
            int t = lg[j].t, s = lg[j].s, v = lg[j].v;
            if (t >= cut) continue;
            int b = wl_base[(size_t)s * nmax + t];
            if (b != -1) {
                int d = kside[s] ? b - v : v - b;
                if (d > best[s]) best[s] = d;
            }
            if (s == S_TK && t_base[t] >= 0
                && t_base[t] - v > best[NSTATS])
                best[NSTATS] = t_base[t] - v;
            if (s == S_RK && r_baseL[t] >= 0
                && v - r_baseL[t] > best[NSTATS + 1])
                best[NSTATS + 1] = v - r_baseL[t];
            if (s == S_RKK && r_baseR[t] >= 0
                && r_baseR[t] - v > best[NSTATS + 2])
                best[NSTATS + 2] = r_baseR[t] - v;
        }
        for (int s = 0; s < NSTATS + 3; s++) row[s] = (int16_t)best[s];
    }
    return 0;
}
