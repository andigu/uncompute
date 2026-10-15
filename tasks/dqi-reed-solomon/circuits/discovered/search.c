/* search.c -- C implementation of the discovered circuit's full
 * (untruncated) Berlekamp-Massey recursion with slot instrumentation,
 * and the weight-law draw scheme.  gen_constants.py runs rs_slots to
 * derive the per-round coefficient windows (IMAXV, KMSV, KMMV).
 *
 * Instance (contract.json; Khattar et al. (2025), arXiv:2510.10967):
 *   GF(2^12), poly 4331 = x^12+x^7+x^6+x^5+x^3+x+1, gamma = 2;
 *   n = NR = 70 syndromes/rounds, w_max = WMAX = 35, m = NPOS = 4095.
 * Every instance literal is a define below; rs_params() exports them so
 * the Python driver can assert it talks to the instance it expects.
 *
 * Entry points:
 *   rs_params:   export the compile-time instance constants.
 *   rs_slots:    draw errors from the grader weight law and sum the
 *                full run's per-(round, slot) mac/swap hit counters and
 *                need-level histogram.
 *
 * Draw scheme (NOT the CPython choices/sample stream -- a
 * self-contained scheme, equal to random.Random(seed).getrandbits(k)
 * calls in this order): per draw,
 *   w: r16 = mt_next() >> 16 (= getrandbits(16)); w = first k in
 *      0..WMAX with r16 < CUM[k]  (CUM = weight_cum_2p16, 36 entries,
 *      CUM[k] = 0 for k < 19, CUM[WMAX] = 2^16, so w lands in 19..35)
 *   then for j = 0..w-1, interleaved:
 *     pos_j: rejection draws of getrandbits(12) = mt_next() >> 20
 *            (reject >= 4095 or already used) until distinct
 *     val_j: getrandbits(12), reject >= 4095, + 1  (1..4095)
 * MT19937 seeded exactly like CPython's random.Random(int): init_by_array
 * over the seed's 32-bit little-endian limbs.
 *
 * d counter: the model bounds d to the 8-bit two's-complement range
 * [-128, 127] (DLIM).  n = 70 lets d reach 69 on the zero-error anchor
 * (every round ctl = 0, d += 1 through r = 68), so the circuit's dW is
 * 8 bits with the sign at dW[7].
 *
 * Build: cc -O2 -shared -fPIC -o libsearch.dylib search.c
 * (gen_constants.py does this on demand).
 */
#include <stdint.h>
#include <string.h>

/* Standard MT19937 (init_by_array seeding, matching the
 * documented draw scheme). */
/* ------------------------- MT19937 ------------------------- */
#define MTN 624

typedef struct { uint32_t mt[MTN]; int mti; } MT;

static void mt_init_genrand(MT *m, uint32_t s) {
    m->mt[0] = s;
    for (int i = 1; i < MTN; i++)
        m->mt[i] = 1812433253u * (m->mt[i-1] ^ (m->mt[i-1] >> 30))
                   + (uint32_t)i;
    m->mti = MTN;
}

static void mt_init_by_array(MT *m, const uint32_t *key, int klen) {
    mt_init_genrand(m, 19650218u);
    int i = 1, j = 0;
    int k = MTN > klen ? MTN : klen;
    for (; k; k--) {
        m->mt[i] = (m->mt[i] ^ ((m->mt[i-1] ^ (m->mt[i-1] >> 30))
                    * 1664525u)) + key[j] + (uint32_t)j;
        i++; j++;
        if (i >= MTN) { m->mt[0] = m->mt[MTN-1]; i = 1; }
        if (j >= klen) j = 0;
    }
    for (k = MTN - 1; k; k--) {
        m->mt[i] = (m->mt[i] ^ ((m->mt[i-1] ^ (m->mt[i-1] >> 30))
                    * 1566083941u)) - (uint32_t)i;
        i++;
        if (i >= MTN) { m->mt[0] = m->mt[MTN-1]; i = 1; }
    }
    m->mt[0] = 0x80000000u;
    m->mti = MTN;
}

static uint32_t mt_next(MT *m) {
    uint32_t y;
    if (m->mti >= MTN) {
        static const uint32_t mag[2] = {0, 0x9908b0dfu};
        int kk;
        for (kk = 0; kk < MTN - 397; kk++) {
            y = (m->mt[kk] & 0x80000000u) | (m->mt[kk+1] & 0x7fffffffu);
            m->mt[kk] = m->mt[kk+397] ^ (y >> 1) ^ mag[y & 1];
        }
        for (; kk < MTN - 1; kk++) {
            y = (m->mt[kk] & 0x80000000u) | (m->mt[kk+1] & 0x7fffffffu);
            m->mt[kk] = m->mt[kk+(397-MTN)] ^ (y >> 1) ^ mag[y & 1];
        }
        y = (m->mt[MTN-1] & 0x80000000u) | (m->mt[0] & 0x7fffffffu);
        m->mt[MTN-1] = m->mt[396] ^ (y >> 1) ^ mag[y & 1];
        m->mti = 0;
    }
    y = m->mt[m->mti++];
    y ^= y >> 11;
    y ^= (y << 7) & 0x9d2c5680u;
    y ^= (y << 15) & 0xefc60000u;
    y ^= y >> 18;
    return y;
}

/* Random(int): non-negative int seeding */
void cpy_seed_u64(MT *m, uint64_t v) {
    uint32_t key[2];
    int klen;
    if (v >> 32) { key[0] = (uint32_t)v; key[1] = (uint32_t)(v >> 32);
                   klen = 2; }
    else { key[0] = (uint32_t)v; klen = 1; }
    mt_init_by_array(m, key, klen);
}


/* ---------------------- instance constants ---------------------- */
#define GFBITS 12               /* field GF(2^GFBITS) */
#define GFORD 4095              /* multiplicative order 2^GFBITS - 1 */
#define NPOS GFORD              /* error positions j = 0..m-1, m = 4095 */
#define NR 70                   /* BM rounds / syndrome count n */
#define WMAX 35                 /* max error weight = n/2 */
#define NSIG (WMAX + 1)         /* sigma_0..sigma_WMAX */
#define NBANK (NSIG + 1)        /* B bank ring: NSIG slots + the wrap slot
                                 * (rotation wraps slot NSIG into slot 0:
                                 * the I2 event) */
#define NOMM WMAX               /* omega_0..omega_{WMAX-1} */
#define RLAST (NR - 1)          /* the elided-ctl final round */
#define DLIM 128                /* d counter two's-complement half-range */
#define TR_FIELDS 7             /* per-round trace fields */
#define TRACE_LEN (2 + NR * TR_FIELDS + NSIG + NOMM)

void rs_params(int32_t *out) {
    out[0] = GFBITS; out[1] = GFORD; out[2] = NPOS; out[3] = NR;
    out[4] = WMAX; out[5] = NSIG; out[6] = NBANK; out[7] = NOMM;
    out[8] = DLIM; out[9] = TRACE_LEN;
}

typedef struct {
    int16_t exp2[2 * GFORD];   /* EXP duplicated: exp2[i] = EXP[i % GFORD] */
    int16_t log_[GFORD + 1];   /* LOG[0] unused (gmul guards on zero) */
} GF;

static void gf_init(GF *g, const int16_t *exp_tab, const int16_t *log_tab) {
    for (int i = 0; i < 2 * GFORD; i++) g->exp2[i] = exp_tab[i % GFORD];
    memcpy(g->log_, log_tab, sizeof(g->log_));
}

static inline int gmul(const GF *g, int a, int b) {
    if (!a || !b) return 0;
    return g->exp2[g->log_[a] + g->log_[b]];
}

static inline int ginv(const GF *g, int a) {
    return g->exp2[GFORD - g->log_[a]];
}

/* The full (untruncated) run with the slot instrumentation.  Per round
 * r it records, when the pointers are non-NULL:
 *   ni_hist[r*NSIG+h]   — max nonzero sigma index entering round r is h
 *                         (the discrepancy-window need level)
 *   swap_hit[r*NSIG+k]  — ctl = 1 and sig[k] != B'[k] (cswap changes state)
 *   mac_hit[r*NSIG+k]   — the mac term would change sig[k]: m != 0 and
 *                         the (rotated) B slot k != 0
 *   trace[2 + r*TR_FIELDS + 0..6] — raw D (pre-park), ctl, d entering
 *                         the round, m, need level h, swap-hit mask,
 *                         mac-hit mask (bit k = slot k hit)
 * trace[0] = viol, trace[1] = pushes, then sig[NSIG], omm[NOMM] after
 * the rounds.  Returns pushes; *viol is set on any model-invariant
 * break (delc == 0, |d| >= DLIM, sig[0] == 0 -- never fires on legal
 * draws).  Every invariant break returns early (sig/omm then hold
 * zeros).  rs_slots passes trace = NULL. */
static int bm_full_instr(const GF *g, const int16_t *S_, int16_t *sig,
                         int16_t *omm, int64_t *mac_hit, int64_t *swap_hit,
                         int64_t *ni_hist, int64_t *trace, int *viol) {
    int16_t sg[NSIG], bb[NBANK];
    memset(sg, 0, sizeof(sg));
    memset(bb, 0, sizeof(bb));
    memset(sig, 0, NSIG * sizeof(int16_t));
    memset(omm, 0, NOMM * sizeof(int16_t));
    sg[0] = 1;
    bb[0] = 1;
    int delc = 1, d = 0, b0 = 0, pushes = 0;
    if (trace) memset(trace, 0, TRACE_LEN * sizeof(int64_t));
    for (int r = 0; r < NR; r++) {
        b0 = (b0 + NBANK - 1) % NBANK;
        int imax_f = r < WMAX ? r : WMAX;
        int kmax_f = r + 1 < WMAX ? r + 1 : WMAX;
        int hi = 0;
        for (int i = imax_f; i > 0; i--) if (sg[i]) { hi = i; break; }
        if (ni_hist) ni_hist[r * NSIG + hi]++;
        int D = 0;
        for (int i = 0; i <= imax_f; i++)
            D ^= gmul(g, sg[i], S_[r - i]);
        int Draw = D;
        int ctl = (D != 0 && d >= 0) ? 1 : 0;
        if (r == RLAST) ctl = 0;
        if (ctl) { int t = D; D = delc; delc = t; }
        int64_t *tr = trace ? trace + 2 + r * TR_FIELDS : 0;
        if (tr) { tr[0] = Draw; tr[1] = ctl; tr[2] = d; }
        if (delc == 0) { *viol = 1; if (trace) trace[0] = 1; return pushes; }
        int m = gmul(g, D, ginv(g, delc));
        if (tr) { tr[3] = m; tr[4] = hi; }
        uint64_t smask = 0, mmask = 0;
        if (ctl)
            for (int k = 0; k <= kmax_f; k++) {
                int16_t *bp = &bb[(b0 + k) % NBANK];
                if (sg[k] != *bp) {
                    if (swap_hit) swap_hit[r * NSIG + k]++;
                    smask |= (uint64_t)1 << k;
                }
                int t = sg[k];
                sg[k] = *bp;
                *bp = (int16_t)t;
            }
        for (int k = 0; k <= kmax_f; k++) {
            int bv = bb[(b0 + k) % NBANK];
            if (m && bv) {
                if (mac_hit) mac_hit[r * NSIG + k]++;
                mmask |= (uint64_t)1 << k;
            }
            sg[k] ^= (int16_t)gmul(g, m, bv);
        }
        if (tr) { tr[5] = (int64_t)smask; tr[6] = (int64_t)mmask; }
        if (r > 0 && r < RLAST && d < 0) pushes++;
        if (r < RLAST) {
            d = ctl ? -d : d + 1;
            if (d < -DLIM || d >= DLIM) {
                *viol = 1; if (trace) trace[0] = 1; return pushes;
            }
        }
    }
    if (trace) trace[1] = pushes;
    if (sg[0] == 0) { *viol = 1; if (trace) trace[0] = 1; return pushes; }
    memcpy(sig, sg, NSIG * sizeof(int16_t));
    for (int i = 0; i < NOMM; i++) {
        int acc = 0;
        for (int j2 = 0; j2 <= i; j2++)
            acc ^= gmul(g, sg[j2], S_[i - j2]);
        omm[i] = (int16_t)acc;
    }
    if (trace) {
        int64_t *t2 = trace + 2 + NR * TR_FIELDS;
        for (int i = 0; i < NSIG; i++) t2[i] = sig[i];
        for (int i = 0; i < NOMM; i++) t2[NSIG + i] = omm[i];
    }
    return pushes;
}

/* One draw (scheme in the header): fills pos/val[w], returns
 * w, and evaluates the syndromes S_[i-1] = sum_j v_j * gamma^(i*pos_j),
 * i = 1..NR, by stepping each term's log by pos_j per syndrome. */
static int draw_synd(const GF *g, MT *mt, const uint32_t *cum_law,
                     int16_t *pos, int16_t *val, int16_t *S_) {
    uint32_t r16 = mt_next(mt) >> 16;
    int w = 0;
    while (cum_law[w] <= r16) w++;
    uint8_t seen[NPOS];
    memset(seen, 0, sizeof(seen));
    for (int j = 0; j < w; j++) {
        uint32_t p;
        do { p = mt_next(mt) >> (32 - GFBITS); } while (p >= NPOS || seen[p]);
        seen[p] = 1;
        pos[j] = (int16_t)p;
        uint32_t v;
        do { v = mt_next(mt) >> (32 - GFBITS); } while (v >= GFORD);
        val[j] = (int16_t)(v + 1);
    }
    int e[WMAX];
    for (int j = 0; j < w; j++) e[j] = g->log_[val[j]];
    for (int i = 1; i <= NR; i++) {
        int acc = 0;
        for (int j = 0; j < w; j++) {
            e[j] += pos[j];
            if (e[j] >= GFORD) e[j] -= GFORD;
            acc ^= g->exp2[e[j]];
        }
        S_[i - 1] = (int16_t)acc;
    }
    return w;
}

/* ---- slots mode: the window derivation instrument ----
 * Full-run per-(round, slot) counters over C-drawn law errors
 * (mac_hit / swap_hit / ni_hist, NR*NSIG each, semantics at
 * bm_full_instr), summed into the caller's arrays.
 * out: [0] draws [1] viol. */
void rs_slots(const int16_t *exp_tab, const int16_t *log_tab,
              const uint32_t *cum_law, uint64_t seed, int64_t draws,
              int64_t *out, int64_t *mac_hit, int64_t *swap_hit,
              int64_t *ni_hist) {
    GF g;
    gf_init(&g, exp_tab, log_tab);
    MT mt;
    cpy_seed_u64(&mt, seed);
    int64_t viol_n = 0;
    int16_t pos[WMAX], val[WMAX], S_[NR];
    int16_t sig_f[NSIG], omm_f[NOMM];
    for (int64_t dr = 0; dr < draws; dr++) {
        draw_synd(&g, &mt, cum_law, pos, val, S_);
        int viol = 0;
        bm_full_instr(&g, S_, sig_f, omm_f, mac_hit, swap_hit, ni_hist, 0,
                      &viol);
        if (viol) viol_n++;
    }
    out[0] = draws; out[1] = viol_n;
}
