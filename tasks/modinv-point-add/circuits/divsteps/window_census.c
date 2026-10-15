/* window_census.c — the hd-divstep width and R-fill census sampler.
 *
 * Implements exactly the iteration of the normative Python
 * trajectory models (gen_constants.py py_walk_widths /
 * py_fill_walk_j) over signed 320-bit two's-complement limbs, and
 * records per-step per-row |value| bit-length HISTOGRAMS (rows F, G,
 * FV, GV; frozen lanes hold their stop-state widths) plus the tstop
 * histogram, and per-step R-stack entry-fill histograms. Trajectory-
 * dump entry points expose the per-step widths and fills of a GIVEN
 * input for bit-exact validation against the Python models, and a
 * fill holdout entry point counts escapes under candidate reach
 * tables (the cap sweep).
 *
 * Build (gen_constants.py compiles this on demand; dlopen'd via
 * ctypes):
 *   clang -O3 -shared -fPIC -o window_census.so window_census.c
 */
#include <stdint.h>
#include <string.h>

#define NL 5              /* 5 x 64 = 320-bit two's complement */
#define SMAX 590
#define ROWS 4
#define WBINS 320

typedef struct { uint64_t l[NL]; } u320;

static inline void set_zero(u320 *a) { memset(a->l, 0, sizeof a->l); }

static inline void add320(u320 *r, const u320 *a, const u320 *b) {
    unsigned __int128 c = 0;
    for (int i = 0; i < NL; i++) {
        c += (unsigned __int128)a->l[i] + b->l[i];
        r->l[i] = (uint64_t)c;
        c >>= 64;
    }
}

static inline void sub320(u320 *r, const u320 *a, const u320 *b) {
    __int128 c = 0;
    for (int i = 0; i < NL; i++) {
        c += (__int128)a->l[i] - b->l[i];
        r->l[i] = (uint64_t)c;
        c >>= 64;             /* arithmetic: borrow propagates */
    }
}

static inline void shr1(u320 *a) {  /* arithmetic >> 1 */
    for (int i = 0; i < NL - 1; i++)
        a->l[i] = (a->l[i] >> 1) | (a->l[i + 1] << 63);
    a->l[NL - 1] = (uint64_t)(((int64_t)a->l[NL - 1]) >> 1);
}

static inline void shl1(u320 *a) {
    for (int i = NL - 1; i > 0; i--)
        a->l[i] = (a->l[i] << 1) | (a->l[i - 1] >> 63);
    a->l[0] <<= 1;
}

static inline int is_zero(const u320 *a) {
    uint64_t o = 0;
    for (int i = 0; i < NL; i++) o |= a->l[i];
    return o == 0;
}

static inline int bitlen_abs(const u320 *a) {
    u320 t = *a;
    if ((int64_t)t.l[NL - 1] < 0) {       /* negate two's complement */
        unsigned __int128 c = 1;
        for (int i = 0; i < NL; i++) {
            c += (unsigned __int128)(~t.l[i]);
            t.l[i] = (uint64_t)c;
            c >>= 64;
        }
    }
    for (int i = NL - 1; i >= 0; i--)
        if (t.l[i])
            return 64 * i + 64 - __builtin_clzll(t.l[i]);
    return 0;
}

/* xoshiro256** */
typedef struct { uint64_t s[4]; } rng_t;

static inline uint64_t rotl(uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}

static uint64_t rng_next(rng_t *r) {
    uint64_t *s = r->s;
    uint64_t res = rotl(s[1] * 5, 7) * 9;
    uint64_t t = s[1] << 17;
    s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3];
    s[2] ^= t; s[3] = rotl(s[3], 45);
    return res;
}

static void rng_seed(rng_t *r, uint64_t seed) {
    /* splitmix64 expansion */
    uint64_t z = seed;
    for (int i = 0; i < 4; i++) {
        z += 0x9e3779b97f4a7c15ULL;
        uint64_t w = z;
        w = (w ^ (w >> 30)) * 0xbf58476d1ce4e5b9ULL;
        w = (w ^ (w >> 27)) * 0x94d9736c719c2dc5ULL;
        r->s[i] = w ^ (w >> 31);
    }
}

/* p and comparison for rejection sampling (p passed as 4 limbs) */
static int lt256(const uint64_t a[4], const uint64_t b[4]) {
    for (int i = 3; i >= 0; i--) {
        if (a[i] < b[i]) return 1;
        if (a[i] > b[i]) return 0;
    }
    return 0;
}

/* one walk; returns tstop (steps executed until g == 0), records
 * per-step widths into w[SMAX][ROWS] (frozen-hold for t >= tstop:
 * F=1, G=0, FV/GV held). S = schedule length to record. */
static int walk(const uint64_t p[4], const u320 *x0, int S,
                uint16_t w[][ROWS]) {
    u320 f, g, fv, gv, tmp;
    set_zero(&f);
    memcpy(f.l, p, 4 * sizeof(uint64_t));
    g = *x0;
    set_zero(&fv);
    set_zero(&gv);
    gv.l[0] = 1;
    int64_t z = 0;
    int t = 0;
    for (; t < S; t++) {
        int s = (z >= 0) && (g.l[0] & 1);
        if (s) {
            z = -z - 1;
            tmp = f; f = g; g = tmp;
            tmp = fv; fv = gv; gv = tmp;
        }
        if (g.l[0] & 1) {
            if (s) { sub320(&g, &f, &g); sub320(&gv, &fv, &gv); }
            else   { add320(&g, &g, &f); add320(&gv, &gv, &fv); }
        }
        z += 1;
        shr1(&g);
        shl1(&fv);
        w[t][0] = (uint16_t)bitlen_abs(&f);
        w[t][1] = (uint16_t)bitlen_abs(&g);
        w[t][2] = (uint16_t)bitlen_abs(&fv);
        w[t][3] = (uint16_t)bitlen_abs(&gv);
        if (is_zero(&g)) {
            uint16_t bfv = w[t][2], bgv = w[t][3];
            for (int tt = t + 1; tt < S; tt++) {
                w[tt][0] = 1;
                w[tt][1] = 0;
                w[tt][2] = bfv;
                w[tt][3] = bgv;
            }
            return t + 1;
        }
    }
    return S + 1;   /* did not freeze inside the recorded schedule */
}

/* census: accumulate per-step per-row width histograms + tstop hist.
 * hist layout: [S][ROWS][WBINS] uint64; tstop_hist: [SMAX+2]. */
void run_census(const uint64_t p[4], uint64_t seed, uint64_t nwalks,
                int S, uint64_t *hist, uint64_t *tstop_hist) {
    rng_t r;
    rng_seed(&r, seed);
    static _Thread_local uint16_t w[SMAX][ROWS];
    for (uint64_t k = 0; k < nwalks; k++) {
        u320 x;
        set_zero(&x);
        do {
            for (int i = 0; i < 4; i++) x.l[i] = rng_next(&r);
        } while (!lt256(x.l, p) ||
                 (x.l[0] | x.l[1] | x.l[2] | x.l[3]) == 0);
        int ts = walk(p, &x, S, w);
        tstop_hist[ts <= SMAX + 1 ? ts : SMAX + 1] += 1;
        for (int t = 0; t < S; t++)
            for (int rr = 0; rr < ROWS; rr++)
                hist[((uint64_t)t * ROWS + rr) * WBINS + w[t][rr]] += 1;
    }
}

/* trajectory dump for bit-exact validation: widths of a GIVEN x. */
int walk_widths(const uint64_t p[4], const uint64_t x[4], int S,
                uint16_t *out /* S*ROWS */) {
    u320 xx;
    set_zero(&xx);
    memcpy(xx.l, x, 4 * sizeof(uint64_t));
    static uint16_t w[SMAX][ROWS];
    int ts = walk(p, &xx, S, w);
    memcpy(out, w, (size_t)S * ROWS * 2);
    return ts;
}

/* ----- R-fill / z-width census (the py_fill_walk_j model:
 * ENTRY fill before the step's push, frozen-inclusive, freeze-push-
 * suppressed, push on class-3 steps with v2(fv') >= 5; z tracked as
 * the machine's two's-complement counter — per-walk max width). */

#define FILL_MAX 96
#define ZW_MAX 24

static inline int v2_320(const u320 *a) {   /* -1 for zero */
    for (int i = 0; i < NL; i++)
        if (a->l[i])
            return 64 * i + __builtin_ctzll(a->l[i]);
    return -1;
}

static inline int zwidth(int64_t z) {       /* min two's-compl. width */
    uint64_t m = z >= 0 ? (uint64_t)z : (uint64_t)(-z - 1);
    int bl = m ? 64 - __builtin_clzll(m) : 0;
    return bl + 1;
}

/* one fill walk: per-step ENTRY fill into fills[S]; returns the
 * per-walk max z width (post-increment values; the sigma negate is
 * width-closed in two's complement). jmin = the store threshold on
 * v2(fv') (5 = the deployed j34-decode variant; deeper decode layers
 * raise it). */
static int fill_walk_j(const uint64_t p[4], const u320 *x0, int S,
                       uint16_t fills[], int jmin) {
    u320 f, g, fv, gv, tmp;
    set_zero(&f);
    memcpy(f.l, p, 4 * sizeof(uint64_t));
    g = *x0;
    set_zero(&fv);
    set_zero(&gv);
    gv.l[0] = 1;
    int64_t z = 0;
    int R = 0, zw = 1;
    for (int t = 0; t < S; t++) {
        fills[t] = (uint16_t)R;
        if (is_zero(&g))
            continue;                        /* frozen: fill holds */
        int s = (z >= 0) && (g.l[0] & 1);
        if (s) {
            z = -z - 1;
            tmp = f; f = g; g = tmp;
            tmp = fv; fv = gv; gv = tmp;
        }
        if (g.l[0] & 1) {
            if (s) { sub320(&g, &f, &g); sub320(&gv, &fv, &gv); }
            else   { add320(&g, &g, &f); add320(&gv, &gv, &fv); }
        }
        z += 1;
        if (zwidth(z) > zw) zw = zwidth(z);
        shr1(&g);
        shl1(&fv);
        if (!s && z <= 0 && !is_zero(&g) && v2_320(&fv) >= jmin)
            R += 1;
    }
    return zw;
}

/* ----- census, holdout and trajectory dump at store threshold jmin */

void run_fill_census_j(const uint64_t p[4], uint64_t seed,
                       uint64_t nwalks, int S, uint64_t *fill_hist,
                       uint64_t *zw_hist, int jmin) {
    rng_t r;
    rng_seed(&r, seed);
    static _Thread_local uint16_t fills[SMAX];
    for (uint64_t k = 0; k < nwalks; k++) {
        u320 x;
        set_zero(&x);
        do {
            for (int i = 0; i < 4; i++) x.l[i] = rng_next(&r);
        } while (!lt256(x.l, p) ||
                 (x.l[0] | x.l[1] | x.l[2] | x.l[3]) == 0);
        int zw = fill_walk_j(p, &x, S, fills, jmin);
        zw_hist[zw < ZW_MAX ? zw : ZW_MAX - 1] += 1;
        for (int t = 0; t < S; t++) {
            int fl = fills[t] < FILL_MAX ? fills[t] : FILL_MAX - 1;
            fill_hist[(uint64_t)t * FILL_MAX + fl] += 1;
        }
    }
}

uint64_t run_fill_holdout_j(const uint64_t p[4], uint64_t seed,
                            uint64_t nwalks, int S,
                            const uint16_t *reach,
                            const uint16_t *ref, int jmin) {
    rng_t r;
    rng_seed(&r, seed);
    static _Thread_local uint16_t fills[SMAX];
    uint64_t exceed = 0;
    for (uint64_t k = 0; k < nwalks; k++) {
        u320 x;
        set_zero(&x);
        do {
            for (int i = 0; i < 4; i++) x.l[i] = rng_next(&r);
        } while (!lt256(x.l, p) ||
                 (x.l[0] | x.l[1] | x.l[2] | x.l[3]) == 0);
        fill_walk_j(p, &x, S, fills, jmin);
        int new_esc = 0, old_esc = 0;
        for (int t = 0; t < S; t++) {
            if (fills[t] > reach[t]) new_esc = 1;
            if (fills[t] > ref[t]) old_esc = 1;
        }
        if (new_esc && !old_esc)
            exceed++;
    }
    return exceed;
}

int fill_walk_trace_j(const uint64_t p[4], const uint64_t x[4], int S,
                      uint16_t *out /* S */, int jmin) {
    u320 xx;
    set_zero(&xx);
    memcpy(xx.l, x, 4 * sizeof(uint64_t));
    static uint16_t fills[SMAX];
    int zw = fill_walk_j(p, &xx, S, fills, jmin);
    memcpy(out, fills, (size_t)S * 2);
    return zw;
}
