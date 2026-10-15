/* uncompute C kernel: the fast executor of the evaluator.
 *
 * The normative reference is the Python evaluator
 * (uncompute/evalcore/compact_reader.py, protocol.py, and the Simulator in
 * uncompute/sim.py). This file produces the same state, counters, verdict
 * inputs, and failure text for every stream.
 *
 * Design:
 *   - Every qubit, outcome id, and bit id is an array of n_words u64 lane
 *     words, n_words = ceil(max(n_shots, 1) / 64); word w bit j = lane
 *     64w + j. The sign vector ("phase") is one such array. The three
 *     address spaces are disjoint, each with its own census cap.
 *   - Measurement RNG in block-counter mode: the k-th HMR or R (stream
 *     order) reads bytes [lane0/8, lane0/8 + ceil(batch/8)) of counter k's
 *     stream, whose 128-byte block i is
 *         SHAKE256(seed_rng32 || u64le(k) || u64le(i)),
 *     little-endian and masked to live lanes. The cost is O(1) in lane0
 *     (kern_set_lane0; sharded evaluation). SHAKE256 / Keccak-f[1600] is
 *     implemented below.
 *   - Billing: Toffoli = CCX/CCZ, Clifford = CX/CZ/SWAP/R/HMR; X/Z/NEG and
 *     the bit-only kinds are free. Every op bills popcount(live) at its cost
 *     class (COST_PLAIN, or COST_WITH_CC under a bit condition). A
 *     condition narrows the op's action mask, never `live`, so the counts
 *     are constants of the stream.
 *   - Bulk register I/O (kern_load_register / kern_read_register) moves
 *     per-shot register values to and from lane columns: Qubit registers
 *     in qubit space, Bit registers in bit space. The per-shot semantics
 *     are util.py _set_register / _get_register.
 *   - Input: the op stream, interpreted directly by the kern_compact_* API
 *     below. The kernel assumes a little-endian host (arm64, x86_64).
 *
 * Build: make kernel
 */

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define NO_ID UINT64_MAX          /* NO_QUBIT == NO_OUT == NO_REG == NO_BIT */
#define MAX_QUBIT_ADDR (1ULL << 20)
#define MAX_OUT_ADDR   (1ULL << 24)
#define MAX_BIT_ADDR   (1ULL << 24)

/* ABI handshake: kernel.py checks this value before it binds any other
 * symbol. Any change to an exported signature or record shape increments
 * it, so a stale library fails the load cleanly instead of misreading
 * arguments during a run. */
#define UNCOMPUTE_KERNEL_ABI 1

long uncompute_kernel_abi_version(void) { return UNCOMPUTE_KERNEL_ABI; }

/* error codes (exec_op) */
#define ERR_OOM          5

/* op kinds (uncompute.ops.OpKind; the stream bytes carry these values).
 * K_COUNT is the number of kinds: a kind byte >= K_COUNT is unknown. */
enum {
    K_X = 0, K_Z = 1, K_NEG = 2, K_CX = 3, K_CZ = 4, K_SWAP = 5,
    K_CCX = 6, K_CCZ = 7, K_R = 8, K_HMR = 9, K_OXOR = 10,
    K_BIT_INVERT = 11, K_BIT_STORE0 = 12, K_BIT_STORE1 = 13, K_COUNT = 14
};

/* Cost classes. COST_PLAIN is the
 * class of an op without a bit condition; COST_WITH_CC the class of an op
 * that carries one, which is the class of its promoted kind
 * (uncompute.ops.promoted_kind). A kind that cannot carry a bit condition
 * never reaches the COST_WITH_CC row: the parser rejects the flag. */
enum { COST_FREE = 0, COST_CLIFFORD = 1, COST_TOFFOLI = 2 };

static const uint8_t COST_PLAIN[K_COUNT] = {
    [K_CX] = COST_CLIFFORD, [K_CZ] = COST_CLIFFORD, [K_SWAP] = COST_CLIFFORD,
    [K_R] = COST_CLIFFORD, [K_HMR] = COST_CLIFFORD,
    [K_CCX] = COST_TOFFOLI, [K_CCZ] = COST_TOFFOLI,
};

static const uint8_t COST_WITH_CC[K_COUNT] = {
    [K_X] = COST_CLIFFORD,   /* billed as CX  */
    [K_Z] = COST_CLIFFORD,   /* billed as CZ  */
    [K_CX] = COST_TOFFOLI,   /* billed as CCX */
    [K_CZ] = COST_TOFFOLI,   /* billed as CCZ */
    /* NEG bills as Z and the bit-only kinds as themselves: all free */
};

/* ---------------- SHAKE256 (Keccak-f[1600], FIPS 202) ----------------
 *
 * The permutation follows FIPS 202, Section 3. The state is a 5x5 array of
 * 64-bit lanes. Lane A[x][y] holds state bits 64(5y + x) to 64(5y + x) + 63
 * (Section 3.1.2), so it is word LANE(x, y) = x + 5y of st[]. The first 17
 * words form the 136-byte SHAKE256 rate. */

#define LANE(x, y) ((x) + 5 * (y))

/* Rotate left by n, 0 <= n < 64. The masked right shift makes n = 0 the
 * identity without a shift by 64. */
static inline uint64_t rotl64(uint64_t v, unsigned n) {
    return (v << n) | (v >> ((64 - n) & 63));
}

/* keccak_init fills these tables from the definitions in FIPS 202:
 *   keccak_rho[LANE(x, y)]  the rho offset of lane A[x][y] (Algorithm 2)
 *   keccak_pi_src[LANE(x, y)]  the lane that pi moves to A[x][y]: pi sets
 *                           A'[x][y] = A[(x + 3y) mod 5][x] (Algorithm 3)
 *   keccak_rc[i]            the iota round constant of round i (Algorithm 6) */
static unsigned keccak_rho[25];
static unsigned keccak_pi_src[25];
static uint64_t keccak_rc[24];

/* rc(t) of FIPS 202 Algorithm 5. Bit i of R holds R[i]. One step shifts R
 * up by one bit (R = 0 || R), adds the outgoing bit R[8] into R[0], R[4],
 * R[5] and R[6], and truncates R to 8 bits. */
static unsigned keccak_rc_bit(unsigned t) {
    unsigned R = 1, i;
    for (i = 0; i < t % 255; i++) {
        R <<= 1;
        if (R & 0x100)
            R ^= 0x100 | 0x01 | 0x10 | 0x20 | 0x40;
    }
    return R & 1;
}

/* The loader runs this once, before any other kernel code. */
__attribute__((constructor)) static void keccak_init(void) {
    unsigned x, y, t, nx, j, ir;
    /* rho: start at (x, y) = (1, 0); step t gives offset (t+1)(t+2)/2 mod 64
     * and moves to (y, (2x + 3y) mod 5). A[0][0] keeps offset 0. */
    keccak_rho[LANE(0, 0)] = 0;
    x = 1;
    y = 0;
    for (t = 0; t < 24; t++) {
        keccak_rho[LANE(x, y)] = ((t + 1) * (t + 2) / 2) % 64;
        nx = y;
        y = (2 * x + 3 * y) % 5;
        x = nx;
    }
    for (x = 0; x < 5; x++)
        for (y = 0; y < 5; y++)
            keccak_pi_src[LANE(x, y)] = LANE((x + 3 * y) % 5, x);
    /* iota: RC[2^j - 1] = rc(j + 7 i_r) for j = 0..6; other bits are 0. */
    for (ir = 0; ir < 24; ir++) {
        keccak_rc[ir] = 0;
        for (j = 0; j <= 6; j++)
            keccak_rc[ir] |= (uint64_t)keccak_rc_bit(j + 7 * ir) << ((1u << j) - 1);
    }
}

/* Keccak-f[1600]: 24 rounds of theta, rho, pi, chi, iota (Algorithm 7). */
static void keccakf(uint64_t st[25]) {
    uint64_t C[5], D[5], B[25];
    unsigned ir, x, y, i;
    for (ir = 0; ir < 24; ir++) {
        /* theta: C[x] is the parity of column x;
         * D[x] = C[x-1] ^ rot(C[x+1], 1) is added to every lane of column x. */
        for (x = 0; x < 5; x++)
            C[x] = st[LANE(x, 0)] ^ st[LANE(x, 1)] ^ st[LANE(x, 2)]
                 ^ st[LANE(x, 3)] ^ st[LANE(x, 4)];
        for (x = 0; x < 5; x++) {
            D[x] = C[(x + 4) % 5] ^ rotl64(C[(x + 1) % 5], 1);
            for (y = 0; y < 5; y++)
                st[LANE(x, y)] ^= D[x];
        }
        /* rho, then pi: B[x][y] = rho(A)[(x + 3y) mod 5][x] */
        for (i = 0; i < 25; i++)
            B[i] = rotl64(st[keccak_pi_src[i]], keccak_rho[keccak_pi_src[i]]);
        /* chi: A[x][y] = B[x][y] ^ (~B[x+1][y] & B[x+2][y]) */
        for (y = 0; y < 5; y++)
            for (x = 0; x < 5; x++)
                st[LANE(x, y)] = B[LANE(x, y)]
                    ^ (~B[LANE((x + 1) % 5, y)] & B[LANE((x + 2) % 5, y)]);
        /* iota */
        st[LANE(0, 0)] ^= keccak_rc[ir];
    }
}

#define SHAKE256_RATE 136

/* Squeeze `outlen` bytes of SHAKE256(in) starting at output offset `skip`
 * (the whole-stream prefix before `skip` is computed and discarded — SHAKE
 * output is sequential, so an offset costs the permutations to reach it). */
static void shake256_skip(uint8_t *out, size_t outlen, size_t skip,
                          const uint8_t *in, size_t inlen) {
    uint64_t st[25];
    size_t i;
    memset(st, 0, sizeof st);
    /* absorb */
    while (inlen >= SHAKE256_RATE) {
        for (i = 0; i < SHAKE256_RATE; i++)
            st[i >> 3] ^= (uint64_t)in[i] << (8 * (i & 7));
        keccakf(st);
        in += SHAKE256_RATE;
        inlen -= SHAKE256_RATE;
    }
    for (i = 0; i < inlen; i++)
        st[i >> 3] ^= (uint64_t)in[i] << (8 * (i & 7));
    st[inlen >> 3] ^= (uint64_t)0x1f << (8 * (inlen & 7));
    st[(SHAKE256_RATE - 1) >> 3] ^= (uint64_t)0x80 << (8 * ((SHAKE256_RATE - 1) & 7));
    /* squeeze, discarding the first `skip` bytes */
    while (outlen) {
        keccakf(st);
        if (skip >= SHAKE256_RATE) {
            skip -= SHAKE256_RATE;
            continue;
        }
        {
            size_t avail = SHAKE256_RATE - skip;
            size_t n = outlen < avail ? outlen : avail;
            for (i = 0; i < n; i++)
                out[i] = (uint8_t)(st[(i + skip) >> 3] >> (8 * ((i + skip) & 7)));
            out += n;
            outlen -= n;
            skip = 0;
        }
    }
}

static void shake256(uint8_t *out, size_t outlen, const uint8_t *in, size_t inlen) {
    shake256_skip(out, outlen, 0, in, inlen);
}

/* ---------------- kernel context ---------------- */

/* ---- uncompute.Simulator fast-path state (kern_sim_*; NULL for evaluator
 * contexts, so nothing below this comment affects the evaluator paths).
 * Python's Simulator is the normative reference; this engine executes one
 * apply() chunk between bulk state loads/dumps and HALTS (walk returns the
 * op index) on anything it cannot run with certain bit-identity — the
 * caller then replays the chunk on the pure-Python engine, which raises
 * the normative exception text or executes leniently under
 * validate=False. No error text is composed in C. */

typedef struct {
    int64_t len, cap;
    uint64_t *ids;          /* draw ids this outcome slot carries (GF(2)) */
} SimSlot;

typedef struct {
    uint64_t id;            /* draw id = sim ctr value at the HMR */
    uint64_t refs;          /* number of slots carrying it */
    uint64_t opn, out_id;   /* provenance: HMR op index + outcome id */
    uint64_t *residual;     /* n_words; unresolved phase-kick lanes */
} SimDraw;

typedef struct SimState {
    uint8_t seed[32];       /* b"uncompute-sim" + u64le(seed) (18 bytes) */
    int64_t seed_len;
    uint64_t ctr;           /* sim.py _ctr: pre-incremented per draw */
    uint64_t op_n;          /* cumulative applied-op count (_op_n) */
    int64_t strict;
    uint8_t sig[K_COUNT][2]; /* per kind: (required, optional) slot masks
                               from uncompute.ops.operand_mask_table() */
    /* strict ledger (sim.py's per-draw fixup ledger) */
    SimDraw *draws; int64_t d_len, d_cap;
    uint64_t *map;          /* open addressing id->index: pairs (id, idx);
                               id 0 = empty, NO_ID = tombstone (draw ids
                               start at 1) */
    int64_t map_cap;        /* in pairs, power of two */
    int64_t map_used;       /* live + tombstoned pairs (empties consumed);
                               rebuild before probe chains can saturate */
    SimSlot *slots; int64_t s_cap;   /* indexed by outcome id */
    uint64_t *scratch;      /* n_words (HMR pre-measure value / fixup mask) */
} SimState;

typedef struct {
    int64_t n_shots;
    int64_t batch;     /* max(n_shots, 1), like LaneSim */
    int64_t n_words;   /* ceil(batch / 64) */
    uint64_t last_mask;

    uint8_t seed[32];
    uint64_t rng_k;
    size_t rng_skip;    /* bytes of each counter's XOF stream to skip (lane0/8) */
    uint8_t *rng_bytes; /* n_words*8 bytes; tail beyond ceil(batch/8) stays 0 */
    uint64_t *rng;      /* n_words scratch */

    uint64_t *qubits;   /* q_alloc wires x n_words, grown on touch */
    int64_t q_alloc, q_cap;
    uint64_t *outs;     /* outcome ids (HMR writes, OXOR accumulates) */
    int64_t o_alloc, o_cap;
    uint64_t *bits;     /* classical bit ids (Bit-register inputs, BIT_* writes,
                           cc conditions); grown on touch, zero-initialized */
    int64_t b_alloc, b_cap;
    uint64_t *phase;    /* n_words */
    uint64_t *cond;     /* n_words scratch (cc/oc-narrowed action masks) */

    uint64_t *live;     /* n_words live-lane mask */
    uint64_t live_pop;  /* cached popcount(live): live is written only by
                           kern_create/kern_set_live, never mid-batch, so the
                           per-billed-op popcount is hoisted here (billing:
                           exec_op). Any future mid-batch write to `live`
                           MUST recount_live() or billing silently corrupts. */

    uint64_t toffoli, clifford;

    /* register events, quadruples of (rt, type, wire, reg_kind); type 0 =
       declare the register slot (reg_kind 0 = Qubit, 1 = Bit), 1 = append
       wire `wire` in the register's own address space. */
    uint64_t *events;
    int64_t ev_len, ev_cap; /* in quadruples */

    SimState *sim;          /* non-NULL only for kern_sim_* contexts */
} Kern;

static void sim_draw(Kern *k);

static void recount_live(Kern *k) {
    uint64_t n = 0;
    int64_t w;
    for (w = 0; w < k->n_words; w++)
        n += (uint64_t)__builtin_popcountll(k->live[w]);
    k->live_pop = n;
}

void *kern_create(int64_t n_shots, const uint8_t *seed_rng32) {
    Kern *k = (Kern *)calloc(1, sizeof(Kern));
    if (!k) return NULL;
    k->n_shots = n_shots;
    k->batch = n_shots < 1 ? 1 : n_shots;
    k->n_words = (k->batch + 63) / 64;
    {
        int rem = (int)(k->batch % 64);
        k->last_mask = rem ? ((1ULL << rem) - 1) : ~0ULL;
    }
    memcpy(k->seed, seed_rng32, 32);
    k->rng_bytes = (uint8_t *)calloc((size_t)k->n_words * 8, 1);
    k->rng = (uint64_t *)malloc((size_t)k->n_words * 8);
    k->phase = (uint64_t *)calloc((size_t)k->n_words, 8);
    k->cond = (uint64_t *)malloc((size_t)k->n_words * 8);
    k->live = (uint64_t *)malloc((size_t)k->n_words * 8);
    if (!k->rng_bytes || !k->rng || !k->phase || !k->cond || !k->live) {
        free(k->rng_bytes); free(k->rng); free(k->phase); free(k->cond);
        free(k->live); free(k);
        return NULL;
    }
    {
        int64_t w;
        for (w = 0; w < k->n_words; w++)
            k->live[w] = ~0ULL;
        k->live[k->n_words - 1] = k->last_mask;
    }
    recount_live(k);
    return k;
}

static void sim_free_state(SimState *s) {
    int64_t i;
    if (!s) return;
    for (i = 0; i < s->d_len; i++)
        free(s->draws[i].residual);
    free(s->draws);
    free(s->map);
    for (i = 0; i < s->s_cap; i++)
        free(s->slots[i].ids);
    free(s->slots);
    free(s->scratch);
    free(s);
}

void kern_free(void *vk) {
    Kern *k = (Kern *)vk;
    if (!k) return;
    free(k->rng_bytes); free(k->rng); free(k->qubits); free(k->outs);
    free(k->bits); free(k->phase); free(k->cond); free(k->live);
    free(k->events);
    sim_free_state(k->sim);
    free(k);
}

/* grow-on-touch, zero-initialized — mirrors Simulator._grow. Returns 0 ok. */
static int ensure_wire(Kern *k, uint64_t **arr, int64_t *alloc, int64_t *cap, uint64_t id) {
    if ((int64_t)id < *alloc)
        return 0;
    if ((int64_t)id >= *cap) {
        int64_t nc = *cap ? *cap : 64;
        uint64_t *na;
        while (nc <= (int64_t)id)
            nc *= 2;
        na = (uint64_t *)realloc(*arr, (size_t)nc * k->n_words * 8);
        if (!na)
            return ERR_OOM;
        *arr = na;
        *cap = nc;
    }
    memset(*arr + *alloc * k->n_words, 0,
           (size_t)((int64_t)id + 1 - *alloc) * k->n_words * 8);
    *alloc = (int64_t)id + 1;
    return 0;
}

#define ENSURE_Q(k, id) ensure_wire((k), &(k)->qubits, &(k)->q_alloc, &(k)->q_cap, (id))
#define ENSURE_O(k, id) ensure_wire((k), &(k)->outs, &(k)->o_alloc, &(k)->o_cap, (id))
#define ENSURE_B(k, id) ensure_wire((k), &(k)->bits, &(k)->b_alloc, &(k)->b_cap, (id))
#define QP(k, id) ((k)->qubits + (int64_t)(id) * (k)->n_words)
#define OP(k, id) ((k)->outs + (int64_t)(id) * (k)->n_words)
#define BP(k, id) ((k)->bits + (int64_t)(id) * (k)->n_words)

/* Global lane offset for sharded evaluation: batch b of size B evaluates
 * global lanes [b*B, b*B+batch), so every rng(k) draw must slice the XOF
 * stream at byte lane0/8. lane0 must be a multiple of 8 (the Python wrapper
 * enforces it). */
void kern_set_lane0(void *vk, int64_t lane0) {
    ((Kern *)vk)->rng_skip = (size_t)(lane0 / 8);
}

/* Dead-lane shot population: a skipped draw (gen_test_indexed -> None) keeps
 * its lane but must be inert. `live` is n_words lane words (bit j of word w =
 * lane 64w+j is live); it replaces the initial all-live mask, so a dead lane
 * never takes a state update, bills a counter, kicks the phase, or stores a
 * measurement. Must be called before any op executes. The final word is
 * masked to the batch width. */
void kern_set_live(void *vk, const uint64_t *live) {
    Kern *k = (Kern *)vk;
    int64_t w;
    for (w = 0; w < k->n_words; w++)
        k->live[w] = live[w];
    k->live[k->n_words - 1] &= k->last_mask;
    recount_live(k);
}

/* Block-counter measurement RNG. Byte j of counter c's stream lives in 128-byte block j/128:
 *
 *     block(c, i) = SHAKE256(seed ‖ u64le(c) ‖ u64le(i)), 128 bytes
 *
 * so a batch at global lane offset L reads only the blocks covering bytes
 * [L/8, L/8 + ceil(batch/8)), at O(1) cost in L. A single sequential
 * stream per counter would make sharded evaluation quadratic in the total
 * shot count. */
#define RNG_BLOCK 128

/* ---- ARMv8.4 SHA3 two-way Keccak (draw() fast path) --------------------
 * Two independent 128-byte RNG blocks per permutation: block i in NEON
 * lane 0, block i+1 in lane 1 (blocks are separate SHAKE256 instances by
 * the block-counter design, so pairing is free). EOR3, RAX1 and BCAX are
 * dedicated Keccak instructions; output bytes are identical to the
 * scalar path. The scalar keccakf above serves hosts without FEAT_SHA3
 * (the Makefile retries a plain compile when the flag is rejected). */
#if defined(__ARM_FEATURE_SHA3)
#include <arm_neon.h>

static void keccakf_x2(uint64_t A[25], uint64_t B[25]) {
    uint64x2_t st[25];
    int i, j, r;
    for (i = 0; i < 25; i++)
        st[i] = vcombine_u64(vcreate_u64(A[i]), vcreate_u64(B[i]));
    for (r = 0; r < 24; r++) {
        uint64x2_t bc[5], b[25], d;
        /* theta */
        for (i = 0; i < 5; i++)
            bc[i] = veor3q_u64(veor3q_u64(st[i], st[i + 5], st[i + 10]),
                               st[i + 15], st[i + 20]);
        for (i = 0; i < 5; i++) {
            d = vrax1q_u64(bc[(i + 4) % 5], bc[(i + 1) % 5]);
            for (j = 0; j < 25; j += 5)
                st[j + i] = veorq_u64(st[j + i], d);
        }
        /* rho, then pi, as in keccakf. XAR takes only an immediate
         * rotate, so the rotate is two USHL shifts by table offsets. */
        for (i = 0; i < 25; i++) {
            int src = (int)keccak_pi_src[i];
            int64_t n = (int64_t)keccak_rho[src];
            b[i] = vorrq_u64(vshlq_u64(st[src], vdupq_n_s64(n)),
                             vshlq_u64(st[src], vdupq_n_s64(n - 64)));
        }
        /* chi: A[x][y] = B[x][y] ^ (B[x+2][y] & ~B[x+1][y]) (BCAX) */
        for (j = 0; j < 25; j += 5)
            for (i = 0; i < 5; i++)
                st[j + i] = vbcaxq_u64(b[j + i], b[j + (i + 2) % 5],
                                       b[j + (i + 1) % 5]);
        /* iota */
        st[0] = veorq_u64(st[0], vdupq_n_u64(keccak_rc[r]));
    }
    for (i = 0; i < 25; i++) {
        A[i] = vgetq_lane_u64(st[i], 0);
        B[i] = vgetq_lane_u64(st[i], 1);
    }
}
#endif /* __ARM_FEATURE_SHA3 */

/* ---- x86 AVX-512 eight-way Keccak (draw() fast path) --------------------
 * Eight independent 128-byte RNG blocks per permutation: block i+j in
 * vector lane j of each __m512i state word (blocks are separate SHAKE256
 * instances by the block-counter design, so grouping is free). vprolvq
 * does the rho rotates, vpternlogq (imm 0xD2 = b ^ (~c & d) per the chi
 * step) fuses chi, and iota broadcasts the round constant. Output bytes
 * are identical to the scalar path. The scalar keccakf above serves hosts
 * without AVX-512F. */
#if !defined(__ARM_FEATURE_SHA3) && defined(__AVX512F__)
#include <immintrin.h>

static void keccakf_x8(__m512i st[25]) {
    int round, i, j;
    for (round = 0; round < 24; round++) {
        __m512i bc[5], b[25], d;
        /* theta */
        for (i = 0; i < 5; i++)
            bc[i] = _mm512_xor_si512(
                _mm512_xor_si512(_mm512_xor_si512(st[i], st[i + 5]),
                                 _mm512_xor_si512(st[i + 10], st[i + 15])),
                st[i + 20]);
        for (i = 0; i < 5; i++) {
            d = _mm512_xor_si512(bc[(i + 4) % 5],
                                 _mm512_rol_epi64(bc[(i + 1) % 5], 1));
            for (j = 0; j < 25; j += 5)
                st[j + i] = _mm512_xor_si512(st[j + i], d);
        }
        /* rho, then pi, as in keccakf */
        for (i = 0; i < 25; i++) {
            int src = (int)keccak_pi_src[i];
            b[i] = _mm512_rolv_epi64(st[src],
                                     _mm512_set1_epi64((long long)keccak_rho[src]));
        }
        /* chi: A[x][y] = B[x][y] ^ (~B[x+1][y] & B[x+2][y]) — vpternlogq 0xD2 */
        for (j = 0; j < 25; j += 5)
            for (i = 0; i < 5; i++)
                st[j + i] = _mm512_ternarylogic_epi64(
                    b[j + i], b[j + (i + 1) % 5], b[j + (i + 2) % 5], 0xD2);
        /* iota */
        st[0] = _mm512_xor_si512(st[0], _mm512_set1_epi64((long long)keccak_rc[round]));
    }
}
#endif /* AVX512F */

static void draw(Kern *k) {
    if (k->sim) {           /* Simulator contexts use sim.py's draw scheme */
        sim_draw(k);
        return;
    }
#if defined(__ARM_FEATURE_SHA3)
    /* Absorb-state template: 48-byte input (seed ‖ u64le(c) ‖ u64le(bidx))
     * XORed into the zero state + SHAKE padding (0x1f at byte 48, 0x80 at
     * byte 135). Little-endian host (assumed file-wide): the first 17
     * state words are the 136-byte rate region, so squeezing 128 bytes is
     * a memcpy. Only word 5 (the block index) varies per block. */
    uint64_t base[25];
    uint64_t c = k->rng_k++;
    size_t nb = (size_t)((k->batch + 7) / 8);
    size_t out = 0, skip = k->rng_skip % RNG_BLOCK;
    uint64_t bidx = (uint64_t)(k->rng_skip / RNG_BLOCK);
    int64_t w;
    int b;
    memset(base, 0, sizeof base);
    memcpy(base, k->seed, 32);
    base[4] = c;
    base[6] = 0x1fULL;
    base[16] = 0x8000000000000000ULL;
    while (out < nb) {
        uint64_t A[25], B[25];
        size_t n;
        memcpy(A, base, sizeof base);
        A[5] = bidx;
        memcpy(B, base, sizeof base);
        B[5] = bidx + 1;          /* unused on an odd tail — discarded */
        keccakf_x2(A, B);
        bidx++;
        n = RNG_BLOCK - skip;
        if (n > nb - out) n = nb - out;
        memcpy(k->rng_bytes + out, (const uint8_t *)A + skip, n);
        out += n;
        skip = 0;
        if (out < nb) {
            bidx++;
            n = RNG_BLOCK;
            if (n > nb - out) n = nb - out;
            memcpy(k->rng_bytes + out, (const uint8_t *)B, n);
            out += n;
        }
    }
    for (w = 0; w < k->n_words; w++) {
        uint64_t v = 0;
        for (b = 0; b < 8; b++)
            v |= (uint64_t)k->rng_bytes[w * 8 + b] << (8 * b);
        k->rng[w] = v;
    }
    k->rng[k->n_words - 1] &= k->last_mask;
#elif defined(__AVX512F__)
    /* Absorb-state template as in the ARM path: only word 5 (block index)
     * varies across the eight blocks of one permutation. After keccakf_x8,
     * vector lane j of state word w holds u64 w of block bidx+j; the 128
     * output bytes of block b are its first 16 state words, so lane word
     * k->rng[16*b + w] = lane j=b word w — a direct 8x16 u64 transpose
     * into the rng array, no byte repack. */
    uint64_t base[25];
    uint64_t c = k->rng_k++;
    size_t nb = (size_t)((k->batch + 7) / 8);
    uint64_t bidx = (uint64_t)(k->rng_skip / RNG_BLOCK);
    size_t skip = k->rng_skip % RNG_BLOCK;
    int64_t w;
    int b, i;
    memset(base, 0, sizeof base);
    memcpy(base, k->seed, 32);
    base[4] = c;
    base[6] = 0x1fULL;
    base[16] = 0x8000000000000000ULL;
    if (skip == 0) {
        /* Aligned fast path (every production batch size: rng_skip is a
         * multiple of RNG_BLOCK). Little-endian host: rng word w equals
         * bytes [8w, 8w+8) of the counter stream, so block b (128 bytes)
         * covers rng words [16b, 16b+16), and block b of a group is vector
         * lane b. Two 8x8 u64 transposes turn the 16 state words into 8
         * contiguous 16-word columns, written per lane; no byte buffer,
         * no repack. Tail groups spill through a 128-word bounce. */
        int64_t words_needed = (int64_t)((nb + 7) / 8);
        int64_t woff = 0;
        while (woff < words_needed) {
            __m512i st[25];
            for (i = 0; i < 25; i++)
                st[i] = _mm512_set1_epi64((long long)base[i]);
            st[5] = _mm512_set_epi64((long long)(bidx + 7), (long long)(bidx + 6),
                                     (long long)(bidx + 5), (long long)(bidx + 4),
                                     (long long)(bidx + 3), (long long)(bidx + 2),
                                     (long long)(bidx + 1), (long long)(bidx + 0));
            keccakf_x8(st);
            bidx += 8;
            {
                uint64_t cols[128] __attribute__((aligned(64)));
                /* transpose halves: words 0..7 then 8..15; column j lands at
                 * cols[16j + half*8 .. +8) */
                int half;
                for (half = 0; half < 2; half++) {
                    __m512i r0 = st[half * 8 + 0], r1 = st[half * 8 + 1];
                    __m512i r2 = st[half * 8 + 2], r3 = st[half * 8 + 3];
                    __m512i r4 = st[half * 8 + 4], r5 = st[half * 8 + 5];
                    __m512i r6 = st[half * 8 + 6], r7 = st[half * 8 + 7];
                    __m512i t0 = _mm512_unpacklo_epi64(r0, r1);
                    __m512i t1 = _mm512_unpackhi_epi64(r0, r1);
                    __m512i t2 = _mm512_unpacklo_epi64(r2, r3);
                    __m512i t3 = _mm512_unpackhi_epi64(r2, r3);
                    __m512i t4 = _mm512_unpacklo_epi64(r4, r5);
                    __m512i t5 = _mm512_unpackhi_epi64(r4, r5);
                    __m512i t6 = _mm512_unpacklo_epi64(r6, r7);
                    __m512i t7 = _mm512_unpackhi_epi64(r6, r7);
                    __m512i u0 = _mm512_shuffle_i64x2(t0, t2, 0x88);
                    __m512i u1 = _mm512_shuffle_i64x2(t1, t3, 0x88);
                    __m512i u2 = _mm512_shuffle_i64x2(t0, t2, 0xdd);
                    __m512i u3 = _mm512_shuffle_i64x2(t1, t3, 0xdd);
                    __m512i u4 = _mm512_shuffle_i64x2(t4, t6, 0x88);
                    __m512i u5 = _mm512_shuffle_i64x2(t5, t7, 0x88);
                    __m512i u6 = _mm512_shuffle_i64x2(t4, t6, 0xdd);
                    __m512i u7 = _mm512_shuffle_i64x2(t5, t7, 0xdd);
                    __m512i c0 = _mm512_shuffle_i64x2(u0, u4, 0x88);
                    __m512i c1 = _mm512_shuffle_i64x2(u1, u5, 0x88);
                    __m512i c2 = _mm512_shuffle_i64x2(u0, u4, 0xdd);
                    __m512i c3 = _mm512_shuffle_i64x2(u1, u5, 0xdd);
                    __m512i c4 = _mm512_shuffle_i64x2(u2, u6, 0x88);
                    __m512i c5 = _mm512_shuffle_i64x2(u3, u7, 0x88);
                    __m512i c6 = _mm512_shuffle_i64x2(u2, u6, 0xdd);
                    __m512i c7 = _mm512_shuffle_i64x2(u3, u7, 0xdd);
                    /* columns in lane order 0,1,4,5,2,3,6,7 after the two
                     * shuffle stages (unpack pairs lanes {2j,2j+1}; the 0x88/
                     * 0xdd stages then group chunk-pairs {0,1|4,5|2,3|6,7}),
                     * so cN holds block [0,1,4,5,2,3,6,7][N]; store each to
                     * its block's slot. */
                    _mm512_store_si512((__m512i *)(cols + 16 * 0 + half * 8), c0);
                    _mm512_store_si512((__m512i *)(cols + 16 * 1 + half * 8), c1);
                    _mm512_store_si512((__m512i *)(cols + 16 * 4 + half * 8), c2);
                    _mm512_store_si512((__m512i *)(cols + 16 * 5 + half * 8), c3);
                    _mm512_store_si512((__m512i *)(cols + 16 * 2 + half * 8), c4);
                    _mm512_store_si512((__m512i *)(cols + 16 * 3 + half * 8), c5);
                    _mm512_store_si512((__m512i *)(cols + 16 * 6 + half * 8), c6);
                    _mm512_store_si512((__m512i *)(cols + 16 * 7 + half * 8), c7);
                }
                {
                    int64_t nw2 = words_needed - woff;
                    if (nw2 > 128) nw2 = 128;
                    memcpy(k->rng + woff, cols, (size_t)nw2 * 8);
                    woff += nw2;
                }
            }
        }
        k->rng[k->n_words - 1] &= k->last_mask;
        return;
    }
    {
        size_t out = 0;               /* bytes written into rng_bytes */
        while (out < nb) {
            __m512i st[25];
            uint64_t blk[25][8] __attribute__((aligned(64)));
            size_t n;
            for (i = 0; i < 25; i++)
                st[i] = _mm512_set1_epi64((long long)base[i]);
            st[5] = _mm512_set_epi64((long long)(bidx + 7), (long long)(bidx + 6),
                                     (long long)(bidx + 5), (long long)(bidx + 4),
                                     (long long)(bidx + 3), (long long)(bidx + 2),
                                     (long long)(bidx + 1), (long long)(bidx + 0));
            keccakf_x8(st);
            for (i = 0; i < 25; i++)
                _mm512_store_si512((__m512i *)blk[i], st[i]);
            for (b = 0; b < 8 && out < nb; b++) {
                uint8_t tmp[RNG_BLOCK];
                for (i = 0; i < 16; i++)
                    ((uint64_t *)tmp)[i] = blk[i][b];
                n = RNG_BLOCK - skip;
                if (n > nb - out) n = nb - out;
                memcpy(k->rng_bytes + out, tmp + skip, n);
                out += n;
                skip = 0;
                bidx++;
            }
        }
    }
    for (w = 0; w < k->n_words; w++) {
        uint64_t v = 0;
        for (b = 0; b < 8; b++)
            v |= (uint64_t)k->rng_bytes[w * 8 + b] << (8 * b);
        k->rng[w] = v;
    }
    k->rng[k->n_words - 1] &= k->last_mask;
#else
    uint8_t inbuf[48];
    uint8_t blk[RNG_BLOCK];
    uint64_t c = k->rng_k++;
    size_t nb = (size_t)((k->batch + 7) / 8);
    size_t out = 0, skip = k->rng_skip % RNG_BLOCK;
    uint64_t bidx = (uint64_t)(k->rng_skip / RNG_BLOCK);
    int64_t w;
    int b;
    memcpy(inbuf, k->seed, 32);
    for (b = 0; b < 8; b++)
        inbuf[32 + b] = (uint8_t)(c >> (8 * b));
    while (out < nb) {
        size_t n;
        for (b = 0; b < 8; b++)
            inbuf[40 + b] = (uint8_t)(bidx >> (8 * b));
        bidx++;
        shake256(blk, RNG_BLOCK, inbuf, 48);
        n = RNG_BLOCK - skip;
        if (n > nb - out)
            n = nb - out;
        memcpy(k->rng_bytes + out, blk + skip, n);
        out += n;
        skip = 0;
    }
    for (w = 0; w < k->n_words; w++) {
        uint64_t v = 0;
        for (b = 0; b < 8; b++)
            v |= (uint64_t)k->rng_bytes[w * 8 + b] << (8 * b);
        k->rng[w] = v;
    }
    k->rng[k->n_words - 1] &= k->last_mask;
#endif
}

/* sim.py Simulator._draw: one fresh random bit per lane from
 *     SHAKE256(seed ‖ u64le(++ctr))
 * squeezed to ceil(batch/8) bytes, interpreted little-endian and masked to
 * the batch (NOT the evaluator's block-counter scheme above — the two RNG
 * streams are deliberately distinct and both normative for their engine).
 * rng_bytes' tail beyond ceil(batch/8) stays zero: every draw writes the
 * same nb bytes over a calloc'd buffer. */
static void sim_draw(Kern *k) {
    SimState *s = k->sim;
    uint8_t inbuf[40];
    size_t nb = (size_t)((k->batch + 7) / 8);
    int64_t w;
    int b;
    s->ctr++;
    memcpy(inbuf, s->seed, (size_t)s->seed_len);
    for (b = 0; b < 8; b++)
        inbuf[s->seed_len + b] = (uint8_t)(s->ctr >> (8 * b));
    shake256(k->rng_bytes, nb, inbuf, (size_t)s->seed_len + 8);
    for (w = 0; w < k->n_words; w++) {
        uint64_t v = 0;
        for (b = 0; b < 8; b++)
            v |= (uint64_t)k->rng_bytes[w * 8 + b] << (8 * b);
        k->rng[w] = v;
    }
    k->rng[k->n_words - 1] &= k->last_mask;
}

/* ---------------- register events ---------------- */

static int push_event(Kern *k, uint64_t rt, uint64_t type, uint64_t wire,
                      uint64_t reg_kind) {
    if (k->ev_len == k->ev_cap) {
        int64_t nc = k->ev_cap ? k->ev_cap * 2 : 256;
        uint64_t *ne = (uint64_t *)realloc(k->events, (size_t)nc * 4 * 8);
        if (!ne)
            return ERR_OOM;
        k->events = ne;
        k->ev_cap = nc;
    }
    k->events[k->ev_len * 4] = rt;
    k->events[k->ev_len * 4 + 1] = type;
    k->events[k->ev_len * 4 + 2] = wire;
    k->events[k->ev_len * 4 + 3] = reg_kind;
    k->ev_len++;
    return 0;
}

/* ---------------- gate execution ----------------
 *
 * The per-lane gate updates, 64 lanes per word. Simulator._apply_py in uncompute/sim.py is the normative
 * reference; this code gives identical state, counters, and RNG use. */

/* Grow every store that a gate op names (zero-filled), before any pointer
 * into the stores is taken. */
static int touch_operands(Kern *k, uint64_t qa, uint64_t qb, uint64_t qt,
                          uint64_t ot, uint64_t oc, uint64_t ct, uint64_t cc) {
    int rc;
    if (qa != NO_ID && (rc = ENSURE_Q(k, qa)) != 0) return rc;
    if (qb != NO_ID && (rc = ENSURE_Q(k, qb)) != 0) return rc;
    if (qt != NO_ID && (rc = ENSURE_Q(k, qt)) != 0) return rc;
    if (ot != NO_ID && (rc = ENSURE_O(k, ot)) != 0) return rc;
    if (oc != NO_ID && (rc = ENSURE_O(k, oc)) != 0) return rc;
    if (ct != NO_ID && (rc = ENSURE_B(k, ct)) != 0) return rc;
    if (cc != NO_ID && (rc = ENSURE_B(k, cc)) != 0) return rc;
    return 0;
}

/* The action mask: the live lanes, narrowed by a bit condition (cc) and,
 * on the sign kinds, by an outcome condition (oc). OXOR's oc is its source
 * operand, not a condition. Billing never uses this mask. */
static const uint64_t *action_mask(Kern *k, uint32_t kind, uint64_t oc,
                                   uint64_t cc) {
    const int by_bit = cc != NO_ID;
    const int by_outcome = oc != NO_ID && kind != K_OXOR;
    uint64_t *m = k->cond;
    int64_t w;
    if (!by_bit && !by_outcome)
        return k->live;
    for (w = 0; w < k->n_words; w++) {
        uint64_t v = k->live[w];
        if (by_bit)
            v &= BP(k, cc)[w];
        if (by_outcome)
            v &= OP(k, oc)[w];
        m[w] = v;
    }
    return m;
}

/* target ^= mask & c1 & c2, for zero, one, or two control rows */
static void toggle_lanes(uint64_t *t, const uint64_t *m, const uint64_t *c1,
                         const uint64_t *c2, int64_t nw) {
    int64_t w;
    if (c2)
        for (w = 0; w < nw; w++) t[w] ^= m[w] & c1[w] & c2[w];
    else if (c1)
        for (w = 0; w < nw; w++) t[w] ^= m[w] & c1[w];
    else
        for (w = 0; w < nw; w++) t[w] ^= m[w];
}

/* sign ^= mask & f1 & f2 & f3, for zero to three factor rows */
static void flip_sign_lanes(uint64_t *sign, const uint64_t *m,
                            const uint64_t *f1, const uint64_t *f2,
                            const uint64_t *f3, int64_t nw) {
    int64_t w;
    if (f3)
        for (w = 0; w < nw; w++) sign[w] ^= m[w] & f1[w] & f2[w] & f3[w];
    else if (f2)
        for (w = 0; w < nw; w++) sign[w] ^= m[w] & f1[w] & f2[w];
    else if (f1)
        for (w = 0; w < nw; w++) sign[w] ^= m[w] & f1[w];
    else
        for (w = 0; w < nw; w++) sign[w] ^= m[w];
}

static int exec_op(Kern *k, uint32_t kind, uint64_t qa, uint64_t qb,
                   uint64_t qt, uint64_t ot, uint64_t oc, uint64_t ct,
                   uint64_t cc) {
    const int64_t nw = k->n_words;
    const uint64_t *act;
    uint8_t cost;
    int64_t w;
    int rc;

    if ((rc = touch_operands(k, qa, qb, qt, ot, oc, ct, cc)) != 0)
        return rc;
    act = action_mask(k, kind, oc, cc);

    /* Billing: every live lane, at the op's cost class. */
    cost = (cc != NO_ID) ? COST_WITH_CC[kind] : COST_PLAIN[kind];
    if (cost == COST_TOFFOLI)
        k->toffoli += k->live_pop;
    else if (cost == COST_CLIFFORD)
        k->clifford += k->live_pop;

    switch (kind) {
    case K_X:
        toggle_lanes(QP(k, qt), act, NULL, NULL, nw);
        break;
    case K_CX:
        toggle_lanes(QP(k, qt), act, QP(k, qb), NULL, nw);
        break;
    case K_CCX:
        toggle_lanes(QP(k, qt), act, QP(k, qa), QP(k, qb), nw);
        break;
    case K_NEG:
        flip_sign_lanes(k->phase, act, NULL, NULL, NULL, nw);
        break;
    case K_Z:
        flip_sign_lanes(k->phase, act, QP(k, qt), NULL, NULL, nw);
        break;
    case K_CZ:
        flip_sign_lanes(k->phase, act, QP(k, qb), QP(k, qt), NULL, nw);
        break;
    case K_CCZ:
        flip_sign_lanes(k->phase, act, QP(k, qa), QP(k, qb), QP(k, qt), nw);
        break;
    case K_SWAP: {
        uint64_t *p = QP(k, qb), *t = QP(k, qt);
        for (w = 0; w < nw; w++) {
            uint64_t diff = (p[w] ^ t[w]) & act[w];
            p[w] ^= diff;
            t[w] ^= diff;
        }
        break;
    }
    case K_HMR: {
        uint64_t *o = OP(k, ot), *t = QP(k, qt);
        draw(k);
        for (w = 0; w < nw; w++) {
            uint64_t r = k->rng[w] & act[w];
            o[w] = (o[w] & ~act[w]) | r;
            k->phase[w] ^= t[w] & r;
            t[w] &= ~act[w];
        }
        break;
    }
    case K_R: {
        uint64_t *t = QP(k, qt);
        uint64_t any = 0;
        /* A reset of a target that is 0 on every acted lane changes no
         * state: its draw would act only through sign ^= t & r. Such a
         * reset steps the measurement counter and skips the draw, which
         * keeps the counter position exact. */
        for (w = 0; w < nw; w++)
            any |= t[w] & act[w];
        if (!any) {
            if (k->sim)
                k->sim->ctr++;   /* the counter the Simulator scheme uses */
            else
                k->rng_k++;
            break;
        }
        draw(k);
        for (w = 0; w < nw; w++) {
            k->phase[w] ^= t[w] & k->rng[w] & act[w];
            t[w] &= ~act[w];
        }
        break;
    }
    case K_OXOR: {
        uint64_t *o = OP(k, ot);
        const uint64_t *src = OP(k, oc);
        for (w = 0; w < nw; w++)
            o[w] ^= src[w] & act[w];
        break;
    }
    case K_BIT_INVERT: {
        uint64_t *c = BP(k, ct);
        for (w = 0; w < nw; w++) c[w] ^= act[w];
        break;
    }
    case K_BIT_STORE0: {
        uint64_t *c = BP(k, ct);
        for (w = 0; w < nw; w++) c[w] &= ~act[w];
        break;
    }
    case K_BIT_STORE1: {
        uint64_t *c = BP(k, ct);
        for (w = 0; w < nw; w++) c[w] |= act[w];
        break;
    }
    default:
        /* unreachable: the parser and sim_validate reject other kinds */
        break;
    }
    return 0;
}

/* ---------------- bulk register I/O ----------------
 *
 * Normative per-shot semantics: util.py _set_register/_get_register — wire i
 * of a register carries bit i of the shot's value; wires are processed in
 * list order (last write wins on aliased wires). Values travel as packed
 * little-endian bit rows: shot s occupies bytes [s*row_bytes, (s+1)*row_bytes)
 * with wire i at byte i/8, bit i%8. */

/* Load per-shot values into the register's lane columns. `valid` is a packed
 * bitmask (bit s = shot s carries a value); shots without a value keep their
 * current lane state, matching the reference's skip of None inputs. `is_bit`
 * selects the address space: 0 = Qubit register (lane state), 1 = Bit
 * register (bit space) — util._set_register's (reg_kind, wires) routing. */
int64_t kern_load_register(void *vk, const uint32_t *ids,
                           int64_t n_wires, const uint8_t *rows,
                           int64_t row_bytes, const uint8_t *valid,
                           int64_t n_shots, int64_t is_bit) {
    Kern *k = (Kern *)vk;
    int64_t i, s;
    for (i = 0; i < n_wires; i++) {
        uint64_t *lane;
        int rc = is_bit ? ENSURE_B(k, (uint64_t)ids[i])
                        : ENSURE_Q(k, (uint64_t)ids[i]);
        if (rc != 0)
            return rc;
        lane = is_bit ? BP(k, ids[i]) : QP(k, ids[i]);
        for (s = 0; s < n_shots; ) {
            uint64_t vm = 0, sm = 0;
            int64_t w = s >> 6;
            int64_t end = (w + 1) << 6;
            if (end > n_shots) end = n_shots;
            for (; s < end; s++) {
                if ((valid[s >> 3] >> (s & 7)) & 1) {
                    vm |= 1ULL << (s & 63);
                    if ((rows[s * row_bytes + (i >> 3)] >> (i & 7)) & 1)
                        sm |= 1ULL << (s & 63);
                }
            }
            lane[w] = (lane[w] & ~vm) | sm;
        }
    }
    return 0;
}

/* Extract per-shot values from the register's lane columns into packed rows
 * (the caller supplies a zeroed buffer). Wires beyond the allocated range
 * read as 0, like the reference. `is_bit` mirrors kern_load_register for
 * symmetry with util._get_register's (reg_kind, wires) routing; the verdict
 * path only ever reads Qubit registers (Bit registers are input-only), but
 * the semantics are defined for both spaces. */
void kern_read_register(void *vk, const uint32_t *ids,
                        int64_t n_wires, uint8_t *rows_out, int64_t row_bytes,
                        int64_t n_shots, int64_t is_bit) {
    Kern *k = (Kern *)vk;
    int64_t i, s;
    int64_t alloc = is_bit ? k->b_alloc : k->q_alloc;
    for (i = 0; i < n_wires; i++) {
        const uint64_t *lane;
        if ((int64_t)ids[i] >= alloc) continue;
        lane = is_bit ? BP(k, ids[i]) : QP(k, ids[i]);
        for (s = 0; s < n_shots; s++) {
            if ((lane[s >> 6] >> (s & 63)) & 1)
                rows_out[s * row_bytes + (i >> 3)] |= (uint8_t)(1u << (i & 7));
        }
    }
}

/* ---------------- accessors ---------------- */

int64_t kern_n_words(void *vk) { return ((Kern *)vk)->n_words; }
int64_t kern_alloc_qubits(void *vk) { return ((Kern *)vk)->q_alloc; }

void kern_counters(void *vk, uint64_t *out_toffoli, uint64_t *out_clifford) {
    Kern *k = (Kern *)vk;
    *out_toffoli = k->toffoli;
    *out_clifford = k->clifford;
}

/* bulk copies (n_words each) — cheaper than per-word ctypes calls */
void kern_read_qubit(void *vk, uint64_t q, uint64_t *out) {
    Kern *k = (Kern *)vk;
    if ((int64_t)q >= k->q_alloc)
        memset(out, 0, (size_t)k->n_words * 8);
    else
        memcpy(out, QP(k, q), (size_t)k->n_words * 8);
}

void kern_read_phase(void *vk, uint64_t *out) {
    Kern *k = (Kern *)vk;
    memcpy(out, k->phase, (size_t)k->n_words * 8);
}

int64_t kern_reg_event_count(void *vk) { return ((Kern *)vk)->ev_len; }
const uint64_t *kern_reg_events(void *vk) { return ((Kern *)vk)->events; }

/* ================= op-stream interpreter =================
 *
 * Normative reference: uncompute/evalcore/compact_reader.py, which follows
 * the format in the uncompute/ops.py docstring. This interpreter walks the segment
 * stream directly and never materializes the expansion. Its results match
 * the reference: census, counters, RNG use, failure codes, op indices, and
 * message text.
 *
 * Loops are interpreted with incremental affine operand updates: each op
 * record keeps its current resolved values (signed 128-bit, exact for any
 * state a finite run can reach; see kern_compact note below); each LOOP
 * level owns an adjust list that adds the level's stride once per iteration
 * and undoes the accumulation with one multiply at loop exit. The hot path
 * contains no multiplies.
 *
 * Error protocol: parse/validation failures store a code (CERR_*), the
 * expanded op index (-1 = none), and — except for aliasing, whose message
 * the Python wrapper regenerates from the resolved op via
 * uncompute.ops.op_violation — the exact reference message text, formatted here with the
 * same format strings as compact_reader.py.
 *
 * Exactness note: operand accumulators are __int128. base < 2^32, each
 * |stride| < 2^63, and reaching iteration i of any loop requires >= i
 * interpreter steps, so any state reachable in a finite run keeps
 * |value| far below 2^127 — the arithmetic is exact wherever it can be
 * observed. (A stream that could overflow would need ~2^64 executed
 * iterations first.)
 */

#include <stdarg.h>
#include <stdio.h>

/* compact segment / item bytes (uncompute/ops.py docstring) */
#define CSEG_REGS 0x01
#define CSEG_DATA 0x02
#define CSEG_FLAT 0x03
#define CSEG_LOOP 0x04
#define CSEG_END  0xFF

#define CMAX_LOOP_DEPTH 4
#define C_UNKNOWN_COUNT UINT64_MAX

/* compact error codes (mapped to report.py codes by evalcore/kernel.py) */
#define CERR_TRUNCATED      10  /* load/truncated */
#define CERR_SEGMENT        11  /* load/segment */
#define CERR_OP_INVALID     12  /* load/op-invalid */
#define CERR_COUNT_MISMATCH 13  /* load/count-mismatch */
#define CERR_TABLE_INDEX    14  /* validate/table-index */
#define CERR_ID_RANGE       15  /* validate/id-range */
#define CERR_ALIASING       16  /* validate/aliasing (msg regenerated in Python) */
#define CERR_ADDR_CAP       17  /* caps/address */
#define CERR_OOM            19

/* kern_compact_start / kern_compact_resume status values */
#define CST_PAUSED 0   /* prelude consumed; load inputs, then resume */
#define CST_DONE   1   /* END consumed, stream fully validated */

#define CERR_MSG_MAX 256
#define C_ID_LIMIT (1ULL << 32)

/* One parsed op record. cur[0] is the data-gate bit index when gated, then
 * one slot per operand (required order, then oc if flagged, then cc if
 * flagged — cc is always LAST) — mirrors compact_reader._OpRec. strides is
 * n_slots x depth (outermost first), NULL at depth 0.
 *
 * Sizing bound: the widest record is a gated CZ (or Z) carrying both
 * conditions — 2 required operands (1 for Z) + oc + cc = 4 operand fields,
 * plus the gate slot = 5 cur slots. No kind has 3 required operands AND a
 * legal condition (CCX/CCZ reject both flags), so 4/5 is exact. */
typedef struct {
    uint8_t kind;
    uint8_t gated;
    uint8_t n_fields;             /* operand count incl. flagged oc/cc */
    uint8_t field_slot[4];        /* flat 9-tuple slot per operand: 1=qa.. */
    __int128 cur[5];              /* [gate?] + fields */
    const int64_t *strides;       /* n_slots * depth */
    uint32_t table_id;
    const uint8_t *table_bits;
    uint64_t table_n_bits;
} CompOp;

struct CompLoop;
typedef struct {
    uint8_t is_loop;
    union { CompOp *op; struct CompLoop *loop; } u;
} CompItem;

typedef struct { __int128 *cur; int64_t stride; } CompAdj;

typedef struct CompLoop {
    uint64_t count;
    uint32_t n_items;
    CompItem *items;
    CompAdj *adjust;
    uint32_t n_adjust;
} CompLoop;

typedef struct { uint32_t id; const uint8_t *bits; uint64_t n_bits; } CompTable;

/* trivial arena: every parse allocation lives until kern_compact_free */
typedef struct CompArenaBlock {
    struct CompArenaBlock *next;
    size_t used, cap;
    /* payload follows */
} CompArenaBlock;

typedef struct {
    Kern *k;
    const uint8_t *buf;
    int64_t len, pos;

    CompArenaBlock *arena;
    CompTable *tables;
    int64_t n_tables, tables_cap;

    int saw_regs;
    int saw_program;
    int done;

    uint64_t n_expanded;
    uint64_t num_qubits, num_outs, num_bits;  /* census incl. REGS wire ids */

    int64_t err_code;               /* 0 = ok */
    int64_t err_op_index;           /* -1 = None */
    char err_msg[CERR_MSG_MAX];
    /* resolved (kind,qa,qb,qt,ot,oc,ct,cc) of the op that failed aliasing */
    uint64_t fail_op[8];
} CompCtx;

static void *comp_alloc(CompCtx *c, size_t n) {
    n = (n + 15) & ~(size_t)15;
    CompArenaBlock *b = c->arena;
    if (!b || b->cap - b->used < n) {
        size_t cap = n > 65536 ? n : 65536;
        b = (CompArenaBlock *)malloc(sizeof(CompArenaBlock) + cap);
        if (!b) return NULL;
        b->next = c->arena;
        b->used = 0;
        b->cap = cap;
        c->arena = b;
    }
    void *p = (char *)(b + 1) + b->used;
    b->used += n;
    return p;
}

/* format an __int128 in decimal (for reference-identical messages) */
static void i128_str(__int128 v, char *out) {
    char tmp[48];
    int i = 0, neg = v < 0;
    unsigned __int128 u = neg ? (unsigned __int128)(-(v + 1)) + 1 : (unsigned __int128)v;
    do { tmp[i++] = (char)('0' + (int)(u % 10)); u /= 10; } while (u);
    int j = 0;
    if (neg) out[j++] = '-';
    while (i) out[j++] = tmp[--i];
    out[j] = 0;
}

static int64_t comp_fail(CompCtx *c, int64_t code, int64_t op_index,
                         const char *fmt, ...) {
    va_list ap;
    c->err_code = code;
    c->err_op_index = op_index;
    va_start(ap, fmt);
    vsnprintf(c->err_msg, CERR_MSG_MAX, fmt, ap);
    va_end(ap);
    return code;
}

/* -- low-level reads (truncation messages match _CompactVM._read_exact) -- */

static int comp_need(CompCtx *c, uint64_t n, const char *what) {
    if ((uint64_t)(c->len - c->pos) < n) {
        comp_fail(c, CERR_TRUNCATED, -1,
                  "stream: truncated reading %s (wanted %llu bytes, got %lld)",
                  what, (unsigned long long)n, (long long)(c->len - c->pos));
        return 0;
    }
    return 1;
}

static int comp_u8(CompCtx *c, const char *what, uint8_t *out) {
    if (!comp_need(c, 1, what)) return 0;
    *out = c->buf[c->pos++];
    return 1;
}

static int comp_u16(CompCtx *c, const char *what, uint16_t *out) {
    if (!comp_need(c, 2, what)) return 0;
    uint16_t v;
    memcpy(&v, c->buf + c->pos, 2);
    c->pos += 2;
    *out = v;
    return 1;
}

static int comp_u32(CompCtx *c, const char *what, uint32_t *out) {
    if (!comp_need(c, 4, what)) return 0;
    uint32_t v;
    memcpy(&v, c->buf + c->pos, 4);
    c->pos += 4;
    *out = v;
    return 1;
}

static int comp_u64(CompCtx *c, const char *what, uint64_t *out) {
    if (!comp_need(c, 8, what)) return 0;
    uint64_t v;
    memcpy(&v, c->buf + c->pos, 8);
    c->pos += 8;
    *out = v;
    return 1;
}

static int comp_i64(CompCtx *c, const char *what, int64_t *out) {
    uint64_t v;
    if (!comp_u64(c, what, &v)) return 0;
    memcpy(out, &v, 8);
    return 1;
}

/* -- per-kind operand tables (the required slots of uncompute.ops.SIGNATURES)
 * Slots into the flat 9-tuple (kind, qa, qb, qt, ot, oc, rt, ct, cc);
 * slot 6 (rt) is never a stream operand. */
#define SLOT_QA 1
#define SLOT_QB 2
#define SLOT_QT  3
#define SLOT_OT  4
#define SLOT_OC  5
#define SLOT_CT  7
#define SLOT_CC  8

static const char *SLOT_NAME[9] = { NULL, "qa", "qb", "qt", "ot", "oc",
                                    NULL, "ct", "cc" };

typedef struct { int8_t n; uint8_t slots[3]; uint8_t oc_ok; uint8_t cc_ok; } CompKindRow;

/* n: the number of required operands, in stream order (slots).
 * oc_ok: the kind may carry the flag-appended outcome condition (Z/CZ/NEG
 * only — uncompute.ops.OC_ALLOWED_KINDS). cc_ok: the kind may carry the
 * flag-appended bit condition (exactly the promotable kinds —
 * uncompute.ops.CC_ALLOWED_KINDS). */
static const CompKindRow COMP_KINDS[K_COUNT] = {
    [K_X]          = { 1, {SLOT_QT}, 0, 1 },
    [K_Z]          = { 1, {SLOT_QT}, 1, 1 },
    [K_NEG]        = { 0, {0}, 1, 1 },
    [K_CX]         = { 2, {SLOT_QB, SLOT_QT}, 0, 1 },
    [K_CZ]         = { 2, {SLOT_QB, SLOT_QT}, 1, 1 },
    [K_SWAP]       = { 2, {SLOT_QB, SLOT_QT}, 0, 0 },
    [K_CCX]        = { 3, {SLOT_QA, SLOT_QB, SLOT_QT}, 0, 0 },
    [K_CCZ]        = { 3, {SLOT_QA, SLOT_QB, SLOT_QT}, 0, 0 },
    [K_R]          = { 1, {SLOT_QT}, 0, 0 },
    [K_HMR]        = { 2, {SLOT_QT, SLOT_OT}, 0, 0 },
    [K_OXOR]       = { 2, {SLOT_OT, SLOT_OC}, 0, 0 },
    [K_BIT_INVERT] = { 1, {SLOT_CT}, 0, 1 },
    [K_BIT_STORE0] = { 1, {SLOT_CT}, 0, 1 },
    [K_BIT_STORE1] = { 1, {SLOT_CT}, 0, 1 },
};

/* ---- producer-side tape encode (kern_tape_encode_flat) --------------------
 *
 * Serializes the packed rows of the Builder's columnar op tape
 * (builder._OpTape._rows: 9 u32 per item — slot 0 = kind byte | sentinel
 * mask << 8, mask bit (slot-1) set = that field holds the shared 2^64-1
 * sentinel; slots 1-8 the field values) into FLAT op records,
 * byte-identically to the reference encoder (compact._encode_tuple_op).
 * A STANDALONE PURE FUNCTION over (rows, the in-file kind tables) -> the
 * caller's chunk buffer: no context, no allocation, no I/O — Python
 * (uncompute/_tapefast.py) owns every fh.write.
 *
 * The encoder never "raises" and never emits bytes the reference would
 * reject: any item it cannot encode with certain byte-identity — a
 * side-table marker row (OpRec / odd tuple), a row carrying a bit
 * condition (the FLAG_CC spelling), a stray or missing operand whose
 * canonical error must surface at the exact byte position, an unknown
 * kind — stops the walk with TAPE_PYOP and *io_idx at the item; the
 * Python reference then produces the identical bytes or the canonical
 * error. Reference quirks are mirrored, not fixed: a stray
 * value in a slot that is neither a canonical operand nor a condition
 * (e.g. rt on a gate kind) is silently dropped, exactly as the
 * reference fast path drops it.
 *
 * out_cap is the FLUSH THRESHOLD, not the storage size: the caller's
 * buffer must hold at least out_cap + 16 bytes, because the encoder
 * appends whole op records (<= 14 bytes each) and returns TAPE_FLUSH at
 * the first record that reaches out_cap — which makes the caller's
 * write-out boundaries exactly the reference's _FLAT_CHUNK boundaries. */
#define TAPE_DONE   0   /* *io_idx reached stop; *out_len bytes valid */
#define TAPE_FLUSH  1   /* out_len crossed out_cap: flush, call again */
#define TAPE_PYOP   2   /* item at *io_idx needs the Python reference */
#define TAPE_BADARG 3   /* malformed arguments (caller bug, never data) */

#define TAPE_OBJ_ROW 0xFFFFFFFFu  /* slot-0 marker: item lives in _objs */

/* The encode-kind table, derived on first use from COMP_KINDS (whose rows
 * match compact.KIND_OPERANDS, each canonical slot list ascending):
 * per kind, the canonical-slot mask + whether slot 5 (oc) may ride as the
 * flag-appended oc. */
typedef struct { uint8_t smask; uint8_t oc_ok; } TapeEncRow;
static TapeEncRow TAPE_ENC[K_COUNT];
static int tape_enc_ready = 0;

static void tape_enc_init(void) {
    for (int k = 0; k < K_COUNT; k++) {
        const CompKindRow *r = &COMP_KINDS[k];
        uint8_t m = 0;
        for (int j = 0; j < r->n; j++)
            m |= (uint8_t)(1u << (r->slots[j] - 1));
        TAPE_ENC[k].smask = m;
        TAPE_ENC[k].oc_ok = r->oc_ok;
    }
    tape_enc_ready = 1;
}

int64_t kern_tape_encode_flat(const uint32_t *rows, int64_t n_items,
                              int64_t stop, int64_t *io_idx,
                              uint8_t *out, int64_t out_cap,
                              int64_t *out_len) {
    if (!io_idx || !out_len)
        return TAPE_BADARG;
    int64_t i = *io_idx;
    int64_t len = 0;
    *out_len = 0;
    if (!rows || !out || i < 0 || stop < i || stop > n_items || out_cap < 1)
        return TAPE_BADARG;
    if (!tape_enc_ready)
        tape_enc_init();
    while (i < stop) {
        const uint32_t *row = rows + i * 9;
        uint32_t s0 = row[0];
        if (s0 == TAPE_OBJ_ROW)
            break;                            /* side-table item: Python */
        uint32_t kind = s0 & 0xFFu;
        uint32_t mask = s0 >> 8;
        if (kind >= K_COUNT)
            break;                            /* unknown kind: canonical error */
        uint8_t smask = TAPE_ENC[kind].smask;
        if (!(mask & 0x80u))
            break;                 /* bit condition: the FLAG_CC spelling */
        if (!(mask & 0x40u) && !(smask & 0x40u))
            break;                 /* stray ct: canonical error */
        if (mask & smask)
            break;                 /* canonical operand missing: error */
        uint32_t flags = 0;
        int has_oc = 0;
        if (!(smask & 0x10u) && !(mask & 0x10u)) {
            /* slot 5 holds a value and is not a canonical operand */
            if (!TAPE_ENC[kind].oc_ok)
                break;             /* stray oc: canonical error */
            flags = 0x01u;         /* FLAG_OC */
            has_oc = 1;
        }
        const CompKindRow *kr = &COMP_KINDS[kind];
        out[len] = (uint8_t)kind;
        out[len + 1] = (uint8_t)flags;
        len += 2;
        for (int j = 0; j < kr->n; j++) {     /* canonical order (== slots) */
            memcpy(out + len, &row[kr->slots[j]], 4);
            len += 4;
        }
        if (has_oc) {
            memcpy(out + len, &row[SLOT_OC], 4);
            len += 4;
        }
        i++;
        if (len >= out_cap) {
            *io_idx = i;
            *out_len = len;
            return TAPE_FLUSH;
        }
    }
    *io_idx = i;
    *out_len = len;
    return i < stop ? TAPE_PYOP : TAPE_DONE;
}

/* REGS register kinds (uncompute/ops.py docstring; util.py
 * WKIND_QUBIT/WKIND_BIT) */
#define CREG_QUBIT 0
#define CREG_BIT   1

/* The caps failure (reference _check_caps): the message reports the census
 * as stored on the ctx — callers commit their maxima first. */
static int64_t comp_fail_caps(CompCtx *c, int64_t op_index) {
    return comp_fail(c, CERR_ADDR_CAP, op_index,
                     "address space too large: %llu qubits (cap %llu)"
                     " / %llu outcomes (cap %llu) / %llu bits (cap %llu)",
                     (unsigned long long)c->num_qubits,
                     (unsigned long long)MAX_QUBIT_ADDR,
                     (unsigned long long)c->num_outs,
                     (unsigned long long)MAX_OUT_ADDR,
                     (unsigned long long)c->num_bits,
                     (unsigned long long)MAX_BIT_ADDR);
}

/* -- REGS segment: registers
 * surface as Kern events (type 0 declares
 * the register slot with its kind byte, 1 appends a wire in the register's
 * own space); Qubit wires census into num_qubits, Bit wires into num_bits. */
static int64_t comp_seg_regs(CompCtx *c) {
    Kern *k = c->k;
    if (c->saw_regs)
        return comp_fail(c, CERR_SEGMENT, -1,
                         "stream: duplicate REGS segment (at most one)");
    if (c->saw_program)
        return comp_fail(c, CERR_SEGMENT, -1,
                         "stream: REGS segment must precede every FLAT/LOOP segment");
    c->saw_regs = 1;
    uint32_t n_regs;
    if (!comp_u32(c, "REGS n_regs", &n_regs)) return c->err_code;
    char what[64];
    for (uint32_t r = 0; r < n_regs; r++) {
        uint8_t kind;
        snprintf(what, sizeof what, "REGS reg %u reg_kind", r);
        if (!comp_u8(c, what, &kind)) return c->err_code;
        if (kind != CREG_QUBIT && kind != CREG_BIT)
            return comp_fail(c, CERR_SEGMENT, -1,
                             "stream: unknown REGS reg_kind %u for register %u",
                             kind, r);
        snprintf(what, sizeof what, "REGS reg %u n_wires", r);
        uint32_t n_wires;
        if (!comp_u32(c, what, &n_wires)) return c->err_code;
        if (push_event(k, r, 0, 0, kind) != 0)
            return comp_fail(c, CERR_OOM, -1, "out of memory");
        for (uint32_t w = 0; w < n_wires; w++) {
            uint32_t wid;
            snprintf(what, sizeof what, "REGS reg %u wire %u id", r, w);
            if (!comp_u32(c, what, &wid)) return c->err_code;
            if (kind == CREG_QUBIT) {
                if ((uint64_t)wid + 1 > c->num_qubits)
                    c->num_qubits = wid + 1ULL;
            } else if ((uint64_t)wid + 1 > c->num_bits) {
                c->num_bits = wid + 1ULL;
            }
            if (push_event(k, r, 1, wid, kind) != 0)
                return comp_fail(c, CERR_OOM, -1, "out of memory");
        }
    }
    if (c->num_qubits > MAX_QUBIT_ADDR || c->num_outs > MAX_OUT_ADDR
            || c->num_bits > MAX_BIT_ADDR)
        return comp_fail_caps(c, -1);
    return 0;
}

/* -- DATA segment: the table bytes
 * alias the stream buffer (the Python
 * wrapper keeps the buffer alive for the whole run). */
static int64_t comp_seg_data(CompCtx *c) {
    uint32_t table_id;
    if (!comp_u32(c, "DATA table_id", &table_id)) return c->err_code;
    for (int64_t t = 0; t < c->n_tables; t++)
        if (c->tables[t].id == table_id)
            return comp_fail(c, CERR_SEGMENT, -1,
                             "stream: duplicate DATA table_id %u", table_id);
    uint64_t n_bits;
    if (!comp_u64(c, "DATA n_bits", &n_bits)) return c->err_code;
    uint64_t n_bytes = n_bits / 8 + ((n_bits % 8) ? 1 : 0);
    char what[48];
    snprintf(what, sizeof what, "DATA table %u bits", table_id);
    if (!comp_need(c, n_bytes, what)) return c->err_code;
    if (c->n_tables == c->tables_cap) {
        int64_t nc = c->tables_cap ? c->tables_cap * 2 : 8;
        CompTable *nt = (CompTable *)comp_alloc(c, (size_t)nc * sizeof(CompTable));
        if (!nt) return comp_fail(c, CERR_OOM, -1, "out of memory");
        if (c->n_tables)
            memcpy(nt, c->tables, (size_t)c->n_tables * sizeof(CompTable));
        c->tables = nt;
        c->tables_cap = nc;
    }
    c->tables[c->n_tables].id = table_id;
    c->tables[c->n_tables].bits = c->buf + c->pos;
    c->tables[c->n_tables].n_bits = n_bits;
    c->n_tables++;
    c->pos += (int64_t)n_bytes;
    return 0;
}

/* -- op / loop parsing (structural — once, at parse). Check order is
 * the reference's: unknown kind -> reserved flags -> oc-flag
 * legality -> cc-flag legality -> gate block (incl. table lookup) ->
 * operands (required order, then oc, then cc LAST). */
static int64_t comp_parse_op(CompCtx *c, int depth, CompOp *op) {
    uint8_t kind, flags;
    if (!comp_u8(c, "op kind", &kind)) return c->err_code;
    if (!comp_u8(c, "op flags", &flags)) return c->err_code;
    if (kind >= K_COUNT)
        return comp_fail(c, CERR_OP_INVALID, -1, "op: unknown kind %u", kind);
    if (flags & ~0x07)
        return comp_fail(c, CERR_OP_INVALID, -1,
                         "op: reserved flag bits set (flags=0x%02x)", flags);
    int has_oc = flags & 0x01;
    int gated = (flags & 0x02) != 0;
    int has_cc = (flags & 0x04) != 0;
    const CompKindRow *row = &COMP_KINDS[kind];
    if (has_oc && !row->oc_ok)
        return comp_fail(c, CERR_OP_INVALID, -1,
                         "op: kind %u sets flag bit 0 (outcome condition), "
                         "which only Z, CZ, and NEG may carry", kind);
    if (has_cc && !row->cc_ok)
        return comp_fail(c, CERR_OP_INVALID, -1,
                         "op: kind %u sets flag bit 2 (bit condition), which "
                         "only X, Z, NEG, CX, CZ, and the bit-only kinds may "
                         "carry", kind);

    op->kind = kind;
    op->gated = (uint8_t)gated;
    op->n_fields = (uint8_t)(row->n + (has_oc ? 1 : 0) + (has_cc ? 1 : 0));
    int n_slots = (gated ? 1 : 0) + op->n_fields;
    int64_t *strides = NULL;
    if (depth > 0 && n_slots > 0) {
        strides = (int64_t *)comp_alloc(c, (size_t)n_slots * depth * 8);
        if (!strides) return comp_fail(c, CERR_OOM, -1, "out of memory");
    }
    op->strides = strides;
    int slot = 0;

    if (gated) {
        uint32_t table_id;
        uint64_t offset;
        if (!comp_u32(c, "gate table_id", &table_id)) return c->err_code;
        if (!comp_u64(c, "gate offset", &offset)) return c->err_code;
        for (int j = 0; j < depth; j++)
            if (!comp_i64(c, "gate stride", &strides[slot * depth + j]))
                return c->err_code;
        const CompTable *tbl = NULL;
        for (int64_t t = 0; t < c->n_tables; t++)
            if (c->tables[t].id == table_id) { tbl = &c->tables[t]; break; }
        if (!tbl)
            return comp_fail(c, CERR_OP_INVALID, -1,
                             "op: gate references table %u with no preceding "
                             "DATA segment", table_id);
        op->table_id = table_id;
        op->table_bits = tbl->bits;
        op->table_n_bits = tbl->n_bits;
        op->cur[slot] = (__int128)offset;
        slot++;
    }

    for (int f = 0; f < op->n_fields; f++) {
        /* appended operands come after the required ones: oc first (when
         * flagged), cc always last. */
        if (f < row->n)
            op->field_slot[f] = row->slots[f];
        else if (has_oc && f == row->n)
            op->field_slot[f] = SLOT_OC;
        else
            op->field_slot[f] = SLOT_CC;
        uint32_t base;
        if (!comp_u32(c, "operand base", &base)) return c->err_code;
        for (int j = 0; j < depth; j++)
            if (!comp_i64(c, "operand stride", &strides[slot * depth + j]))
                return c->err_code;
        op->cur[slot] = (__int128)base;
        slot++;
    }
    return 0;
}

static int64_t comp_parse_loop(CompCtx *c, int enclosing, CompLoop **out) {
    if (enclosing + 1 > CMAX_LOOP_DEPTH)
        return comp_fail(c, CERR_SEGMENT, -1,
                         "stream: LOOP nesting depth exceeds %d", CMAX_LOOP_DEPTH);
    CompLoop *node = (CompLoop *)comp_alloc(c, sizeof(CompLoop));
    if (!node) return comp_fail(c, CERR_OOM, -1, "out of memory");
    if (!comp_u64(c, "LOOP count", &node->count)) return c->err_code;
    uint16_t n_items;
    if (!comp_u16(c, "LOOP n_items", &n_items)) return c->err_code;
    if (n_items == 0)
        return comp_fail(c, CERR_SEGMENT, -1, "stream: LOOP with n_items == 0");
    node->n_items = n_items;
    node->adjust = NULL;
    node->n_adjust = 0;
    node->items = (CompItem *)comp_alloc(c, (size_t)n_items * sizeof(CompItem));
    if (!node->items) return comp_fail(c, CERR_OOM, -1, "out of memory");
    for (uint32_t i = 0; i < n_items; i++) {
        uint8_t item_kind;
        if (!comp_u8(c, "LOOP item kind", &item_kind)) return c->err_code;
        if (item_kind == 0x00) {
            CompOp *op = (CompOp *)comp_alloc(c, sizeof(CompOp));
            if (!op) return comp_fail(c, CERR_OOM, -1, "out of memory");
            if (comp_parse_op(c, enclosing + 1, op) != 0) return c->err_code;
            node->items[i].is_loop = 0;
            node->items[i].u.op = op;
        } else if (item_kind == 0x01) {
            CompLoop *sub;
            if (comp_parse_loop(c, enclosing + 1, &sub) != 0) return c->err_code;
            node->items[i].is_loop = 1;
            node->items[i].u.loop = sub;
        } else {
            return comp_fail(c, CERR_SEGMENT, -1,
                             "stream: unknown LOOP item kind 0x%02x", item_kind);
        }
    }
    *out = node;
    return 0;
}

/* -- adjust wiring (compact_reader._wire_adjust): each loop level gets the
 * (cur-slot, stride) pairs of every op in its subtree whose stride at that
 * level is nonzero. ``pos`` is the level's 0-based depth; an op at depth d
 * stores strides slot-major, so slot k's stride at level pos is
 * strides[k*d + pos]. Depth is uniform inside one top-level LOOP subtree
 * only for the ops it encloses — each op's own depth is passed down. */
static int comp_count_subtree_adjust(const CompLoop *node, int pos, int depth) {
    int n = 0;
    for (uint32_t i = 0; i < node->n_items; i++) {
        const CompItem *it = &node->items[i];
        if (it->is_loop) {
            n += comp_count_subtree_adjust(it->u.loop, pos, depth + 1);
        } else {
            const CompOp *op = it->u.op;
            int n_slots = (op->gated ? 1 : 0) + op->n_fields;
            for (int s = 0; s < n_slots; s++)
                if (op->strides[s * depth + pos] != 0)
                    n++;
        }
    }
    return n;
}

static void comp_fill_subtree_adjust(CompLoop *node, int pos, int depth,
                                     CompAdj **w) {
    for (uint32_t i = 0; i < node->n_items; i++) {
        CompItem *it = &node->items[i];
        if (it->is_loop) {
            comp_fill_subtree_adjust(it->u.loop, pos, depth + 1, w);
        } else {
            CompOp *op = it->u.op;
            int n_slots = (op->gated ? 1 : 0) + op->n_fields;
            for (int s = 0; s < n_slots; s++) {
                int64_t st = op->strides[s * depth + pos];
                if (st != 0) {
                    (*w)->cur = &op->cur[s];
                    (*w)->stride = st;
                    (*w)++;
                }
            }
        }
    }
}

/* assign every loop of the tree its adjust list; ``pos`` = this node's level */
static int64_t comp_wire_adjust(CompCtx *c, CompLoop *node, int pos) {
    int n = comp_count_subtree_adjust(node, pos, pos + 1);
    node->n_adjust = (uint32_t)n;
    if (n) {
        node->adjust = (CompAdj *)comp_alloc(c, (size_t)n * sizeof(CompAdj));
        if (!node->adjust) return comp_fail(c, CERR_OOM, -1, "out of memory");
        CompAdj *w = node->adjust;
        comp_fill_subtree_adjust(node, pos, pos + 1, &w);
    }
    for (uint32_t i = 0; i < node->n_items; i++)
        if (node->items[i].is_loop)
            if (comp_wire_adjust(c, node->items[i].u.loop, pos + 1) != 0)
                return c->err_code;
    return 0;
}

/* -- one expanded op (compact_reader.py check order:
 * gate table-index -> gate bit ->
 * operand id-range -> aliasing -> census/caps). Gated-off ops leave no
 * trace: no checks, no census, no stats, no RNG tick. */
static int64_t comp_exec_op(CompCtx *c, CompOp *op) {
    int i0 = 0;
    char num[48];
    if (op->gated) {
        __int128 b = op->cur[0];
        if (b < 0 || (unsigned __int128)b >= op->table_n_bits) {
            i128_str(b, num);
            return comp_fail(c, CERR_TABLE_INDEX, (int64_t)c->n_expanded,
                             "data-gate bit index %s out of [0, %llu) (table %u)",
                             num, (unsigned long long)op->table_n_bits,
                             op->table_id);
        }
        if (!((op->table_bits[(uint64_t)b >> 3] >> ((uint64_t)b & 7)) & 1))
            return 0;  /* "Data gating": the op does not exist */
        i0 = 1;
    }

    uint64_t tup[9] = { op->kind, NO_ID, NO_ID, NO_ID, NO_ID, NO_ID, NO_ID,
                        NO_ID, NO_ID };
    uint64_t maxq = c->num_qubits, maxo = c->num_outs, maxb = c->num_bits;
    for (int f = 0; f < op->n_fields; f++) {
        __int128 v = op->cur[i0 + f];
        int slot = op->field_slot[f];
        if (v < 0 || v >= (__int128)C_ID_LIMIT) {
            i128_str(v, num);
            return comp_fail(c, CERR_ID_RANGE, (int64_t)c->n_expanded,
                             "operand %s resolves to %s, outside [0, 2^32)",
                             SLOT_NAME[slot], num);
        }
        uint64_t id = (uint64_t)v;
        tup[slot] = id;
        if (slot <= SLOT_QT) {          /* 1..3: qubit space */
            if (id + 1 > maxq) maxq = id + 1;
        } else if (slot >= SLOT_CT) {   /* 7..8: bit space */
            if (id + 1 > maxb) maxb = id + 1;
        } else {                        /* 4..5: outcome space */
            if (id + 1 > maxo) maxo = id + 1;
        }
    }

    uint64_t qa = tup[1], qb = tup[2], qt = tup[3], ot = tup[4], oc = tup[5],
             ct = tup[7], cc = tup[8];
    if ((qt == qb && qt != NO_ID) || (qt == qa && qt != NO_ID)
            || (qb == qa && qb != NO_ID) || (ot == oc && ot != NO_ID)
            || (ct == cc && ct != NO_ID)) {
        memcpy(c->fail_op, tup, 6 * sizeof(uint64_t));
        c->fail_op[6] = ct;             /* skip tup[6] (rt, never set) */
        c->fail_op[7] = cc;
        return comp_fail(c, CERR_ALIASING, (int64_t)c->n_expanded,
                         "aliasing");  /* text regenerated by the wrapper */
    }

    c->num_qubits = maxq;
    c->num_outs = maxo;
    c->num_bits = maxb;
    if (maxq > MAX_QUBIT_ADDR || maxo > MAX_OUT_ADDR || maxb > MAX_BIT_ADDR)
        return comp_fail_caps(c, (int64_t)c->n_expanded);

    c->n_expanded++;
    int rc = exec_op(c->k, op->kind, qa, qb, qt, ot, oc, ct, cc);
    if (rc != 0)
        return comp_fail(c, CERR_OOM, (int64_t)(c->n_expanded - 1),
                         "out of memory");
    return 0;
}

/* -- direct loop interpretation:
 * pure adds per iteration; one
 * multiply per loop exit undoes the accumulation for the parent. */
static int64_t comp_run_loop(CompCtx *c, CompLoop *node) {
    const uint64_t count = node->count;
    if (count == 0)
        return 0;  /* contributes nothing */
    CompItem *items = node->items;
    const uint32_t n_items = node->n_items;
    CompAdj *adj = node->adjust;
    const uint32_t n_adj = node->n_adjust;
    for (uint64_t it = 0; it < count; it++) {
        for (uint32_t i = 0; i < n_items; i++) {
            int64_t rc = items[i].is_loop
                ? comp_run_loop(c, items[i].u.loop)
                : comp_exec_op(c, items[i].u.op);
            if (rc != 0)
                return rc;
        }
        for (uint32_t a = 0; a < n_adj; a++)
            *adj[a].cur += adj[a].stride;
    }
    for (uint32_t a = 0; a < n_adj; a++)
        *adj[a].cur -= (__int128)adj[a].stride * (__int128)count;
    return 0;
}

static int64_t comp_seg_flat(CompCtx *c) {
    uint32_t n_ops;
    if (!comp_u32(c, "FLAT n_ops", &n_ops)) return c->err_code;
    CompOp op;
    for (uint32_t i = 0; i < n_ops; i++) {
        if (comp_parse_op(c, 0, &op) != 0) return c->err_code;
        if (comp_exec_op(c, &op) != 0) return c->err_code;
    }
    return 0;
}

static int64_t comp_seg_end(CompCtx *c) {
    uint64_t declared;
    if (!comp_u64(c, "END expanded_op_count", &declared)) return c->err_code;
    if (c->pos != c->len)
        return comp_fail(c, CERR_SEGMENT, -1,
                         "stream: trailing bytes after END segment");
    if (declared != C_UNKNOWN_COUNT && declared != c->n_expanded)
        return comp_fail(c, CERR_COUNT_MISMATCH, -1,
                         "stream: expanded op count mismatch: got %llu ops,"
                         " END declared %llu",
                         (unsigned long long)c->n_expanded,
                         (unsigned long long)declared);
    return 0;
}

/* Walk segments until END. ``prelude`` pauses (without consuming the head
 * byte) at the first FLAT/LOOP so the wrapper can load register inputs —
 * the reference loads them at REGS-parse time, and nothing between REGS and
 * the first executable segment touches sim state. */
static int64_t comp_walk(CompCtx *c, int prelude) {
    while (1) {
        if (c->pos >= c->len)
            return comp_fail(c, CERR_TRUNCATED, -1,
                             "stream: EOF before END segment");
        uint8_t seg = c->buf[c->pos];
        if (prelude && (seg == CSEG_FLAT || seg == CSEG_LOOP))
            return CST_PAUSED;
        c->pos++;
        switch (seg) {
        case CSEG_REGS:
            if (comp_seg_regs(c) != 0) return c->err_code;
            break;
        case CSEG_DATA:
            if (comp_seg_data(c) != 0) return c->err_code;
            break;
        case CSEG_FLAT:
            c->saw_program = 1;
            if (comp_seg_flat(c) != 0) return c->err_code;
            break;
        case CSEG_LOOP: {
            c->saw_program = 1;
            CompLoop *node;
            if (comp_parse_loop(c, 0, &node) != 0) return c->err_code;
            if (comp_wire_adjust(c, node, 0) != 0) return c->err_code;
            if (comp_run_loop(c, node) != 0) return c->err_code;
            break;
        }
        case CSEG_END:
            if (comp_seg_end(c) != 0) return c->err_code;
            c->done = 1;
            return CST_DONE;
        default:
            return comp_fail(c, CERR_SEGMENT, -1,
                             "stream: unknown segment type 0x%02x", seg);
        }
    }
}

/* ---------------- compact API ---------------- */

void *kern_compact_create(void *vk) {
    CompCtx *c = (CompCtx *)calloc(1, sizeof(CompCtx));
    if (!c) return NULL;
    c->k = (Kern *)vk;
    c->err_op_index = -1;
    return c;
}

void kern_compact_free(void *vc) {
    CompCtx *c = (CompCtx *)vc;
    if (!c) return;
    CompArenaBlock *b = c->arena;
    while (b) {
        CompArenaBlock *next = b->next;
        free(b);
        b = next;
    }
    free(c);
}

/* Parse the REGS/DATA prelude of ``buf`` (which the caller must keep alive
 * for the whole run — DATA tables alias it). Returns CST_PAUSED (load
 * inputs, then kern_compact_resume), CST_DONE, or a CERR_* code. */
int64_t kern_compact_start(void *vc, const uint8_t *buf, int64_t len) {
    CompCtx *c = (CompCtx *)vc;
    c->buf = buf;
    c->len = len;
    c->pos = 0;
    c->k->ev_len = 0;
    return comp_walk(c, 1);
}

int64_t kern_compact_resume(void *vc) {
    return comp_walk((CompCtx *)vc, 0);
}

int64_t kern_compact_error_code(void *vc) { return ((CompCtx *)vc)->err_code; }
int64_t kern_compact_error_op_index(void *vc) { return ((CompCtx *)vc)->err_op_index; }
const char *kern_compact_error_msg(void *vc) { return ((CompCtx *)vc)->err_msg; }

int64_t kern_compact_ops(void *vc) { return (int64_t)((CompCtx *)vc)->n_expanded; }
int64_t kern_compact_num_qubits(void *vc) { return (int64_t)((CompCtx *)vc)->num_qubits; }
int64_t kern_compact_num_outs(void *vc) { return (int64_t)((CompCtx *)vc)->num_outs; }
int64_t kern_compact_num_bits(void *vc) { return (int64_t)((CompCtx *)vc)->num_bits; }

/* resolved (kind, qa, qb, qt, ot, oc, ct, cc) of the op that failed aliasing */
void kern_compact_fail_op(void *vc, uint64_t *out8) {
    memcpy(out8, ((CompCtx *)vc)->fail_op, 8 * sizeof(uint64_t));
}

/* ================= uncompute.Simulator fast path (kern_sim_*) =================
 *
 * See the SimState comment near the top of the file for the contract:
 * per-apply context, bulk state load -> walk -> bulk dump; any op the walk
 * cannot run with certain bit-identity halts it (return = local op index)
 * and the caller replays the chunk on the pure-Python engine. All ledger
 * work happens HERE, around exec_op — exec_op itself is shared with the
 * evaluator paths and stays sim-agnostic apart from the draw() dispatch. */

/* ---- draw-record map: open addressing, id 0 empty / NO_ID tombstone ---- */

static int64_t sim_map_find(const SimState *s, uint64_t id) {
    uint64_t mask, h;
    if (!s->map_cap)
        return -1;
    mask = (uint64_t)s->map_cap - 1;
    h = (id * 0x9e3779b97f4a7c15ULL) & mask;
    for (;;) {
        uint64_t got = s->map[2 * h];
        if (got == id)
            return (int64_t)s->map[2 * h + 1];
        if (got == 0)
            return -1;
        h = (h + 1) & mask;
    }
}

/* Returns 1 when the placement consumed an EMPTY pair (vs reusing a
 * tombstone / overwriting the same id) — the caller tracks map_used so the
 * table is rebuilt before empties run out (a full-of-tombstones table would
 * make a missing-id probe spin forever). */
static int sim_map_place(uint64_t *map, int64_t cap, uint64_t id,
                         uint64_t idx) {
    uint64_t mask = (uint64_t)cap - 1;
    uint64_t h = (id * 0x9e3779b97f4a7c15ULL) & mask;
    while (map[2 * h] != 0 && map[2 * h] != NO_ID && map[2 * h] != id)
        h = (h + 1) & mask;
    {
        int was_empty = map[2 * h] == 0;
        map[2 * h] = id;
        map[2 * h + 1] = idx;
        return was_empty;
    }
}

static int sim_map_rebuild(SimState *s, int64_t need) {
    int64_t cap = 16, i;
    uint64_t *nm;
    while (cap < 2 * need)
        cap *= 2;
    nm = (uint64_t *)calloc((size_t)cap * 2, 8);
    if (!nm)
        return ERR_OOM;
    for (i = 0; i < s->d_len; i++)
        sim_map_place(nm, cap, s->draws[i].id, (uint64_t)i);
    free(s->map);
    s->map = nm;
    s->map_cap = cap;
    s->map_used = s->d_len;
    return 0;
}

static void sim_map_remove(SimState *s, uint64_t id) {
    uint64_t mask = (uint64_t)s->map_cap - 1;
    uint64_t h = (id * 0x9e3779b97f4a7c15ULL) & mask;
    for (;;) {
        uint64_t got = s->map[2 * h];
        if (got == id) {
            s->map[2 * h] = NO_ID;   /* tombstone */
            return;
        }
        if (got == 0)
            return;
        h = (h + 1) & mask;
    }
}

/* Insert a draw record (residual copied). Returns 0 ok / ERR_OOM. */
static int sim_draw_insert(Kern *k, uint64_t id, uint64_t opn,
                           uint64_t out_id, const uint64_t *residual,
                           uint64_t refs) {
    SimState *s = k->sim;
    SimDraw *d;
    uint64_t *res;
    int rc;
    if (s->d_len == s->d_cap) {
        int64_t nc = s->d_cap ? s->d_cap * 2 : 64;
        SimDraw *nd = (SimDraw *)realloc(s->draws,
                                         (size_t)nc * sizeof(SimDraw));
        if (!nd)
            return ERR_OOM;
        s->draws = nd;
        s->d_cap = nc;
    }
    /* rebuild at 3/4 occupancy counting tombstones (map_used), so probe
     * chains always terminate at an empty pair */
    if (s->map_cap == 0 || 4 * (s->map_used + 1) > 3 * s->map_cap)
        if ((rc = sim_map_rebuild(s, s->d_len + 1)) != 0)
            return rc;
    res = (uint64_t *)malloc((size_t)k->n_words * 8);
    if (!res)
        return ERR_OOM;
    memcpy(res, residual, (size_t)k->n_words * 8);
    d = &s->draws[s->d_len];
    d->id = id;
    d->refs = refs;
    d->opn = opn;
    d->out_id = out_id;
    d->residual = res;
    s->map_used += sim_map_place(s->map, s->map_cap, id, (uint64_t)s->d_len);
    s->d_len++;
    return 0;
}

/* Swap-remove draw record idx (map fixed up for the moved record). */
static void sim_draw_remove(SimState *s, int64_t idx) {
    SimDraw *d = &s->draws[idx];
    sim_map_remove(s, d->id);
    free(d->residual);
    s->d_len--;
    if (idx != s->d_len) {
        *d = s->draws[s->d_len];
        sim_map_remove(s, d->id);
        s->map_used += sim_map_place(s->map, s->map_cap, d->id,
                                     (uint64_t)idx);
    }
}

/* ---- outcome-slot sets (usually singletons) ---- */

static int sim_slot_reserve(SimState *s, uint64_t ot) {
    if ((int64_t)ot < s->s_cap)
        return 0;
    {
        int64_t nc = s->s_cap ? s->s_cap : 64;
        SimSlot *ns;
        while (nc <= (int64_t)ot)
            nc *= 2;
        ns = (SimSlot *)realloc(s->slots, (size_t)nc * sizeof(SimSlot));
        if (!ns)
            return ERR_OOM;
        memset(ns + s->s_cap, 0, (size_t)(nc - s->s_cap) * sizeof(SimSlot));
        s->slots = ns;
        s->s_cap = nc;
    }
    return 0;
}

static int sim_slot_add(SimSlot *ss, uint64_t id) {
    if (ss->len == ss->cap) {
        int64_t nc = ss->cap ? ss->cap * 2 : 2;
        uint64_t *ni = (uint64_t *)realloc(ss->ids, (size_t)nc * 8);
        if (!ni)
            return ERR_OOM;
        ss->ids = ni;
        ss->cap = nc;
    }
    ss->ids[ss->len++] = id;
    return 0;
}

static int64_t sim_slot_pos(const SimSlot *ss, uint64_t id) {
    int64_t i;
    for (i = 0; i < ss->len; i++)
        if (ss->ids[i] == id)
            return i;
    return -1;
}

/* ---- strict-ledger transitions (mirror sim.py exactly) ---- */

/* Simulator._retire_slot: drop slot ot's draw set ahead of a fresh HMR
 * store. Returns 0 ok, 1 = halt (a draw's last carrier vanishes with a
 * nonzero residual — the Python replay raises the CleanlinessError). */
static int sim_retire_slot(Kern *k, uint64_t ot) {
    SimState *s = k->sim;
    SimSlot *ss;
    int64_t i;
    if ((int64_t)ot >= s->s_cap)
        return 0;
    ss = &s->slots[ot];
    for (i = 0; i < ss->len; i++) {
        int64_t idx = sim_map_find(s, ss->ids[i]);
        SimDraw *d;
        if (idx < 0)
            return 1;   /* invariant breach: replay decides */
        d = &s->draws[idx];
        if (--d->refs == 0) {
            int64_t w;
            for (w = 0; w < k->n_words; w++)
                if (d->residual[w])
                    return 1;
            sim_draw_remove(s, idx);
        }
    }
    ss->len = 0;
    return 0;
}

/* Conditioned Z/CZ/NEG fixup: XOR the draw-coefficient mask
 *     m = live & bits[cc]? & <value terms>          (NO outs[oc] factor —
 * the residual tracks the coefficient of the draw variable, sim.py Z/CZ/NEG)
 * into every draw slot oc carries. All wires already ensured by exec_op. */
static int sim_fixup(Kern *k, uint32_t kind, uint64_t qt, uint64_t qb,
                     uint64_t oc, uint64_t cc) {
    SimState *s = k->sim;
    SimSlot *ss;
    uint64_t *m = s->scratch;
    int64_t w, i;
    if ((int64_t)oc >= s->s_cap || s->slots[oc].len == 0)
        return 0;
    ss = &s->slots[oc];
    for (w = 0; w < k->n_words; w++)
        m[w] = k->live[w];
    if (cc != NO_ID) {
        const uint64_t *bp = BP(k, cc);
        for (w = 0; w < k->n_words; w++)
            m[w] &= bp[w];
    }
    if (kind == K_Z || kind == K_CZ) {
        const uint64_t *tp = QP(k, qt);
        for (w = 0; w < k->n_words; w++)
            m[w] &= tp[w];
    }
    if (kind == K_CZ) {
        const uint64_t *ap = QP(k, qb);
        for (w = 0; w < k->n_words; w++)
            m[w] &= ap[w];
    }
    for (i = 0; i < ss->len; i++) {
        int64_t idx = sim_map_find(s, ss->ids[i]);
        SimDraw *d;
        if (idx < 0)
            return 1;   /* invariant breach: replay decides */
        d = &s->draws[idx];
        for (w = 0; w < k->n_words; w++)
            d->residual[w] ^= m[w];
    }
    return 0;
}

/* OXOR slot merge (sim.py: GF(2) union with cancellation; refs cannot hit
 * zero here because oc itself still carries every id it contributes). */
static int sim_oxor(Kern *k, uint64_t ot, uint64_t oc) {
    SimState *s = k->sim;
    SimSlot *src, *dst;
    int64_t i;
    int rc;
    if ((int64_t)oc >= s->s_cap || s->slots[oc].len == 0)
        return 0;
    if ((rc = sim_slot_reserve(s, ot)) != 0)
        return rc;
    src = &s->slots[oc];
    dst = &s->slots[ot];
    for (i = 0; i < src->len; i++) {
        uint64_t id = src->ids[i];
        int64_t pos = sim_slot_pos(dst, id);
        int64_t idx = sim_map_find(s, id);
        SimDraw *d;
        if (idx < 0)
            return 1;   /* invariant breach: replay decides */
        d = &s->draws[idx];
        if (pos >= 0) {
            dst->ids[pos] = dst->ids[--dst->len];
            d->refs--;
        } else {
            if ((rc = sim_slot_add(dst, id)) != 0)
                return rc;
            d->refs++;
        }
    }
    return 0;
}

/* ---- per-op operand check (Simulator fast path) ----
 * The rules of uncompute.ops.op_violation, read from the (required, optional)
 * slot masks that kern_sim_init receives (uncompute.ops.operand_mask_table;
 * bit s-1 stands for tuple slot s). Returns nonzero when the op breaks a
 * rule; the Python replay then words the error. */
static int sim_validate(const SimState *s, const uint64_t *o) {
    const uint64_t kind = o[0];
    uint32_t present = 0, need, may;
    int i;
    if (kind >= K_COUNT)
        return 1;
    for (i = 1; i <= 8; i++)
        if (o[i] != NO_ID)
            present |= 1u << (i - 1);
    /* distinct ids per space: qubits (slots 1-3), outcomes (4-5), bits (7-8) */
    if (o[1] != NO_ID && (o[1] == o[2] || o[1] == o[3]))
        return 1;
    if (o[2] != NO_ID && o[2] == o[3])
        return 1;
    if (o[4] != NO_ID && o[4] == o[5])
        return 1;
    if (o[7] != NO_ID && o[7] == o[8])
        return 1;
    need = s->sig[kind][0];
    may = s->sig[kind][1];
    if ((present & need) != need)
        return 1;
    if (present & ~(need | may))
        return 1;
    return 0;
}

/* ---- context setup / bulk state I/O ---- */

int64_t kern_sim_init(void *vk, const uint8_t *seed, int64_t seed_len,
                      int64_t strict, const uint8_t *sig34) {
    Kern *k = (Kern *)vk;
    SimState *s;
    if (seed_len < 1 || seed_len > 32)
        return ERR_OOM;
    s = (SimState *)calloc(1, sizeof(SimState));
    if (!s)
        return ERR_OOM;
    s->scratch = (uint64_t *)malloc((size_t)k->n_words * 8);
    if (!s->scratch) {
        free(s);
        return ERR_OOM;
    }
    memcpy(s->seed, seed, (size_t)seed_len);
    s->seed_len = seed_len;
    s->strict = strict;
    memcpy(s->sig, sig34, K_COUNT * 2);
    k->sim = s;
    return 0;
}

int64_t kern_sim_load_state(void *vk, const uint64_t *qw, int64_t nq,
                            const uint64_t *ow, int64_t no,
                            const uint64_t *bw, int64_t nb,
                            const uint64_t *phase, uint64_t toffoli,
                            uint64_t clifford, uint64_t ctr, uint64_t op_n) {
    Kern *k = (Kern *)vk;
    int rc;
    if (nq > 0) {
        if ((rc = ENSURE_Q(k, (uint64_t)(nq - 1))) != 0)
            return rc;
        memcpy(k->qubits, qw, (size_t)nq * k->n_words * 8);
    }
    if (no > 0) {
        if ((rc = ENSURE_O(k, (uint64_t)(no - 1))) != 0)
            return rc;
        memcpy(k->outs, ow, (size_t)no * k->n_words * 8);
    }
    if (nb > 0) {
        if ((rc = ENSURE_B(k, (uint64_t)(nb - 1))) != 0)
            return rc;
        memcpy(k->bits, bw, (size_t)nb * k->n_words * 8);
    }
    memcpy(k->phase, phase, (size_t)k->n_words * 8);
    k->toffoli = toffoli;
    k->clifford = clifford;
    k->sim->ctr = ctr;
    k->sim->op_n = op_n;
    return 0;
}

/* Ledger load: draws4 = (id, refs, opn, out_id) per draw, residuals =
 * n_draws x n_words, slots = flat [out_id, count, id...] runs. */
int64_t kern_sim_load_ledger(void *vk, const uint64_t *draws4,
                             int64_t n_draws, const uint64_t *residuals,
                             const uint64_t *slots, int64_t slots_len) {
    Kern *k = (Kern *)vk;
    SimState *s = k->sim;
    int64_t i, p;
    int rc;
    for (i = 0; i < n_draws; i++) {
        if ((rc = sim_draw_insert(k, draws4[4 * i], draws4[4 * i + 2],
                                  draws4[4 * i + 3],
                                  residuals + i * k->n_words,
                                  draws4[4 * i + 1])) != 0)
            return rc;
    }
    for (p = 0; p < slots_len; ) {
        uint64_t ot = slots[p];
        int64_t n = (int64_t)slots[p + 1];
        p += 2;
        if ((rc = sim_slot_reserve(s, ot)) != 0)
            return rc;
        for (i = 0; i < n; i++)
            if ((rc = sim_slot_add(&s->slots[ot], slots[p + i])) != 0)
                return rc;
        p += n;
    }
    return 0;
}

/* One op through the sim engine — validate, strict-pre, exec, strict-post,
 * events, count — exactly sim.py apply's per-op body. Returns 0 applied,
 * nonzero = halt (caller discards the context and replays in Python). */
static int sim_step(Kern *k, const uint64_t *o) {
    SimState *s = k->sim;
    uint64_t kind = o[0];
    int64_t w;
    if (sim_validate(s, o))
        return 1;
    if (s->strict && kind == K_HMR) {
        /* pre-measure value (the residual a clean teardown must repay),
         * captured before exec_op demolishes the target */
        if (ENSURE_Q(k, o[3]) != 0 || ENSURE_O(k, o[4]) != 0)
            return 1;
        {
            const uint64_t *tp = QP(k, o[3]);
            for (w = 0; w < k->n_words; w++)
                s->scratch[w] = tp[w] & k->live[w];
        }
        if (sim_retire_slot(k, o[4]))
            return 1;
    } else if (s->strict && kind == K_R) {
        uint64_t acc = 0;
        if (ENSURE_Q(k, o[3]) != 0)
            return 1;
        {
            const uint64_t *tp = QP(k, o[3]);
            for (w = 0; w < k->n_words; w++)
                acc |= tp[w] & k->live[w];
        }
        if (acc)
            return 1;   /* reset of a dirty wire: unfixable phase kick */
    }
    if (exec_op(k, (uint32_t)kind, o[1], o[2], o[3], o[4], o[5],
                o[7], o[8]) != 0)
        return 1;
    if (s->strict) {
        if (kind == K_HMR) {
            if (sim_slot_reserve(s, o[4]) != 0)
                return 1;
            s->slots[o[4]].len = 0;
            if (sim_slot_add(&s->slots[o[4]], s->ctr) != 0)
                return 1;
            if (sim_draw_insert(k, s->ctr, s->op_n, o[4], s->scratch,
                                1) != 0)
                return 1;
        } else if ((kind == K_Z || kind == K_CZ || kind == K_NEG)
                   && o[5] != NO_ID) {
            if (sim_fixup(k, (uint32_t)kind, o[3], o[2], o[5], o[8]) != 0)
                return 1;
        } else if (kind == K_OXOR) {
            if (sim_oxor(k, o[4], o[5]) != 0)
                return 1;
        }
    }
    s->op_n++;
    return 0;
}

/* ---- the flat walk ----
 * ops = n_ops x 9 u64 (kind, qa, qb, qt, ot, oc, rt, ct, cc), exactly the
 * Simulator op tuples. Returns -1 = all applied, else the local index of
 * the op where the walk halted (validation reject, strict violation,
 * OOM) — the caller replays the WHOLE chunk in Python from the untouched
 * resting state. Billing/validation order matches sim.py apply: validate,
 * bill, execute. */
int64_t kern_sim_apply(void *vk, const uint64_t *ops, int64_t n_ops) {
    Kern *k = (Kern *)vk;
    int64_t i;
    for (i = 0; i < n_ops; i++)
        if (sim_step(k, ops + i * 9))
            return i;
    return -1;
}

/* ---- readback ---- */

/* out10: q_alloc, o_alloc, b_alloc, toffoli, clifford, ctr, op_n, n_draws,
 * slots_flat_len, ev_len */
void kern_sim_sizes(void *vk, uint64_t *out10) {
    Kern *k = (Kern *)vk;
    SimState *s = k->sim;
    int64_t i, slots_len = 0;
    for (i = 0; i < s->s_cap; i++)
        if (s->slots[i].len)
            slots_len += 2 + s->slots[i].len;
    out10[0] = (uint64_t)k->q_alloc;
    out10[1] = (uint64_t)k->o_alloc;
    out10[2] = (uint64_t)k->b_alloc;
    out10[3] = k->toffoli;
    out10[4] = k->clifford;
    out10[5] = s->ctr;
    out10[6] = s->op_n;
    out10[7] = (uint64_t)s->d_len;
    out10[8] = (uint64_t)slots_len;
    out10[9] = (uint64_t)k->ev_len;
}

void kern_sim_dump_state(void *vk, uint64_t *qw, uint64_t *ow, uint64_t *bw,
                         uint64_t *phase) {
    Kern *k = (Kern *)vk;
    if (k->q_alloc)
        memcpy(qw, k->qubits, (size_t)k->q_alloc * k->n_words * 8);
    if (k->o_alloc)
        memcpy(ow, k->outs, (size_t)k->o_alloc * k->n_words * 8);
    if (k->b_alloc)
        memcpy(bw, k->bits, (size_t)k->b_alloc * k->n_words * 8);
    memcpy(phase, k->phase, (size_t)k->n_words * 8);
}

static int sim_cmp_u64(const void *a, const void *b) {
    uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;
    return x < y ? -1 : x > y ? 1 : 0;
}

static int sim_cmp_draw(const void *a, const void *b) {
    uint64_t x = ((const SimDraw *)a)->id, y = ((const SimDraw *)b)->id;
    return x < y ? -1 : x > y ? 1 : 0;
}

/* Ledger dump, deterministically ordered: draws ascending by id (creation
 * order — ids are the monotonic draw counter), slot runs ascending by
 * outcome id with ids ascending inside each run. Formats as in load.
 * TERMINAL: sorts the draws array in place, invalidating the id map — the
 * context must be freed (not applied further) after this call. */
void kern_sim_dump_ledger(void *vk, uint64_t *draws4, uint64_t *residuals,
                          uint64_t *slots) {
    Kern *k = (Kern *)vk;
    SimState *s = k->sim;
    int64_t i, p;
    qsort(s->draws, (size_t)s->d_len, sizeof(SimDraw), sim_cmp_draw);
    for (i = 0; i < s->d_len; i++) {
        const SimDraw *d = &s->draws[i];
        draws4[4 * i] = d->id;
        draws4[4 * i + 1] = d->refs;
        draws4[4 * i + 2] = d->opn;
        draws4[4 * i + 3] = d->out_id;
        memcpy(residuals + i * k->n_words, d->residual,
               (size_t)k->n_words * 8);
    }
    p = 0;
    for (i = 0; i < s->s_cap; i++) {
        SimSlot *ss = &s->slots[i];
        if (!ss->len)
            continue;
        qsort(ss->ids, (size_t)ss->len, 8, sim_cmp_u64);
        slots[p] = (uint64_t)i;
        slots[p + 1] = (uint64_t)ss->len;
        memcpy(slots + p + 2, ss->ids, (size_t)ss->len * 8);
        p += 2 + ss->len;
    }
}

/* ---- the recorded (loop-form) walk ----
 *
 * Native mirror of compact.iter_expand_item over the Builder's RECORDED
 * items — Python cost scales with the description, not the expansion.
 * Packed program (all u64 words; uncompute._simfast._pack_recorded):
 *
 *   RAW:   0, kind, qa, qb, qt, ot, oc, rt, ct, cc      (depth 0 only)
 *   OPREC: 1, kind, amask, gflag,
 *          [gflag: table_id, offset, D x stride],
 *          per amask slot ascending (tuple slots 1..8): base, D x stride
 *   LOOP:  2, count, body_words, <body items>
 *
 * D = enclosing loop depth at pack time; strides/offsets/bases of OPREC
 * fields are two's-complement i64; RAW fields are verbatim u64 (sentinels
 * included). Resolution is base + sum(stride*idx) in 128-bit, range-checked
 * to [0, 2^32) exactly like the reference; a gated-off op skips resolution
 * entirely (iter_expand_item's order). Any out-of-range resolution, bad
 * table/bit, malformed program, or sim_step halt aborts the whole run —
 * the caller replays through iter_expanded()/_apply_py. Tables buffer:
 * n, then (len_bits, word_offset) per table, then the bitmap words. */

typedef struct {
    const uint64_t *buf;
    int64_t len;            /* words */
    int64_t n_tables;
} SimTables;

static int sim_rec_items(Kern *k, const uint64_t *p, const uint64_t *end,
                         const SimTables *tb, int64_t *idx, int depth) {
    while (p < end) {
        uint64_t tag = p[0];
        if (tag == 0) {                                   /* RAW */
            if (end - p < 10 || depth != 0)
                return 1;
            if (sim_step(k, p + 1))
                return 1;
            p += 10;
        } else if (tag == 1) {                            /* OPREC */
            uint64_t kind, amask, gflag;
            uint64_t o[9];
            int gated_off = 0;
            int slot, d;
            if (end - p < 4)
                return 1;
            kind = p[1];
            amask = p[2];
            gflag = p[3];
            p += 4;
            if (gflag) {
                __int128 b;
                uint64_t tid;
                if (end - p < 2 + depth)
                    return 1;
                tid = p[0];
                b = (__int128)(int64_t)p[1];
                for (d = 0; d < depth; d++)
                    b += (__int128)(int64_t)p[2 + d] * idx[d];
                p += 2 + depth;
                if (tid >= (uint64_t)tb->n_tables)
                    return 1;
                {
                    int64_t nbits = (int64_t)tb->buf[1 + 2 * tid];
                    int64_t off = (int64_t)tb->buf[2 + 2 * tid];
                    if (b < 0 || b >= nbits)
                        return 1;
                    if (!((tb->buf[off + (int64_t)(b >> 6)]
                           >> ((int64_t)b & 63)) & 1))
                        gated_off = 1;
                }
            }
            if (gated_off) {
                /* iter_expand_item returns before resolving operands */
                for (slot = 1; slot <= 8; slot++)
                    if (amask & (1u << (slot - 1)))
                        p += 1 + depth;
                if (p > end)
                    return 1;
                continue;
            }
            o[0] = kind;
            for (slot = 1; slot <= 8; slot++)
                o[slot] = NO_ID;
            for (slot = 1; slot <= 8; slot++) {
                if (!(amask & (1u << (slot - 1))))
                    continue;
                if (end - p < 1 + depth)
                    return 1;
                {
                    __int128 v = (__int128)(int64_t)p[0];
                    for (d = 0; d < depth; d++)
                        v += (__int128)(int64_t)p[1 + d] * idx[d];
                    p += 1 + depth;
                    if (v < 0 || v > (__int128)0xFFFFFFFFULL)
                        return 1;
                    o[slot] = (uint64_t)v;
                }
            }
            if (sim_step(k, o))
                return 1;
        } else if (tag == 2) {                            /* LOOP */
            uint64_t count, body;
            int64_t t;
            if (end - p < 3 || depth >= 8)
                return 1;
            count = p[1];
            body = p[2];
            p += 3;
            if ((uint64_t)(end - p) < body)
                return 1;
            for (t = 0; t < (int64_t)count; t++) {
                idx[depth] = t;
                if (sim_rec_items(k, p, p + body, tb, idx, depth + 1))
                    return 1;
            }
            p += body;
        } else {
            return 1;                                     /* malformed */
        }
    }
    return 0;
}

int64_t kern_sim_run_rec(void *vk, const uint64_t *prog, int64_t prog_len,
                         const uint64_t *tables, int64_t tables_len) {
    Kern *k = (Kern *)vk;
    SimTables tb;
    int64_t idx[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    tb.buf = tables;
    tb.len = tables_len;
    tb.n_tables = tables_len > 0 ? (int64_t)tables[0] : 0;
    if (tb.n_tables < 0 || 1 + 2 * tb.n_tables > tables_len)
        return 0;
    if (sim_rec_items(k, prog, prog + prog_len, &tb, idx, 0))
        return 0;
    return -1;
}
