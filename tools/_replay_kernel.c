/* Optional C core of tools/liveness_profile.py's builder replay.
 *
 * replay_rows() walks a run of packed flat rows (the Builder's _OpTape
 * record: 9 u32 slots per op) and records their activity effects: which
 * allocation episode each qubit operand belongs to, the span it is touched
 * in, and the |0>-state changes of written and demolished wires. It is a
 * line-for-line C version of the per-item replay (_scalar) for flat
 * tuples; the Python path stays the reference and the fallback.
 *
 * Build: make replay-kernel (loaded by liveness_profile.py via ctypes).
 */
#include <stdint.h>

#define REPLAY_KERNEL_ABI 1

int64_t replay_abi(void) { return REPLAY_KERNEL_ABI; }

/* info[slot0] bits: 0-2 slots 1-3 are qubit operands, 3-5 slots 1-3 are
 * written, 6 slot 3 is the zeroing target.
 *
 * Events before row i apply before it: span changes (seg_pos/seg_sp, the
 * touch span of rows from seg_pos on) and episode changes (mod_*: wire
 * mod_q opens episode mod_e >= 0 in state mod_init at op index mod_x, or
 * closes with mod_e = -1).
 *
 * Outputs append to touch_e/touch_sp (first touch of an episode per span)
 * and tl_e/tl_x/tl_z (timeline entries: the alloc entry, then each state
 * change); *n_touch and *n_tl return the counts. Capacities: 3 n touches,
 * 3 n + n_mod timeline entries. */
void replay_rows(const uint32_t *rows, int64_t n, const uint8_t *info,
                 int64_t x0,
                 const int64_t *seg_pos, const int64_t *seg_sp, int64_t n_seg,
                 const int64_t *mod_pos, const int64_t *mod_q,
                 const int64_t *mod_e, const int8_t *mod_init,
                 int64_t n_mod,
                 int64_t *ep_of, int64_t nq, int8_t *ep_state,
                 int64_t *ep_last,
                 int64_t *touch_e, int64_t *touch_sp, int64_t *n_touch,
                 int64_t *tl_e, int64_t *tl_x, int8_t *tl_z, int64_t *n_tl)
{
    int64_t nt = 0, nl = 0, mi = 0, si = 0;
    for (int64_t i = 0; i <= n; i++) {
        while (mi < n_mod && mod_pos[mi] <= i) {
            int64_t q = mod_q[mi], e = mod_e[mi];
            ep_of[q] = e;
            if (e >= 0) {
                ep_state[e] = mod_init[mi];
                ep_last[e] = -1;
                tl_e[nl] = e;
                tl_x[nl] = x0 + mod_pos[mi];
                tl_z[nl] = mod_init[mi];
                nl++;
            }
            mi++;
        }
        if (i == n)
            break;
        while (si + 1 < n_seg && seg_pos[si + 1] <= i)
            si++;
        const uint32_t *r = rows + 9 * i;
        uint8_t inf = info[r[0] & 0xFFFF];
        if (!inf)
            continue;
        int64_t sp = seg_sp[si], x = x0 + i;
        for (int f = 1; f <= 3; f++) {
            if (!(inf & (1 << (f - 1))))
                continue;
            int64_t q = r[f];
            if (q >= nq)
                continue;
            int64_t e = ep_of[q];
            if (e < 0)
                continue;  /* episode closed (recycle()'s own R) */
            if (ep_last[e] != sp) {
                ep_last[e] = sp;
                touch_e[nt] = e;
                touch_sp[nt] = sp;
                nt++;
            }
            if ((inf & (8 << (f - 1))) && ep_state[e]) {
                ep_state[e] = 0;  /* written: zero lapses */
                tl_e[nl] = e;
                tl_x[nl] = x;
                tl_z[nl] = 0;
                nl++;
            }
        }
        if (inf & 64) {
            int64_t q = r[3];
            int64_t e = q < nq ? ep_of[q] : -1;
            if (e >= 0 && !ep_state[e]) {
                ep_state[e] = 1;  /* demolition leaves |0> */
                tl_e[nl] = e;
                tl_x[nl] = x;
                tl_z[nl] = 1;
                nl++;
            }
        }
    }
    *n_touch = nt;
    *n_tl = nl;
}
