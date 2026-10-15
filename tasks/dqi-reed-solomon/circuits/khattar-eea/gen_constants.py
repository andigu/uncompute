#!/usr/bin/env python3
"""Regenerate constants.json for the dqi-reed-solomon khattar-eea circuit.

constants.json holds the one extracted value the circuit consumes: the
flattened gate list of the released decoder's quantum-quantum multiplier,
dqi GF2MulViaOptimizedPolyMul(QGF(2, 12)) (51 Toffoli + 1580 CNOT).
main() extracts it from the Khattar et al. Zenodo release
(10.5281/zenodo.17301475, CC-BY-4.0) and writes the file.  The script
downloads dqi.zip once into ~/.cache/uncompute/ and checks its sha256.
It needs Qualtran at the pinned commit, which conflicts with the uncompute
venv's numpy, so it runs in a throwaway uv environment.

RUN: uv run --no-project --isolated --python 3.12 \
         --with "qualtran @ git+https://github.com/quantumlib/Qualtran@a68adbb8709d8b63025dd9a7902008f4cbec22d7" \
         --with galois python gen_constants.py
"""

import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
PATH = HERE / "constants.json"

QUALTRAN_COMMIT = "a68adbb8709d8b63025dd9a7902008f4cbec22d7"
DQI_URL = "https://zenodo.org/api/records/17301475/files/dqi.zip/content"
DQI_SHA256 = "3fed91f7643ad2aabeb212d5cad2f01b4f8f3e50d4b2aec9ec8838b1e46d4013"
DQI_CACHE = pathlib.Path.home() / ".cache" / "uncompute" / "zenodo-17301475"
B = 12  # field degree; GF(2^12), galois default poly 4331 = x^12+x^7+x^6+x^5+x^3+x+1
GF_POLY = 4331
EXPECT_CCX, EXPECT_CX = 51, 1580  # 3 x poly_mul_6 (17 Toffoli, 482 CNOT each) + linear layers
OUT0 = 2 * B  # out[i] = wire OUT0 + i; x[i] = wire i; y[i] = wire B + i


def encode(gates):
    """Return the canonical file bytes: one gate per line."""
    body = ",\n".join("  " + json.dumps(g) for g in gates)
    return ('{\n "gates": [\n' + body + "\n ]\n}\n").encode()


# -------------------------------------------------------------------------
# Extraction of the gate list
# -------------------------------------------------------------------------


def _dqi_on_path():
    """Put the dqi package of the Zenodo release on sys.path, downloading
    and sha256-checking dqi.zip into DQI_CACHE on first use.
    """
    import hashlib
    import ssl
    import urllib.request
    import zipfile

    src = DQI_CACHE / "src"
    if not (src / "dqi").is_dir():
        DQI_CACHE.mkdir(parents=True, exist_ok=True)
        try:  # python.org builds ship no CA bundle; certifi rides with qualtran
            import certifi

            ctx = ssl.create_default_context(cafile=certifi.where())
        except ImportError:
            ctx = ssl.create_default_context()
        blob = urllib.request.urlopen(DQI_URL, context=ctx).read()
        assert hashlib.sha256(blob).hexdigest() == DQI_SHA256, (
            "dqi.zip sha256 != the pinned Zenodo archive"
        )
        (DQI_CACHE / "dqi.zip").write_bytes(blob)
        zipfile.ZipFile(DQI_CACHE / "dqi.zip").extractall(DQI_CACHE)
    sys.path.insert(0, str(src))


def extract():  # noqa: C901, PLR0915 - one linear bloq-flattening walk
    """Flatten dqi GF2MulViaOptimizedPolyMul(QGF(2, 12)) into the gate list.

    This is the multiplier the released decoder uses:
    poly_eea_zalka_bloqs.py imports ``GF2MulViaOptimizedPolyMul as
    GF2MulViaKaratsuba`` (an alias that shadows the qualtran class of the
    same name).  The bloq builds on the Maslov-optimized hard-coded circuits
    in ``dqi/gf_arithmetic/circuits/poly_mul_6.txt`` (3 x 17 = 51 Toffoli at
    b = 12).

    Flatten inventory at the pinned commit: 51 Toffoli + 1580 CNOT + 1
    Allocate (the 12-qubit result register), no workspace ancillas.
    A full flatten does not reproduce Adjoint(GF2MulK) at the pinned
    commit, so this function stops the flatten at Adjoint(GF2MulK) and
    expands each instance as the verified inverse of its forward twin.

    Emitted wire numbering (little-endian uncompute convention, B = 12):
    0..11 x[i] = coeff of x^i (THRU); 12..23 y[i] (THRU); 24..35 out[i]
    (RIGHT: born |0>).
    """
    _dqi_on_path()
    try:
        from dqi.gf_arithmetic.gf_multiplication import GF2MulViaOptimizedPolyMul
        from qualtran import QGF
        from qualtran._infra.quantum_graph import DanglingT, LeftDangle, RightDangle
        from qualtran.bloqs.gf_arithmetic.gf2_multiplication import SynthesizeLRCircuit
    except ImportError as exc:
        raise SystemExit(
            f"gen_constants.py needs Qualtran pinned at {QUALTRAN_COMMIT}; run it "
            "with the uv command in this file's docstring "
            f"(import failed: {exc})"
        ) from exc

    qgf = QGF(2, B)
    assert int(qgf.gf_type.irreducible_poly) == GF_POLY
    assert int(qgf.gf_type.primitive_element) == 2
    bloq = GF2MulViaOptimizedPolyMul(qgf)

    # GF2MulK decomposes into SynthesizeLRCircuit.  A full flatten does not
    # reproduce Adjoint(GF2MulK) (the inverse constant-multiply) at the pinned
    # commit, so this function keeps Adjoint(GF2MulK) instances unflattened.
    # It expands each as the exact inverse of its forward twin's flatten: undo
    # the PLU exit permutation (wire relabeling), then reverse the forward
    # CNOT list.  A per-instance check compares each expansion against
    # on_classical_vals on all basis vectors.
    from qualtran.bloqs.gf_arithmetic.gf2_multiplication import GF2MulK

    def _is_adjoint_mulk(b) -> bool:
        sub = getattr(b, "subbloq", None)  # Adjoint wrapper (GF2MulK has no is_adjoint attr)
        return isinstance(sub, GF2MulK)

    def walk(cb, wire_of, on_cx, on_other):
        """Track wire tuples through cb's Split/Join plumbing and CNOTs
        (reported as on_cx(c, t)); hand every other bloq instance to
        on_other(binst, bloq, name, in_soq, set_out).
        """
        for binst, pred_cxns, succ_cxns in cb.iter_bloqnections():
            for cxn in pred_cxns:
                wire_of[cxn.right] = wire_of[cxn.left]
            b = binst.bloq
            name = b.__class__.__name__

            def in_soq(regname, idx=()):
                for cxn in pred_cxns:
                    if cxn.right.reg.name == regname and cxn.right.idx == idx:
                        return wire_of[cxn.right]
                raise KeyError((binst, regname, idx))

            def set_out(regname, idx, wires):
                for cxn in succ_cxns:
                    if cxn.left.reg.name == regname and cxn.left.idx == idx:
                        wire_of[cxn.left] = tuple(wires)
                        return
                raise KeyError((binst, regname, idx))

            if name == "Split":
                w = in_soq("reg")
                for i in range(len(w)):
                    set_out("reg", (i,), (w[i],))
            elif name == "Join":
                n = b.signature[0].shape[0]
                set_out("reg", (), [in_soq("reg", (i,))[0] for i in range(n)])
            elif name == "CNOT":
                c, t = in_soq("ctrl")[0], in_soq("target")[0]
                on_cx(c, t)
                set_out("ctrl", (), (c,))
                set_out("target", (), (t,))
            else:
                on_other(binst, b, name, in_soq, set_out)

    def _mulk_forward_cnots_and_perm(fwd: GF2MulK):
        """(cnots, perm) of the forward MulK flatten in slot space (MSB-first).

        cnots = ordered [(c_slot, t_slot)]; perm[i] = the entry slot that the
        forward circuit's exit position i carries (the PLU relabeling).
        """
        fcb = fwd.as_composite_bloq().flatten()
        wire_of = {}
        nbits = fwd.signature[0].dtype.num_qubits
        for cxn in fcb.connections:
            if isinstance(cxn.left.binst, DanglingT):
                wire_of[cxn.left] = tuple(range(nbits))
        cnots = []

        def reject(_binst, _b, name, _in_soq, _set_out):
            raise NotImplementedError(f"forward MulK flatten contains {name}")

        walk(fcb, wire_of, lambda c, t: cnots.append((c, t)), reject)
        # exit wiring: perm[i] = entry slot carried at exit position i
        perm = None
        for cxn in fcb.connections:
            if isinstance(cxn.right.binst, DanglingT):
                perm = list(wire_of[cxn.left])
        assert perm is not None and sorted(perm) == list(range(nbits))
        return cnots, perm

    cb = bloq.as_composite_bloq().flatten(lambda binst: not _is_adjoint_mulk(binst.bloq))
    n_adj_mulk = sum(1 for bi in cb.bloq_instances if _is_adjoint_mulk(bi.bloq))
    print("unflattened adjoint GF2MulK instances:", n_adj_mulk)
    assert n_adj_mulk >= 1
    assert not any(
        isinstance(bi.bloq, SynthesizeLRCircuit)
        or isinstance(getattr(bi.bloq, "subbloq", None), SynthesizeLRCircuit)
        for bi in cb.bloq_instances
    ), "SynthesizeLRCircuit appeared; re-check the adjoint expansion"

    wire_of = {}
    for cxn in cb.connections:
        if isinstance(cxn.left.binst, DanglingT):
            assert cxn.left.binst is LeftDangle
            reg = cxn.left.reg
            assert reg.name in ("x", "y") and cxn.left.idx == ()
            base = 0 if reg.name == "x" else B
            wire_of[cxn.left] = tuple(range(base, base + B))
    next_wire = 2 * B
    events = []

    def fresh(n):
        nonlocal next_wire
        w = tuple(range(next_wire, next_wire + n))
        next_wire += n
        return w

    def on_other(_binst, b, name, in_soq, set_out):
        if name == "Allocate":
            w = fresh(b.dtype.num_qubits)
            events.append(("alloc", w))
            set_out("reg", (), w)
        elif name == "Free":
            events.append(("free", in_soq("reg")))
        elif name == "Toffoli":
            c0 = in_soq("ctrl", (0,))[0]
            c1 = in_soq("ctrl", (1,))[0]
            t = in_soq("target")[0]
            events.append(("ccx", c0, c1, t))
            set_out("ctrl", (0,), (c0,))
            set_out("ctrl", (1,), (c1,))
            set_out("target", (), (t,))
        elif _is_adjoint_mulk(b):
            cnots, perm = _mulk_forward_cnots_and_perm(b.subbloq)
            w_in = in_soq("g")
            nbits = len(w_in)
            # adjoint = (P o L)^-1: undo the exit relabeling (v[perm[i]] = w_in[i]),
            # then the forward CNOT list reversed, exit position s carries slot s.
            v = [None] * nbits
            for i in range(nbits):
                v[perm[i]] = w_in[i]
            events.extend(("cx", v[c], v[t]) for (c, t) in reversed(cnots))
            set_out("g", (), tuple(v))
            # self-contained check: (our adjoint expansion) o (forward) == identity
            # on every basis vector, in slot space.
            for k in range(nbits):
                state = [0] * nbits  # state[slot]
                state[perm[k]] = 1  # adjoint entry position k
                for c, t in reversed(cnots):
                    state[t] ^= state[c]
                for c, t in cnots:  # now the forward circuit
                    state[t] ^= state[c]
                out_positions = [state[perm[i]] for i in range(nbits)]
                want = [1 if i == k else 0 for i in range(nbits)]
                assert out_positions == want, f"adjoint MulK expansion not inverse on basis {k}"
        else:
            raise NotImplementedError(name)

    walk(cb, wire_of, lambda c, t: events.append(("cx", c, t)), on_other)

    finals = {}
    for cxn in cb.connections:
        if isinstance(cxn.right.binst, DanglingT):
            assert cxn.right.binst is RightDangle
            finals[cxn.right.reg.name] = wire_of[cxn.left]

    assert finals["x"] == tuple(range(B)), finals["x"]
    assert finals["y"] == tuple(range(B, 2 * B)), finals["y"]
    result_raw = finals["result"]

    # Qualtran QGF wires are MSB-first: flip each register to little-endian.
    remap = {}
    for w in range(B):
        remap[w] = B - 1 - w
        remap[B + w] = B + B - 1 - w
    for pos, w in enumerate(result_raw):
        remap[w] = OUT0 + (B - 1 - pos)
    assert len(remap) == 3 * B and next_wire == 3 * B, (len(remap), next_wire)

    gates = []
    for ev in events:
        if ev[0] == "alloc":
            ws = tuple(sorted(remap[w] for w in ev[1]))
            assert all(OUT0 <= w < 3 * B for w in ws), ws  # only the result alloc exists
            gates.append(("alloc_result", ws))
        elif ev[0] == "free":
            raise AssertionError("unexpected free event")
        elif ev[0] == "cx":
            gates.append(("cx", remap[ev[1]], remap[ev[2]]))
        else:
            gates.append(("ccx", remap[ev[1]], remap[ev[2]], remap[ev[3]]))

    n_ccx = sum(1 for g in gates if g[0] == "ccx")
    n_cx = sum(1 for g in gates if g[0] == "cx")
    print(f"counts: ccx={n_ccx} cx={n_cx}")
    assert n_ccx == EXPECT_CCX and n_cx == EXPECT_CX, (n_ccx, n_cx)
    return [
        list(g) if g[0] != "alloc_result" else [g[0], list(g[1])]  # ty: ignore[invalid-argument-type]  # g[1] is the ws tuple
        for g in gates
    ]


def main():
    new = encode(extract())
    old = PATH.read_bytes() if PATH.exists() else None
    PATH.write_bytes(new)
    print("constants.json %s" % ("unchanged" if new == old else "WRITTEN"))


if __name__ == "__main__":
    main()
