#!/usr/bin/env python3
"""divstep circuit variant: the decode ladder at store threshold
v2(fv') >= 9 (algebraic branch reconstruction through
j = 8). Cost: 7,866,779 Toffoli x 893 qubits.
"""

import pathlib

from uncompute.testing import load_module

HERE = pathlib.Path(__file__).resolve().parent

E = load_module(HERE / "engine.py", "divstep_engine")
# The lowest ceiling this build attains, found by gen_constants.py (893).
CFG = {"jstore": 9, "qcap": E.qcap_floor("j9")}

if __name__ == "__main__":
    E.staged_main(cfg=CFG, entry_file=__file__)
