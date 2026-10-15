#!/usr/bin/env python3
"""divstep circuit variant: the decode ladder at store threshold
v2(fv') >= 6 (algebraic branch reconstruction through
j = 5). Cost: 4,427,338 Toffoli x 913 qubits.
"""

import pathlib

from uncompute.testing import load_module

HERE = pathlib.Path(__file__).resolve().parent

E = load_module(HERE / "engine.py", "divstep_engine")
# The lowest ceiling this build attains, found by gen_constants.py (913).
CFG = {"jstore": 6, "qcap": E.qcap_floor("j6")}

if __name__ == "__main__":
    E.staged_main(cfg=CFG, entry_file=__file__)
