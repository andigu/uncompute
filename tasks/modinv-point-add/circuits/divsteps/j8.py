#!/usr/bin/env python3
"""divstep circuit variant: the decode ladder at store threshold
v2(fv') >= 8 (algebraic branch reconstruction through
j = 7). Cost: 5,331,729 Toffoli x 897 qubits.
"""

import pathlib

from uncompute.testing import load_module

HERE = pathlib.Path(__file__).resolve().parent

E = load_module(HERE / "engine.py", "divstep_engine")
# The lowest ceiling this build attains, found by gen_constants.py (897).
CFG = {"jstore": 8, "qcap": E.qcap_floor("j8")}

if __name__ == "__main__":
    E.staged_main(cfg=CFG, entry_file=__file__)
