#!/usr/bin/env python3
"""divstep circuit variant: the decode ladder at store threshold
v2(fv') >= 7 (algebraic branch reconstruction through
j = 6). Cost: 4,649,514 Toffoli x 904 qubits.
"""

import pathlib

from uncompute.testing import load_module

HERE = pathlib.Path(__file__).resolve().parent

E = load_module(HERE / "engine.py", "divstep_engine")
# Hand-set ceiling, one wire above this build's floor (903).  At 903
# the build costs 4,655,041 Toffoli, a higher TQ-cost than at 904.
CFG = {"jstore": 7, "qcap": 904}

if __name__ == "__main__":
    E.staged_main(cfg=CFG, entry_file=__file__)
