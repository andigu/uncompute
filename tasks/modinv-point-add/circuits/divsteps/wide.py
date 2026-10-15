#!/usr/bin/env python3
"""divstep circuit variant: subtraction erase instead of measurement
fixup pairs, fresh host words, carries filled up to the ceiling.
Cost: 3,293,278 Toffoli x 1,191 qubits.
"""

import pathlib

from uncompute.testing import load_module

HERE = pathlib.Path(__file__).resolve().parent

E = load_module(HERE / "engine.py", "divstep_engine")
# The lowest ceiling this build attains, found by gen_constants.py (1,191).
CFG = {"qcap": E.qcap_floor("wide"), "wide": True}

if __name__ == "__main__":
    E.hmr_main(cfg=CFG, entry_file=__file__)
