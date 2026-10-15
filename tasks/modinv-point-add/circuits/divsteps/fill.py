#!/usr/bin/env python3
"""divstep circuit variant: walk adders fill their carries up to a
1,031-qubit ceiling, and the multiplier uses one block.
Cost: 3,637,655 Toffoli x 1,031 qubits.
"""

import pathlib

from uncompute.testing import load_module

HERE = pathlib.Path(__file__).resolve().parent

E = load_module(HERE / "engine.py", "divstep_engine")
# The ceiling is the smallest one at which the off-plateau multiplier
# pool holds a full single block of n wires (1,031).
CFG = {"qcap": E.MUL_BASE + E.N, "wide": False}

if __name__ == "__main__":
    E.hmr_main(cfg=CFG, entry_file=__file__)
