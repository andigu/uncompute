"""Shared circuit gadgets for circuit authoring.

Small reversible subroutines from the literature that several circuits use,
with one shared copy each, plus the wire plumbing (``wires``) that prepares a
circuit's wires for them.

    from uncompute.gadgets import walk

Design rules:

* Every gadget takes the live :class:`uncompute.Builder` as its first argument
  and emits ops directly — no gate lists, no deferred IR.
* Each gadget documents its source (arXiv id / figure) and its exact
  executed-Toffoli cost.
* Gadgets are library code like ``builder.py``: an edit to one gadget
  changes the emitted stream of every circuit that uses it.

Classical table-building and reference models live in ``uncompute.classical``;
Bennett-reversal choreography (Journal, Stages) lives in ``uncompute.bennett``.
"""

from .arith import (
    add,
    and_chain,
    and_ladder,
    and_ladder_release,
    booth_columns,
    booth_columns_release,
    cdkm_add_co,
    eq_const,
    ge_const_flag,
    ge_const_release,
    gidney_add,
    gidney_sub,
    hrs_mul_rows,
    label_increment,
    maj,
    maj_inv,
    mcx,
    release_chain,
    uma,
    uma_inv,
)
from .gf_tower import GF
from .linear import linear_in_place, linear_xor
from .qrom import (
    berry_ku,
    erase_register,
    phase_fixup,
    pruned_walk,
    selectswap_lookup,
    swap_perm,
    walk,
    xor_outcomes,
    xor_word,
)
from .wires import WirePool, declare_registers

__all__ = [
    "GF",
    "WirePool",
    "add",
    "and_chain",
    "and_ladder",
    "and_ladder_release",
    "berry_ku",
    "booth_columns",
    "booth_columns_release",
    "cdkm_add_co",
    "declare_registers",
    "eq_const",
    "erase_register",
    "ge_const_flag",
    "ge_const_release",
    "gidney_add",
    "gidney_sub",
    "hrs_mul_rows",
    "label_increment",
    "linear_in_place",
    "linear_xor",
    "maj",
    "maj_inv",
    "mcx",
    "phase_fixup",
    "pruned_walk",
    "release_chain",
    "selectswap_lookup",
    "swap_perm",
    "uma",
    "uma_inv",
    "walk",
    "xor_outcomes",
    "xor_word",
]
