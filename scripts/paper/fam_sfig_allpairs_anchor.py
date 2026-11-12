"""fam_sfig_allpairs_anchor.py -- Supplementary Table 2 family "sfig-allpairs-anchor": 12 tests.

    python3 scripts/reproduce_paper.py sfig-allpairs-anchor

The all-pairs twin of Fig. 3 (Supplementary Fig. 10, "sequence anchor and gene readout, all pairs"): the same
panels, arms, moves and draws as Fig. 3 (lib_b.fig3_panels), but every panel scored on all 10,353 pairs, the pairs
outside an arm's window carrying the table's imputed 0.0.
The 12 tests are the 12 drawn moves (Enformer 1, Borzoi 4, AlphaGenome 7), each as drawn (anchor moves are
gene-midpoint minus single-TSS). values() also returns each move's matched-support delta the figure prints under it.

Every AP is tie-aware; test names follow Supplementary Table 2.

KNOWN DISAGREEMENT: Supplementary Table 2 prints the Enformer row as -0.1069 [-0.1524, -0.0663]; the shipped
tables give -0.1071 [-0.1528, -0.0667], the value Supplementary Fig. 10 prints.
"""
from __future__ import annotations

import lib_b as L

_RUN: dict = {}


def _run() -> dict:
    if not _RUN:
        panels, B = L.fig3_panels(all_pairs=True)
        _RUN.update(panels=panels, B=B)
    return _RUN


def compute() -> list:
    out = []
    for p in _run()["panels"]:
        for mv in p["moves"]:
            out.append(L.row("sfig-allpairs-anchor:" + L.fig3_move_name(p["model"], mv) + ", all pairs", mv))
    return out


def values() -> dict:
    """The all-pairs figure as printed: per panel n / positives, cells, moves (and each move's matched-support
    delta), corner-to-corner totals, the q of each test; and Section 4.3's four differences of this figure's cells
    (published protocol = Borzoi RNA-seq@exons / gene-midpoint and AlphaGenome
    likewise, Enformer CAGE@TSS / single-TSS; matched = CAGE@TSS / single-TSS for all three), with the two ratios the
    sentence reads off them ("half of the gap", "nearly doubles")."""
    P = {p["model"]: p["cells"] for p in _run()["panels"]}
    v = L.fig3_values(_run()["panels"], "sfig-allpairs-anchor")
    v.update(L.q_values(compute()))
    pub = P["Borzoi"][(1, 1)] - P["Enformer"][(0, 0)]
    mat = P["Borzoi"][(0, 0)] - P["Enformer"][(0, 0)]
    ag_mat = P["AlphaGenome"][(0, 0)] - P["Borzoi"][(0, 0)]
    ag_pub = P["AlphaGenome"][(1, 1)] - P["Borzoi"][(1, 1)]
    v.update({"sec4.3/Borzoi minus Enformer, published protocols, all pairs": pub,
              "sec4.3/Borzoi minus Enformer, matched protocol, all pairs": mat,
              "sec4.3/AlphaGenome minus Borzoi, matched protocol, all pairs": ag_mat,
              "sec4.3/AlphaGenome minus Borzoi, published protocols, all pairs": ag_pub,
              "sec4.3/matched share of the published Borzoi-Enformer gap": mat / pub,
              "sec4.3/matched over published AlphaGenome-Borzoi lead": ag_mat / ag_pub})
    return v
