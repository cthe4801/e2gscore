"""fam_fig3.py -- Supplementary Table 2 family "fig3" (Fig. 3 and Section 4.3): 16 tests.

    python3 scripts/reproduce_paper.py fig3

Fig. 3 is the anchor x readout matrix of input gradients (Section 4.3). Every panel is scored on the pairs that
EVERY arm of that panel reaches (its common support: Enformer 1,635 pairs, Borzoi 3,896, AlphaGenome 6,617), with
the paper's estimator (lib_b: tie-aware AP, 10,000 component draws, seed 20260816, shared by every move of a
panel). The 16 tests:
  * 5 moves only the figure draws (Borzoi and AlphaGenome readout moves, AlphaGenome's gene-midpoint CAGE -> PRO-cap);
  * 9 comparisons quoted in Section 4.3 that are moves of the same figure: the six anchor moves, quoted single-TSS
    minus gene-midpoint (the figure draws gene-midpoint minus single-TSS, so the quoted comparison is the move
    negated), the two corner-to-corner totals and AlphaGenome's single-TSS CAGE -> PRO-cap;
  * AlphaGenome RNA-seq protocol (A6) vs PRO-cap protocol (A8) on the AlphaGenome panel's support, the pooled row
    of the screens figure (Supplementary Fig. 16; same draws, same support as the panel);
  * Borzoi under the RNA-seq protocol (F1R) vs ABC (ours) on all 10,353 pairs (the matched-protocol comparison of
    Section 4.1).

Test names follow Supplementary Table 2.
"""
from __future__ import annotations

import numpy as np

import lib_b as L

#: Section 4.3's tuples that are drawn moves: (model, kind, src, dst) -> (text tuple, sign of the text vs the move)
TEXT_MOVES = {
    ("Enformer", "anchor", (0, 0), (1, 0)): ("Enformer CAGE: single-TSS vs gene-midpoint anchor", -1),
    ("Borzoi", "anchor", (0, 0), (1, 0)): ("Borzoi CAGE: single-TSS vs gene-midpoint anchor", -1),
    ("Borzoi", "anchor", (0, 1), (1, 1)): ("Borzoi RNA-seq: single-TSS vs gene-midpoint anchor", -1),
    ("AlphaGenome", "anchor", (0, 0), (1, 0)): ("AlphaGenome CAGE: single-TSS vs gene-midpoint anchor", -1),
    ("AlphaGenome", "anchor", (0, 1), (1, 1)): ("AlphaGenome RNA-seq: single-TSS vs gene-midpoint anchor", -1),
    ("AlphaGenome", "anchor", (0, 2), (1, 2)): ("AlphaGenome PRO-cap: single-TSS vs gene-midpoint anchor", -1),
    ("AlphaGenome", "readout-alt", (0, 0), (0, 2)): ("AlphaGenome: PRO-cap@TSS vs CAGE@TSS", 1),
}
TOTALS = {"Borzoi": "Borzoi: RNA-seq@exons/gene-mid vs CAGE@TSS/single-TSS",
          "AlphaGenome": "AlphaGenome: RNA-seq@exons/gene-mid vs CAGE@TSS/single-TSS"}
_RUN: dict = {}


def _run() -> dict:
    if not _RUN:
        coh = L.cohort("a1w")
        f1r, _ = coh.arm("F1R-borzoi-gene-midpoint-rnaseq-tssnull")
        a2wr, _ = coh.arm("A2wR-borzoi-single-tss")
        abc, _ = coh.column(L.ABC)
        # A2wR (the matched protocol) only for values(): Section 4.3 re-quotes Section 4.1's Borzoi vs ABC
        extra = {"all pairs": (np.ones(coh.n, bool), {"F1R": f1r, "A2wR": a2wr, "ABC": abc})}
        panels, B = L.fig3_panels(all_pairs=False, extra=extra)
        _RUN.update(panels=panels, B=B)
    return _RUN


def compute() -> list:
    R = _run()
    panels, B = R["panels"], R["B"]
    out = []
    for p in panels:
        for mv in p["moves"]:
            k = (p["model"], mv["kind"], tuple(mv["src"]), tuple(mv["dst"]))
            if k in TEXT_MOVES:
                name, sign = TEXT_MOVES[k]
                out.append(L.row("text:" + name, mv if sign > 0 else L.flip(mv)))
            elif p["model"] != "Enformer" and mv["kind"] != "anchor":
                # the figure-only moves; every anchor move is a text tuple (above)
                out.append(L.row("fig3:" + L.fig3_move_name(p["model"], mv), mv))
        if p["model"] in TOTALS:
            out.append(L.row("text:" + TOTALS[p["model"]], p["total"]))
    # AlphaGenome RNA-seq protocol (A6, cell 1,1) vs PRO-cap protocol (A8, cell 0,2), on the AlphaGenome panel
    out.append(L.row("text:AlphaGenome: RNA-seq@exons/gene-mid vs PRO-cap@TSS/single-TSS",
                     L.contrast(B, "AlphaGenome", (1, 1), (0, 2))))
    out.append(L.row("text:Borzoi under the RNA-seq protocol vs ABC", L.contrast(B, "all pairs", "F1R", "ABC")))
    return out


def values() -> dict:
    """Fig. 3 as printed: per panel n / positives, the cell auPRCs, every move and the corner-to-corner total; the
    all-pairs auPRCs behind 'Borzoi under the RNA-seq protocol vs ABC'; Section 4.1's Borzoi vs ABC, which 4.3
    re-quotes (a test of Fig. 1's family, given here as a value); and the q every test of this family prints."""
    R = _run()
    v = L.fig3_values(R["panels"], "fig3")
    v["sec4.3/Borzoi RNA-seq protocol (F1R) auPRC, all pairs"] = R["B"]["all pairs"]["pt"]["F1R"]
    v["sec4.3/ABC auPRC, all pairs"] = R["B"]["all pairs"]["pt"]["ABC"]
    t = L.contrast(R["B"], "all pairs", "A2wR", "ABC")
    for f in ("delta", "lo", "hi", "p"):
        v[f"sec4.3/Borzoi vs ABC, all pairs (Section 4.1's tuple)/{f}"] = t[f]
    v.update(L.q_values(compute()))
    return v
