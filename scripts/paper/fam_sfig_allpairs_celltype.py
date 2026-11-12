"""fam_sfig_allpairs_celltype.py -- Supplementary Fig. 12, cell-type specific tracks on all pairs (Fig. 5's twin).

Family `sfig-allpairs-celltype`: Fig. 5's nine brackets on ALL 10,353 pairs instead of each model's receptive field.
Same arms, blocks, draws and estimator as fam_fig5 (lib_c.fig15_panel with support `all pairs`, tie-aware AP).
Out-of-window pairs score 0 in every arm, so each bracket is the rf bracket scaled by the share of positives in the
window, and its p equals the rf bracket's.
"""
from __future__ import annotations

import lib_c as L

FAMILY = "sfig-allpairs-celltype"


def compute() -> list:
    return [{k: t[k] for k in ("id", "delta", "lo", "hi", "p")} for t in L.fig5_tests(FAMILY, rf=False)]


def values() -> dict:
    v = L.fig5_values(rf=False, prefix="Supp. Fig. all-pairs cell type")
    v.update(L.q_values(L.fig5_tests(FAMILY, rf=False), prefix="Supp. Fig. all-pairs cell type q"))
    return v
