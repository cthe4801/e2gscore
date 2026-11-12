"""fam_fig6_anchor.py -- Fig. 6, sequence anchor block (Section 4.6).

Family `fig6-anchor`: four Enformer rows, each `arm - A1w` on the pairs both runs reach (common-reachable):
    O3  major TSS re-picked by the model (GENCODE v50)    in the text (Section 4.6) and a figure row
    B1  input gradients averaged over every annotated TSS  in the text and a figure row
    J1  readout at every in-window TSS                     figure row
    R2  enhancer-TSS midpoint anchor, readout pinned       figure row
Estimator: lib_c.fig6_pair -> lib_c.subset_eval, the computation of scripts/evaluate.py: tie-aware
AP; blocks = the evaluator's 663 components (rows in sorted key order, blocks ordered by their union-find root);
10,000 replicates from default_rng(20260816), one rng.integers(0, 663, 663) each; a replicate is skipped when it is
empty or all one label; percentile CI; two-sided p. See lib_c.subset_eval.

Not recomputable from scores: the red cost labels (16x / 15.7x / 2.8x ..., GPU-hours) come from run logs and
manifests (the manifests' gpu_calls), not from score tables.
"""
from __future__ import annotations

import lib_c as L

FAMILY = "fig6-anchor"
C = L.CTRL
ROWS = [("text:Enformer: major-TSS re-pick (GENCODE v50) vs annotated TSS", (C, L.FIG6_POINT["O3"][1]), True),
        ("text:Enformer: anchor averaged over every annotated TSS vs the RefSeq TSS", (C, L.FIG6_POINT["B1"][1]), True),
        ("fig6-anchor:Enformer: readout at every in-window TSS", (C, L.FIG6_POINT["J1"][1]), False),
        ("fig6-anchor:Enformer: enh-TSS midpoint, readout pinned", (C, L.FIG6_POINT["R2"][1]), False)]


def compute() -> list:
    return [{k: t[k] for k in ("id", "delta", "lo", "hi", "p")} for t in L.point_tests(ROWS)]


def values() -> dict:
    return L.point_values(L.point_tests(ROWS))
