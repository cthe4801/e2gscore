"""fam_fig5.py -- Fig. 5 (Section 4.5): cell-type specific vs cell-type agnostic tracks.

Family `fig5`: the nine brackets of Fig. 5, three per model on the model's own receptive field (rf-enformer,
rf-borzoi, rf-alphagenome):
    K562 tracks vs all-cell-type tracks          K562 arm - all-cell-type arm         (drawn as -confounded)
    K562 tracks vs random non-K562 tracks         K562 arm - mean of the ten draws     (specificity)
    random non-K562 tracks vs all-cell-type tracks mean of the ten draws - all arm     (drawn as -count)
Arms: Enformer A1w / U0-U9 / N1, Borzoi A2wR / UB0R-UB9R / NB1R, AlphaGenome A3 / UA0-UA9 / NA1.

Estimator (lib_c.fig15_panel, tie-aware AP): pairs keyed in the K562 table's row order; blocks = connected components of
the gene-enhancer graph (663), numbered at first appearance; support |distance_to_tss| < the trunk's half window;
10,000 replicates from default_rng(20260816), one rng.integers(0, 663, 663) per replicate, re-seeded per support;
replicates without a positive dropped; percentile CI; p = min(1, 2 min(P(d <= 0), P(d >= 0))). The paper prints
every number to 4 dp (as Supplementary Table 2 holds them); compute() returns them unrounded.
"""
from __future__ import annotations

import lib_c as L

FAMILY = "fig5"


def compute() -> list:
    return [{k: t[k] for k in ("id", "delta", "lo", "hi", "p")} for t in L.fig5_tests(FAMILY, rf=True)]


def values() -> dict:
    """What Fig. 5 prints (and draws), its q-values, and the Section 4.5 numbers."""
    v = L.fig5_values(rf=True, prefix="Fig. 5")
    v.update(L.q_values(L.fig5_tests(FAMILY, rf=True), prefix="Fig. 5 q"))
    for t in L.TRUNKS:
        sup = f"rf-{t}"
        P = L.fig15_panel(t, sup)
        k, n = P["k562"][0], P["N1"][0]
        k4, n4 = round(k, 4), round(n, 4)
        nm = L.TRUNK_NAME[t]
        # "the relative auPRC changes ... were 16.7%, 20.6%, and 42.5%": (K562 - all) / all from the 4-dp auPRCs the
        # figure prints
        v[f"Sec 4.5 relative auPRC change {nm} (printed, %)"] = round(100 * (k4 - n4) / n4, 1)
        v[f"Sec 4.5 relative auPRC change {nm} (full precision)"] = (k - n) / n
        # "sit only 0.0202, 0.0557, and 0.0952 above distance": all-cell-types arm - distance, own support
        v[f"Sec 4.5 all-cell-types above distance {nm}"] = n - L.distance_floor(sup)
        # "sit 0.2053, 0.1580, and 0.1115 below ABC": ABC (ours) - all-cell-types arm, own support
        v[f"Sec 4.5 ABC above all-cell-types {nm}"] = L.abc_on_support(t, sup) - n
        v[f"ABC (ours) auPRC ({sup})"] = L.abc_on_support(t, sup)
    return v
