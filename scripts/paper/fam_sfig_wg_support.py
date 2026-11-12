"""fam_sfig_wg_support.py -- Supplementary Table 2 family `sfig-wg-support` (Supplementary Fig. 8, "Macro within-gene
ranking for all supports"): 9 tests.

Each trunk against ABC in macro within-gene auROC over the rankable genes (>= 1 positive and >= 1 negative inside the
support) of the three receptive-field supports (rf-AlphaGenome / rf-Borzoi / rf-Enformer: the pairs inside the
trunk's window, by the score table's in_window flag); the all-pairs brackets are comparisons of Fig. 1's family
(fig1). Per-gene tie-corrected auROC, macro = mean over genes, 10,000 component resamples (a component's rankable
genes enter once per draw), delta = difference of the macro auROCs, CI and p from the per-resample difference.

values(): every bar of the figure (four supports x six arms: macro auROC and 95% CI), each support's pairs, rankable
genes and their pairs, every bracket (incl. the all-pairs one), the "reachable" row (each arm's share of the rankable
genes' pairs) and the family's q-values.
"""
from __future__ import annotations

import numpy as np

import lib_a as L

SUPPORTS = [s for s, _ in L.WG_SUPPORTS]


def _tests() -> list:
    out = []
    for s in SUPPORTS[1:]:
        W = L.within_gene(s)
        for t in L.TRUNKS:
            name = f"{t} vs ABC, within-gene auROC, {s} genes"
            out.append(dict(id="sfig-wg-support:" + name, name=name, **L.wg_contrast(W, t)))
    return out


def compute() -> list:
    return [{k: t[k] for k in ("id", "delta", "lo", "hi", "p")} for t in _tests()]


FIG = "Supp Fig within-gene by support"


def values() -> dict:
    """Labels: "Supp Fig within-gene by support, <arm>, <support>, macro within-gene auROC", "..., <support>, <trunk>
    vs ABC, delta", "..., <support>, rankable genes", ... (as in paper/expected/values.json)."""
    c = L.cohort()
    V = {}
    for i, (q, bound, _) in L.bh_q(_tests()).items():
        V[f"q, {i}"], V[f"q is an upper bound, {i}"] = q, float(bound)
    for s in SUPPORTS:
        W = L.within_gene(s)
        m = W["mask"]
        sel = np.zeros(c.n, bool)
        for ix in W["gidx"]:
            sel[ix] = True
        nsel = int(sel.sum())
        S = L.SUPPORT_NAME[s]
        V[f"{FIG}, {S}, pairs"], V[f"{FIG}, {S}, positives"] = int(m.sum()), int(c.y[m].sum())
        V[f"{FIG}, {S}, rankable genes"], V[f"{FIG}, {S}, pairs of rankable genes"] = len(W["genes"]), nsel
        V[f"{FIG}, {S}, positives of rankable genes"] = int(c.y[sel].sum())
        for a in L.ARMS:
            lo, hi = L.ci(W["boot"][a])
            V[f"{FIG}, {a}, {S}, macro within-gene auROC"] = W["macro"][a]
            V[f"{FIG}, {a}, {S}, macro within-gene auROC 95% CI low"] = lo
            V[f"{FIG}, {a}, {S}, macro within-gene auROC 95% CI high"] = hi
            cov = int(c.reached(a)[sel].sum())
            V[f"{FIG}, {a}, {S}, pairs reached"], V[f"{FIG}, {a}, {S}, reachable (%)"] = cov, 100 * cov / nsel
        for t in L.TRUNKS:
            r = L.wg_contrast(W, t)
            for k, nm in L.STAT_NAMES:
                V[f"{FIG}, {S}, {t} vs ABC, {nm}"] = r[k]
        # the rankable genes on which all six arms score the same auROC, and the median elements per gene
        # (ties_all_six, discriminating, median_elements, spread)
        A = np.vstack([W["auc"][a] for a in L.ARMS])
        ties = int((A == A[0]).all(axis=0).sum())
        V[f"{FIG}, {S}, genes on which all six methods score identically"] = ties
        V[f"{FIG}, {S}, discriminating genes"] = len(W["genes"]) - ties
        V[f"{FIG}, {S}, median tested elements per gene"] = float(np.median([len(ix) for ix in W["gidx"]]))
        V[f"{FIG}, {S}, spread of the six macro auROCs"] = max(W["macro"].values()) - min(W["macro"].values())
    return V
