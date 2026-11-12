"""fam_fig1.py -- Supplementary Table 2 family `fig1` (Fig. 1 and Section 4.1 of the paper): 21 tests.

  * 10 comparisons quoted in Section 4.1 (pooled auPRC, all 10,353 pairs): tie-aware AP, 10,000 component
    resamples, delta = unrounded point difference.
  * 3 comparisons quoted in the text: within-gene macro auROC against ABC over the 127 rankable genes, support
    `all` (what Fig. 1B and Section 4.1 read).
  * 8 Fig. 1A distance-bin brackets (each trunk against ABC in 0-10 kb, 10-100 kb, >100 kb; Enformer reaches no
    pair past 100 kb and has no bracket there), tie-aware AP: the bracket's delta is the difference of the 4-dp
    point estimates, as Fig. 1A prints it.

values(): Fig. 1A (four groups x six arms: auPRC, 95% CI, "reachable" share; brackets), Fig. 1B (macro within-gene
auROC, CI, reachable share), the within-gene counts of Section 4.1 and the q-value of every test (BH in the family).
"""
from __future__ import annotations

import numpy as np

import lib_a as L

#: (Supplementary Table 2 name, a, b), in the text's order
TEXT = [("Enformer vs distance", "Enformer", "1/distance"),
        ("Enformer vs ABC", "Enformer", "ABC"),
        ("Enformer vs ABC*, all pairs", "Enformer", "ABC*"),
        ("ABC vs ABC*", "ABC", "ABC*"),
        ("Borzoi vs ABC", "Borzoi", "ABC"),
        ("Borzoi vs ABC*, all pairs", "Borzoi", "ABC*"),
        ("Borzoi vs Enformer", "Borzoi", "Enformer"),
        ("AlphaGenome vs ABC", "AlphaGenome", "ABC"),
        ("AlphaGenome vs Enformer", "AlphaGenome", "Enformer"),
        ("AlphaGenome vs Borzoi", "AlphaGenome", "Borzoi")]
GROUPS = [g for g, _, _ in L.FIG1_GROUPS]
BRACKET_TRUNKS = ["AlphaGenome", "Borzoi", "Enformer"]


def _boot() -> dict:
    return L.boot_ap(GROUPS, L.ARMS)


def _has_bar(group: str, arm: str) -> bool:
    """An arm whose 1-dp reach of the group is 0 gets no bar and no bracket (ABC is defined where it scores
    nonzero, every other arm by its table's in_window flag; 1/distance everywhere)."""
    c = L.cohort()
    m = c.group(group)
    w = (c.score(arm) != 0) if arm == L.REF else c.arm(arm)[1]
    return round(100 * int(w[m].sum()) / int(m.sum()), 1) != 0.0


def _tests() -> list:
    B = _boot()
    out = []
    for name, a, b in TEXT:
        out.append(dict(id="text:" + name, name=name, **L.contrast(B, "all pairs", a, b)))
    W = L.within_gene("all")
    for t in L.TRUNKS:
        name = f"{t} vs ABC, within-gene auROC ({len(W['genes'])} genes)"
        out.append(dict(id="text:" + name, name=name, **L.wg_contrast(W, t)))
    for g in GROUPS[1:]:
        for t in BRACKET_TRUNKS:
            if not _has_bar(g, t):
                continue
            name = f"{t} vs ABC, {g}"
            out.append(dict(id="fig1:" + name, name=name, **L.contrast(B, g, t, L.REF, rounded_points=True)))
    return out


def compute() -> list:
    return [{k: t[k] for k in ("id", "delta", "lo", "hi", "p")} for t in _tests()]


def values() -> dict:
    """Labels: "Fig 1A, <arm>, <group>, auPRC" etc., as in paper/expected/values.json. Shares and prevalences are
    unrounded percentages; the figure prints them as reach_label / 2-dp strings."""
    c = L.cohort()
    B = _boot()
    V = {}
    tests = _tests()
    for i, (q, bound, _) in L.bh_q(tests).items():
        V[f"q, {i}"], V[f"q is an upper bound, {i}"] = q, float(bound)
    # Fig. 1A: four groups; every arm's bar, CI and "reachable" share; the brackets against ABC
    for g in GROUPS:
        m = c.group(g)
        n = int(m.sum())
        V[f"Fig 1A, {g}, pairs"], V[f"Fig 1A, {g}, positives"] = n, int(c.y[m].sum())
        V[f"Fig 1A, {g}, prevalence (%)"] = 100 * float(c.y[m].mean())
        V[f"Fig 1A, {g}, bootstrap replicates used"] = len(B[(g, L.REF)][1])
        for a in L.ARMS:
            cov = int(c.reached(a)[m].sum())
            V[f"Fig 1A, {a}, {g}, pairs reached"], V[f"Fig 1A, {a}, {g}, reachable (%)"] = cov, 100 * cov / n
            if not _has_bar(g, a):
                continue
            pt, reps = B[(g, a)]
            lo, hi = L.ci(reps)
            V[f"Fig 1A, {a}, {g}, auPRC"] = pt
            V[f"Fig 1A, {a}, {g}, auPRC 95% CI low"], V[f"Fig 1A, {a}, {g}, auPRC 95% CI high"] = lo, hi
        for t in BRACKET_TRUNKS:
            if _has_bar(g, t):
                r = L.contrast(B, g, t, L.REF, rounded_points=True)
                for k, nm in L.STAT_NAMES:
                    V[f"Fig 1A, {g}, {t} vs ABC, {nm}"] = r[k]
    # Fig. 1B: macro within-gene auROC over the rankable genes (support `all`); its reachable row counts the
    # rankable genes' pairs (Fig. 1B = the `all` group of the within-gene figure, Supplementary Fig. 8)
    W = L.within_gene("all")
    sel = np.zeros(c.n, bool)
    for ix in W["gidx"]:
        sel[ix] = True
    nsel = int(sel.sum())
    V["Fig 1B, rankable genes"], V["Fig 1B, pairs of rankable genes"] = len(W["genes"]), nsel
    V["Fig 1B, positives of rankable genes"] = int(c.y[sel].sum())
    for a in L.ARMS:
        lo, hi = L.ci(W["boot"][a])
        V[f"Fig 1B, {a}, macro within-gene auROC"] = W["macro"][a]
        V[f"Fig 1B, {a}, macro within-gene auROC 95% CI low"] = lo
        V[f"Fig 1B, {a}, macro within-gene auROC 95% CI high"] = hi
        cov = int(c.reached(a)[sel].sum())
        V[f"Fig 1B, {a}, pairs reached"], V[f"Fig 1B, {a}, reachable (%)"] = cov, 100 * cov / nsel
    # Section 4.1: the within-gene paragraph's counts (ties_all_six, discriminating, median)
    A = np.vstack([W["auc"][a] for a in L.ARMS])
    ties = int((A == A[0]).all(axis=0).sum())
    V["Section 4.1, within-gene, rankable genes"] = len(W["genes"])
    V["Section 4.1, within-gene, median tested elements per gene"] = float(np.median([len(ix) for ix in W["gidx"]]))
    V["Section 4.1, within-gene, genes on which all six methods score identically"] = ties
    V["Section 4.1, within-gene, discriminating genes"] = len(W["genes"]) - ties
    # Section 4.1's ranking sentence: the all-pairs point estimates, rank 1 = highest
    for i, a in enumerate(sorted(L.ARMS, key=lambda a: -B[("all pairs", a)][0]), 1):
        V[f"Section 4.1, all-pairs auPRC rank, {a}"] = i
    return V
