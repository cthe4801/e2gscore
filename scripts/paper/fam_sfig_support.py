"""fam_sfig_support.py -- Supplementary Table 2 family `sfig-support` (Supplementary Fig. 7, "Cross-gene ranking for
all supports"): 9 tests.

Each trunk against ABC in pooled tie-aware auPRC on the three receptive-field supports (rf-alphagenome / rf-borzoi /
rf-enformer: the pairs with |distance_to_tss| below 524,288 / 262,144 / 98,304 bp); the all-pairs brackets of the
figure are comparisons of Fig. 1's family (fig1). Tie-aware AP; shared component resamples restricted to each
support, delta = the unrounded point difference.

values(): every bar of the figure (four supports x six arms: auPRC and 95% CI), each support's n / positives /
prevalence, every bracket (incl. the all-pairs one), the "reachable" share row and the family's q-values.
"""
from __future__ import annotations

import lib_a as L

SUPPORTS = [s for s, _ in L.SUPPORTS]
BRACKET_TRUNKS = ["AlphaGenome", "Borzoi", "Enformer"]


def _boot() -> dict:
    return L.boot_ap(SUPPORTS, L.ARMS)


def _tests() -> list:
    B = _boot()
    out = []
    for s in SUPPORTS[1:]:
        for t in BRACKET_TRUNKS:
            name = f"{t} vs ABC, {s} pairs"
            out.append(dict(id="sfig-support:" + name, name=name, **L.contrast(B, s, t, L.REF)))
    return out


def compute() -> list:
    return [{k: t[k] for k in ("id", "delta", "lo", "hi", "p")} for t in _tests()]


FIG = "Supp Fig auPRC by support"


def values() -> dict:
    """Labels: "Supp Fig auPRC by support, <arm>, <support>, auPRC", "..., <support>, <trunk> vs ABC, delta", ...
    (as in paper/expected/values.json). Supports as the figure prints them: all pairs, rf-AlphaGenome, rf-Borzoi,
    rf-Enformer."""
    c = L.cohort()
    B = _boot()
    V = {}
    for i, (q, bound, _) in L.bh_q(_tests()).items():
        V[f"q, {i}"], V[f"q is an upper bound, {i}"] = q, float(bound)
    for s in SUPPORTS:
        m = c.group(s)
        n = int(m.sum())
        S = L.SUPPORT_NAME[s]
        V[f"{FIG}, {S}, pairs"], V[f"{FIG}, {S}, positives"] = n, int(c.y[m].sum())
        V[f"{FIG}, {S}, prevalence (%)"] = 100 * float(c.y[m].mean())
        V[f"{FIG}, {S}, bootstrap replicates used"] = len(B[(s, L.REF)][1])
        for a in L.ARMS:
            pt, reps = B[(s, a)]
            lo, hi = L.ci(reps)
            V[f"{FIG}, {a}, {S}, auPRC"] = pt
            V[f"{FIG}, {a}, {S}, auPRC 95% CI low"], V[f"{FIG}, {a}, {S}, auPRC 95% CI high"] = lo, hi
            cov = int(c.reached(a)[m].sum())
            V[f"{FIG}, {a}, {S}, pairs reached"], V[f"{FIG}, {a}, {S}, reachable (%)"] = cov, 100 * cov / n
        for t in BRACKET_TRUNKS:
            r = L.contrast(B, s, t, L.REF)
            for k, nm in L.STAT_NAMES:
                V[f"{FIG}, {S}, {t} vs ABC, {nm}"] = r[k]
        # Sections 4.1 / 4.2: "this ranking is conserved under all supports" -- the order of the six point estimates
        for i, a in enumerate(sorted(L.ARMS, key=lambda a: -B[(s, a)][0]), 1):
            V[f"{FIG}, {S}, auPRC rank, {a}"] = i
    return V
