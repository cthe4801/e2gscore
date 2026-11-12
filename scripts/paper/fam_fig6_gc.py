"""fam_fig6_gc.py -- Fig. 6, gradient correction block (Section 4.6).

Family `fig6-gc`: 36 range cells, 12 per model. Each cell compares two reductions of ONE run's free grid, mean
subtraction over the 4-channel gradient map on vs off, at one Gaussian pooling kernel g{sigma}t{truncate}:
    score__g{s}t{t}-global-ms1{suf} - score__g{s}t{t}-global-ms0{suf},  s in 500..3000, t in 1..3
on that run's own in-window pairs: Enformer A1w (1,796 pairs), Borzoi F1R gene-midpoint RNA-seq with the
`-rafter` columns (3,973), AlphaGenome A6 gene-midpoint RNA-seq (6,677). Fig. 6 draws each row as the min-max of its
12 deltas and "k/12 sig", k = the cells that survive Benjamini-Hochberg within this family.
Estimator: lib_c.range_tests (lib_c.subset_eval on explicit pairs, tie-aware AP); bootstrap as in fam_fig6_anchor.
"""
from __future__ import annotations

import lib_c as L

FAMILY = "fig6-gc"


def compute() -> list:
    return [{k: t[k] for k in ("id", "delta", "lo", "hi", "p")} for t in L.range_tests(FAMILY, ["gc"])]


def values() -> dict:
    return L.range_values(L.range_tests(FAMILY, ["gc"]), FAMILY)
