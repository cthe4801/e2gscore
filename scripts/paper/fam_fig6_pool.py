"""fam_fig6_pool.py -- Fig. 6, pooling geometry block (Section 4.6).

Family `fig6-pool`: 48 range cells, 16 per model in its two pooling rows, all with mean subtraction on (ms1):
    kernel sweep (12 cells)         score__g{s}t{t}-global-ms1{suf} - score   (`score` = the protocol kernel,
                                    sigma 300 bp, truncate 4)
    Gaussian vs average (4 cells)   score__g{W}t2-global-ms1{suf} - score__s{W}-global-ms1{suf}, W = 500..3000
on each run's own in-window pairs (Enformer A1w, Borzoi F1R `-rafter`, AlphaGenome A6). Fig. 6 draws each row as
the min-max of its deltas and "k/n sig", k = the cells that survive Benjamini-Hochberg within this family.
Estimator: lib_c.range_tests (lib_c.subset_eval on explicit pairs, tie-aware AP); bootstrap as in fam_fig6_anchor.
"""
from __future__ import annotations

import lib_c as L

FAMILY = "fig6-pool"


def compute() -> list:
    return [{k: t[k] for k in ("id", "delta", "lo", "hi", "p")} for t in L.range_tests(FAMILY, ["ks", "gv"])]


def values() -> dict:
    return L.range_values(L.range_tests(FAMILY, ["ks", "gv"]), FAMILY)
