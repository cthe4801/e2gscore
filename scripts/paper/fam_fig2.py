"""fam_fig2.py -- Supplementary Table 2 family `fig2` (Fig. 2 and Section 4.2 of the paper): 1 test.

  * "ABC* vs prevalence, on Enformer's out-of-window pairs": ABC*'s tie-aware AP minus the prevalence on the 8,557
    pairs outside Enformer's window, formed within each of the 10,000 component resamples.

values(): Section 4.2's counts (8,557 pairs, 82 positives, Enformer at prevalence 0.0096, 83% / 17%), Fig. 2A's
support and gene counts, Fig. 2B (tie-aware auPRC of each method on each support, and each support's prevalence) and
Fig. 2C (the closed form auPRC_all = w auPRC_reachable + (1 - w) P/N, and the penalty), with the closed form's
consistency checks.
"""
from __future__ import annotations

import numpy as np

import lib_a as L

OUTSIDE = "outside Enformer's window"
NAME = "ABC* vs prevalence, on Enformer's out-of-window pairs"
#: Fig. 2B's methods (in the figure's order) and supports; a support is the pairs inside a trunk's window
ORDER = ["AlphaGenome", "ABC", "Borzoi", "Enformer", "1/distance"]
SUPPORT_OF = [("all", None), ("rf-alphagenome", "AlphaGenome"), ("rf-borzoi", "Borzoi"),
              ("rf-enformer", "Enformer")]


def _excess() -> dict:
    """delta = AP(ABC*) - prevalence on the pairs; per resample the same."""
    B = L.boot_ap([OUTSIDE], ["ABC*", L.PREVALENCE])
    (ap_pt, ap_r), (pi_pt, pi_r) = B[(OUTSIDE, "ABC*")], B[(OUTSIDE, L.PREVALENCE)]
    dv = ap_r - pi_r
    lo, hi = L.ci(dv)
    return {"delta": ap_pt - pi_pt, "lo": lo, "hi": hi, "p": L.pval(dv), "reps": len(dv),
            "abcstar_ap": ap_pt, "prevalence": pi_pt}


def compute() -> list:
    r = _excess()
    return [{"id": "text:" + NAME, "delta": r["delta"], "lo": r["lo"], "hi": r["hi"], "p": r["p"]}]


def _fig2_tables():
    """Fig. 2's reading of the tables: every table covers the cohort once; a trunk scores > 0 inside its window and
    exactly 0 outside; a baseline reaches the pairs it scores > 0; 1/distance is the distance baseline's table."""
    c = L.cohort()
    lab = L.labels()
    assert len(c.key) == len(lab) and set(c.key) == set(lab), "Enformer table does not cover the cohort once"
    pos = {k: i for i, k in enumerate(c.key)}
    score, reach = {}, {}
    for m in ORDER:
        rows = L.read_rows(L.DISTANCE_TABLE if m == "1/distance" else L.TABLE[m])
        tk = [(r["gene"], r["enhancer"]) for r in rows]
        assert len(tk) == c.n and set(tk) == set(c.key), f"{m}: table does not cover the cohort once"
        idx = np.array([pos[k] for k in tk])
        sc = np.empty(c.n)
        sc[idx] = [float(r["score"]) for r in rows]
        assert (sc >= 0).all(), f"{m}: negative scores break the zero-imputation identity"
        if m in L.TRUNKS:
            iw = np.empty(c.n, bool)
            iw[idx] = [r["in_window"] == "True" for r in rows]
            assert (sc[iw] > 0).all() and (sc[~iw] == 0).all(), f"{m}: window and zeros disagree"
            reach[m] = iw
        else:
            reach[m] = sc > 0
        score[m] = sc
    return c, score, reach


#: Fig. 2C's quantities and their values() names
C_NAMES = [("pairs_reached", "pairs reached"), ("positives_reached", "positives reached"), ("w", "w"),
           ("auprc_reachable", "auPRC on reachable pairs"), ("auprc_all", "auPRC on all pairs"),
           ("closed_form", "closed form"), ("penalty", "penalty")]


def values() -> dict:
    """Labels: "Section 4.2, outside Enformer's window, pairs", "Fig 2B, <method>, <support>, auPRC",
    "Fig 2C, <method>, penalty", "Fig 2A, <support>, genes", ... (as in paper/expected/values.json)."""
    V = {}
    r = _excess()
    c = L.cohort()
    out = c.group(OUTSIDE)
    enf = c.score("Enformer")
    # Section 4.2, first paragraph
    S = "Section 4.2, outside Enformer's window"
    V[f"{S}, pairs"], V[f"{S}, positives"] = int(out.sum()), int(c.y[out].sum())
    V[f"{S}, prevalence"] = r["prevalence"]
    V[f"{S}, Enformer auPRC"] = L.ap_tie(enf[out], c.y[out])
    V[f"{S}, ABC* auPRC"] = r["abcstar_ap"]
    V[f"{S}, share of the cohort's pairs (%)"] = 100 * int(out.sum()) / c.n
    V[f"{S}, share of the cohort's positives (%)"] = 100 * int(c.y[out].sum()) / int(c.y.sum())
    V[f"{S}, bootstrap replicates used"] = r["reps"]
    V["Section 4.2, cohort, pairs"], V["Section 4.2, cohort, positives"] = c.n, int(c.y.sum())
    for i, (q, bound, _) in L.bh_q([{"id": "text:" + NAME, "name": NAME, "p": r["p"]}]).items():
        V[f"q, {i}"], V[f"q is an upper bound, {i}"] = q, float(bound)
    # Fig. 2B and 2C
    c, score, reach = _fig2_tables()
    y, n, p = c.y, c.n, int(c.y.sum())
    sup = {k: (np.ones(n, bool) if t is None else reach[t]) for k, t in SUPPORT_OF}
    assert not (sup["rf-enformer"] & ~sup["rf-borzoi"]).any() and not (sup["rf-borzoi"] & ~sup["rf-alphagenome"]).any()
    for k, _ in SUPPORT_OF:
        s = L.SUPPORT_NAME[k]
        V[f"Fig 2B, {s}, pairs"], V[f"Fig 2B, {s}, positives"] = int(sup[k].sum()), int(y[sup[k]].sum())
        V[f"Fig 2B, {s}, prevalence"] = float(y[sup[k]].mean())
        for m in ORDER:
            V[f"Fig 2B, {m}, {s}, auPRC"] = L.ap_tie(score[m][sup[k]], y[sup[k]])
    pi = p / n
    for m in ORDER:
        rm = reach[m]
        pr = int(y[rm].sum())
        w = pr / p
        ap_r, ap_all = L.ap_tie(score[m][rm], y[rm]), L.ap_tie(score[m], y)
        closed = w * ap_r + (1 - w) * pi
        assert abs(ap_all - closed) < 1e-12 and abs((ap_r - ap_all) - (1 - w) * (ap_r - pi)) < 1e-12, m
        got = dict(pairs_reached=int(rm.sum()), positives_reached=pr, w=w, auprc_reachable=ap_r, auprc_all=ap_all,
                   closed_form=closed, penalty=ap_r - ap_all)
        for k, nm in C_NAMES:
            V[f"Fig 2C, {m}, {nm}"] = got[k]
    # Fig. 2A: each support's genes and rankable genes (>= 1 positive and >= 1 negative there), and the pairs no
    # model reaches
    g = c.g
    for k, _ in SUPPORT_OF:
        m = sup[k]
        genes = np.unique(g[m])
        rk = [gg for gg in genes if y[m & (g == gg)].sum() >= 1 and (y[m & (g == gg)] == 0).sum() >= 1]
        V[f"Fig 2A, {L.SUPPORT_NAME[k]}, genes"], V[f"Fig 2A, {L.SUPPORT_NAME[k]}, rankable genes"] = len(genes), len(rk)
    none = ~(reach["Enformer"] | reach["Borzoi"] | reach["AlphaGenome"])
    V["Fig 2A, no model reaches, pairs"], V["Fig 2A, no model reaches, positives"] = int(none.sum()), int(y[none].sum())
    return V
