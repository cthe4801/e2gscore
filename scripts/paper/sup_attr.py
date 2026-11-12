"""sup_attr.py -- Supplementary Fig. 4 of the paper: overall (all-pairs) auPRC by attribution
method, per model and cohort, under each published protocol.

    values() -> {label: float}

Sixteen bars: Enformer (Avsec et al. 2021) on Fulco and Gasperini x {attention, gradient x input K562 / agnostic,
ISM K562 / agnostic}; Borzoi (Linder et al. 2025, the authors' tables) on FlowFISH+ and Gasperini x {gradient, ISM
(signed)}; AlphaGenome (Avsec et al. 2026) x {gradient, ISM} with the two published Extended Data Fig. 7a values.
Per panel: n, positives, negatives, % positive and chance (prevalence); the "reachable" row (share of the bar's pairs
inside the arm's receptive field, printed only where it is below 100%).

auPRC = sklearn's average_precision_score; the tables are those of Supplementary Figs. 1 and 2 (Borzoi ISM signed);
the AlphaGenome bars are the "All" auPRC of the Avsec et al. (2026) reproduction; reach = the score table's in_window
flag.
"""
from __future__ import annotations

import lib_e as L
from rep_linder2025 import signed_scores

A21 = "scores/avsec2021/avsec2021-enformer-{}.scores.tsv.gz"
L25 = "scores/linder2025/linder2025-borzoi-{}.scores.tsv.gz"
METHODS = ["Attention", "Gradient x input (K562)", "Gradient x input (agnostic)", "ISM (K562)", "ISM (agnostic)"]
# the figure's panels: (panel, truth table, gene key, {method: table})
PANELS = [
    ("Enformer | Fulco et al. (2019)", "fulco2019", "gene_symbol",
     {"Attention": A21.format("attention-fulco2019"), "Gradient x input (K562)": A21.format("gradient-fulco2019"),
      "Gradient x input (agnostic)": A21.format("gradient-agnostic-fulco2019"),
      "ISM (K562)": A21.format("ism-fulco2019"), "ISM (agnostic)": A21.format("ism-agnostic-fulco2019")}),
    ("Enformer | Gasperini et al. (2019)", "gasperini2019", "gene_id",
     {"Attention": A21.format("attention-gasperini2019"), "Gradient x input (K562)": A21.format("gradient-gasperini2019"),
      "Gradient x input (agnostic)": A21.format("gradient-agnostic-gasperini2019"),
      "ISM (K562)": A21.format("ism-gasperini2019"), "ISM (agnostic)": A21.format("ism-agnostic-gasperini2019")}),
    ("Borzoi | FlowFISH+ (Linder's own table)", "flowfish-plus-authors", "gene_id",
     {"Gradient x input (K562)": L25.format("gradient-flowfish-plus-authors"),
      "ISM (K562)": L25.format("ism-flowfish-plus-authors")}),
    ("Borzoi | Gasperini et al. (2019) (Linder's own table)", "gasperini2019-linder-authors", "gene_id",
     {"Gradient x input (K562)": L25.format("gradient-gasperini2019-linder-authors"),
      "ISM (K562)": L25.format("ism-gasperini2019-linder-authors")}),
]
#: the receptive field of every Enformer arm of a cohort is the K562 gradient run's in_window column (same anchor and
#: window for all five arms; re-deriving it per table with score_eg's geometry gives the same pairs)
REACH_TABLE = {"fulco2019": A21.format("gradient-fulco2019"), "gasperini2019": A21.format("gradient-gasperini2019")}


def labels_for(screen, gk):
    """(gene key, enhancer_id) -> 1 if significant in ('True', '1', 'TRUE')."""
    return {(r[gk], r["enhancer_id"]): int(r["significant"] in ("True", "1", "TRUE"))
            for r in L.read_rows(f"inputs/{screen}.hg38.tsv.gz")}


def bar(rel, lab, signed=None):
    """auPRC on the pairs of the table that are in the truth table. `signed` replaces a Borzoi ISM table's |.| score
    with the released code's signed per-pair mean, as the figure reads it."""
    y, v, keys, inwin = [], [], [], []
    for r in L.read_rows(rel):
        k = (r["gene"], r["enhancer"])
        if k in lab:
            y.append(lab[k])
            v.append(signed.get(k, float(r["score"])) if signed is not None else float(r["score"]))
            keys.append(k)
            inwin.append(r.get("in_window"))
    return L.ap_sklearn(y, v), len(y), int(sum(y)), keys, inwin


def values():
    v = {}
    sg = signed_scores()
    for title, screen, gk, paths in PANELS:
        lab = labels_for(screen, gk)
        reach = None
        if screen in REACH_TABLE:
            reach = {(r["gene"], r["enhancer"]): r["in_window"] == "True" for r in L.read_rows(REACH_TABLE[screen])}
        for m in METHODS:
            if m not in paths:
                continue
            rel = paths[m]
            signed = None
            if m == "ISM (K562)" and "linder2025" in rel:
                stem = rel.split("linder2025-borzoi-")[1].split(".scores")[0]
                signed = {(g, e): x for (a, g, e), x in sg.items() if a == stem}
            ap, n, pos, keys, inwin = bar(rel, lab, signed)
            cov = sum(reach[k] for k in keys) if reach is not None else sum(w == "True" for w in inwin)
            k = f"bar|{title}|{m}"
            v[f"{k}|auprc"], v[f"{k}|n"], v[f"{k}|pos"], v[f"{k}|reached"] = ap, float(n), float(pos), float(cov)
            v[f"{k}|reach_pct_1dp"] = L.reach_label(cov, n)[0]
        v[f"panel|{title}|n"], v[f"panel|{title}|pos"] = float(n), float(pos)
        v[f"panel|{title}|neg"], v[f"panel|{title}|pct_positive"] = float(n - pos), 100 * pos / n
        v[f"panel|{title}|chance"] = pos / n
    # AlphaGenome: the "All" auPRC of the Avsec et al. (2026) reproduction (every table row joined to the cohort)
    cohort = {(r["gene_id"], r["enhancer_id"]): int(r["significant"]) for r in L.read_rows("inputs/gschwind.hg38.tsv.gz")}
    tgt = L.read_json("paper/data/avsec2026_targets.json")["ed_fig7a"]
    title = "AlphaGenome | Gschwind et al. (2026) (ENCODE-rE2G)"
    for m, arm, key in (("Gradient x input (K562)", "gradient", "input x gradient"),
                        ("ISM (K562)", "ism", "RNA-seq scorer")):
        rows = L.read_rows(f"scores/avsec2026/avsec2026-alphagenome-{arm}-gschwind.scores.tsv.gz")
        y = [cohort[(r["gene"], r["enhancer"])] for r in rows]
        s = [float(r["score"]) for r in rows]
        # reach: these runs carry no in_window column and score out-of-window pairs exactly 0 (out_of_window: zero);
        # no in-window pair of either table scores 0, so the reached pairs are the nonzero ones
        cov = sum(x != 0.0 for x in s)
        k = f"bar|{title}|{m}"
        v[f"{k}|auprc"], v[f"{k}|n"], v[f"{k}|pos"], v[f"{k}|reached"] = L.ap_step(y, s), float(len(y)), float(sum(y)), \
            float(cov)
        v[f"{k}|reach_pct_1dp"] = L.reach_label(cov, len(y))[0]
        v[f"{k}|published"] = tgt[key]["AlphaGenome"]
    n, pos = len(y), sum(y)
    v[f"panel|{title}|n"], v[f"panel|{title}|pos"] = float(n), float(pos)
    v[f"panel|{title}|neg"], v[f"panel|{title}|pct_positive"], v[f"panel|{title}|chance"] = float(n - pos), \
        100 * pos / n, pos / n
    return v


if __name__ == "__main__":
    import time
    t0 = time.time()
    vals = values()
    print(f"{len(vals)} values in {time.time() - t0:.0f} s")
    for k, x in vals.items():
        if k.endswith("|auprc"):
            print(f"  {k:75s} {x:.3f}")
