"""rep_avsec2026.py -- Appendix B.3 of the paper: reproducing Avsec et al. (2026) (AlphaGenome) Fig. 4j and Extended
Data Fig. 7a, Supplementary Fig. 3.

    values() -> {label: float}

Two arms (input x gradient; the RNA-seq variant-effect scorer, i.e. ISM) on the Gschwind et al. cohort, four bins each
(All, [0, 10), [10, 100), [100, 2,500) kb): auPRC, 2,000-replicate bootstrap 95% interval, n and positives. Five
published bars: the gradient arm's four Fig. 4j labels and the ISM arm's one Extended Data Fig. 7a bar; "5 of 5". Also
the cohort check gate 8 (10,353 rows, 3,676 out-of-window zeros, 453 in-window positives), the figure's 1/distance and
prevalence lines, and two extra columns of the evaluation (auROC, ENCODE-rE2G's trapezoidal auPRC).
"""
from __future__ import annotations

import numpy as np

import lib_e as L
from rep_linder2025 import roc_auc

ARMS = ("gradient", "ism")
ED7A_KEY = {"gradient": "input x gradient", "ism": "RNA-seq scorer"}
EXPECT_ROWS, EXPECT_OOW, EXPECT_POS_IN = 10353, 3676, 453          # gate 8


def re2g_auprc(y, s):
    """ENCODE-rE2G's auPRC: the PR curve without its extreme rows, integrated by trapezoid."""
    y = np.asarray(y, dtype=bool)
    s = np.asarray(s, dtype=float)
    if y.sum() == 0:
        return L.NAN
    o = np.argsort(-s, kind="mergesort")
    y, s = y[o], s[o]
    idx = np.r_[np.where(np.diff(s))[0], y.size - 1]
    tp = np.cumsum(y)[idx]
    fp = np.cumsum(~y)[idx]
    prec = tp / np.maximum(tp + fp, 1)
    rec = tp / y.sum()
    if len(rec) < 3:
        return L.NAN
    trapz = getattr(np, "trapz", None) or np.trapezoid
    return float(trapz(prec[:-1], rec[:-1]))


def evaluate(arm, tgt):
    """One arm's evaluation: per-bin auPRC with a 2,000-replicate bootstrap, and gate 8."""
    cohort = {(r["gene_id"], r["enhancer_id"]): (int(r["significant"]), int(r["distance"]))
              for r in L.read_rows("inputs/gschwind.hg38.tsv.gz")}
    rows = L.read_rows(f"scores/avsec2026/avsec2026-alphagenome-{arm}-gschwind.scores.tsv.gz")
    recs = []
    for r in rows:
        sig, dist = cohort[(r["gene"], r["enhancer"])]               # a scored pair outside the cohort raises
        recs.append((sig, dist, float(r["score"])))
    out = {"_gate8": dict(rows=len(rows), zeros=sum(1 for x in recs if x[2] == 0.0),
                          pos_in=sum(1 for x in recs if x[0] == 1 and x[2] != 0.0))}
    edges, labels = tgt["_bins"]["edges_bp"], tgt["_bins"]["labels"]
    pub = tgt["fig4j_zeroshot"]["AlphaGenome"] if arm == "gradient" else {}
    for k, lab in enumerate(labels):
        sub = recs if lab == "All" else [x for x in recs if edges[k - 1] <= x[1] < edges[k]]
        y = [x[0] for x in sub]
        s = [x[2] for x in sub]
        lo, hi = L.boot_ci_a26(y, s, L.ap_step, n=2000)
        row = dict(n=len(sub), pos=int(sum(y)), auPRC=L.ap_step(y, s), ci=[lo, hi], re2g_auprc=re2g_auprc(y, s),
                   auroc=roc_auc(y, s))
        if pub:
            row.update(published_label=pub[lab], published_inside_ci=bool(lo <= pub[lab] <= hi))
        out[lab] = row
    ed = tgt["ed_fig7a"][ED7A_KEY[arm]]["AlphaGenome"]
    lo, hi = out["All"]["ci"]
    out["_ed_fig7a"] = dict(published=ed, ours=out["All"]["auPRC"], inside_ci=bool(lo <= ed <= hi))
    return out, labels


def distance_lines(labels):
    """The Gschwind cohort's baseline lines: 1/distance AP and prevalence per bin ((lo, hi] on the pair
    table's distance) and over all pairs."""
    rows = [(int(r["significant"] in ("True", "1", "TRUE")), int(r["distance"]))
            for r in L.read_rows("inputs/gschwind.hg38.tsv.gz")]
    bins = {"0-10 kb": (0, 10000), "10-100 kb": (10000, 100000), "100-2500 kb": (100000, 2500000)}
    out = {}
    for lab in labels:
        sub = rows if lab == "All" else [t for t in rows if bins[lab][0] < t[1] <= bins[lab][1]]
        y = [t[0] for t in sub]
        out[lab] = dict(prevalence=sum(y) / len(sub),
                        mine_tss_ap=L.ap_group_mean(y, [L.inv_distance(d) for _, d in sub]))
    return out


def values():
    tgt = L.read_json("paper/data/avsec2026_targets.json")
    v, n_pub, n_ok = {}, 0, 0
    for arm in ARMS:
        ev, labels = evaluate(arm, tgt)
        for f in ("rows", "zeros", "pos_in"):
            v[f"gate8|{arm}|{f}"] = float(ev["_gate8"][f])
        v[f"gate8|{arm}|pass"] = float(ev["_gate8"] == dict(rows=EXPECT_ROWS, zeros=EXPECT_OOW, pos_in=EXPECT_POS_IN))
        for lab in labels:
            d = ev[lab]
            k = f"bar|{arm}|{lab}"
            v[f"{k}|n"], v[f"{k}|pos"], v[f"{k}|auPRC"] = float(d["n"]), float(d["pos"]), d["auPRC"]
            v[f"{k}|ci_lo"], v[f"{k}|ci_hi"] = d["ci"]
            v[f"{k}|re2g_auprc"], v[f"{k}|auroc"] = d["re2g_auprc"], d["auroc"]
            if "published_label" in d:
                v[f"{k}|published"] = d["published_label"]
                v[f"{k}|consistent"] = float(d["published_inside_ci"])
        ed = ev["_ed_fig7a"]
        v[f"ed7a|{arm}|published"], v[f"ed7a|{arm}|ours"] = ed["published"], ed["ours"]
        v[f"ed7a|{arm}|consistent"] = float(ed["inside_ci"])
        # the gradient arm is judged on its four per-bin labels, ISM on ED Fig. 7a only
        if arm == "gradient":
            n_pub += len(labels)
            n_ok += sum(bool(ev[lab]["published_inside_ci"]) for lab in labels)
        else:
            n_pub += 1
            n_ok += bool(ed["inside_ci"])
    v["tally|published_bars"], v["tally|consistent"] = float(n_pub), float(n_ok)
    pub_d = tgt["fig4j_zeroshot"]["1/TSS distance"]
    for lab, d in distance_lines(labels).items():
        v[f"line|{lab}|prevalence"], v[f"line|{lab}|mine_tss_ap"] = d["prevalence"], d["mine_tss_ap"]
        v[f"line|{lab}|paper_tss_ap"] = pub_d[lab]
    return v


if __name__ == "__main__":
    import time
    t0 = time.time()
    vals = values()
    print(f"{len(vals)} values in {time.time() - t0:.0f} s; tally {vals['tally|consistent']:.0f} of "
          f"{vals['tally|published_bars']:.0f}")
