"""rep_linder2025.py -- Appendix B.2 of the paper: reproducing Linder et al. (2025) on the authors' own pair tables,
Supplementary Fig. 2.

    values() -> {label: float}

Panel b: 24 published bars (FlowFISH+ and Gasperini x Gaussian gradient, flat 2 kb gradient, ISM x 4 distance bins), each
with our auPRC, our 1,000-replicate bootstrap 95% interval, the published value and interval; "24 of 24"; the largest
gap of a gradient bar (the text's "within 0.002 in every bin"). The ISM bars use the released code's SIGNED statistic:
per pair, the mean of log u_ref - log u_mut over its 128 stored passes, read from
paper/data/linder2025_ism_signed_per_pair.tsv.gz (built from the stored per-pass draws).
Panel a: the authors' tables cut to the notebooks' cohort vs the published per-bin counts ("exactly in the two shortest
bins and within three pairs in the two longest"). Also the counts Supplementary Table 3 prints.
"""
from __future__ import annotations

import collections

import numpy as np

import lib_e as L

BIN_EDGES = [0, 15000, 45000, 98000, 262144]
BIN_LABELS = ["<15K", "15K-45K", "45K-98K", "98K-262K"]
ENFORMER_REACH = 196608 // 2 - 2000 // 2 - 64        # 97,240
BORZOI_REACH = 524288 // 2 - 2000 // 2 - 64          # 261,080
INTERSECT_WITHIN = 98000
GAUSS = "borzoi scores (gradients; ensemble; gaussian)"
# the arms in figure order -> (cohort key in the published values, pair table, target arm, figure arm)
ARMS = [
    ("gradient-flowfish-plus-authors", "flowfish-plus", "flowfish-plus-authors", GAUSS, "gradient"),
    ("gradient-flat2kb-flowfish-plus-authors", "flowfish-plus", "flowfish-plus-authors",
     "borzoi scores (gradients; ensemble; average)", "gradient-flat2kb"),
    ("ism-flowfish-plus-authors", "flowfish-plus", "flowfish-plus-authors", "borzoi scores (ism shuffle; ensemble)", "ism"),
    ("gradient-gasperini2019-linder-authors", "gasperini2019", "gasperini2019-linder-authors", GAUSS, "gradient"),
    ("gradient-flat2kb-gasperini2019-linder-authors", "gasperini2019", "gasperini2019-linder-authors",
     "borzoi scores (gradients; ensemble; average)", "gradient-flat2kb"),
    ("ism-gasperini2019-linder-authors", "gasperini2019", "gasperini2019-linder-authors",
     "borzoi scores (ism shuffle; ensemble)", "ism"),
]


def roc_auc(y, s):
    """auROC: mid-rank Mann-Whitney."""
    y = np.asarray(y, dtype=bool)
    s = np.asarray(s, dtype=float)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return L.NAN
    o = np.argsort(s, kind="mergesort")
    ss = s[o]
    ranks = np.empty(len(s), dtype=float)
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and ss[j + 1] == ss[i]:
            j += 1
        ranks[o[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def load_table(table):
    """The authors' pair table (distance, significant as int)."""
    rows = L.read_rows(f"inputs/{table}.hg38.tsv.gz")
    for r in rows:
        r["distance"] = int(r["distance"])
        r["gene"] = r["gene_id"] or r["gene_symbol"]
        r["significant"] = int(r["significant"])
    return rows


def apply_cohort(rows, cap=262144):
    """The notebooks' cohort cut for the authors' tables (gencode linder2025, exon_source all-transcripts,
    no min_elements_per_gene: the shipped Gasperini table already carries the >= 3 rule): the 262,144 bp cap, then
    Borzoi's window (the gene's merged-exon centroid within BORZOI_REACH of the element midpoint), then
    intersect_or_beyond: within 98 kb one of the gene's basic-isoform TSSs must also lie within ENFORMER_REACH."""
    cent = L.read_json("paper/data/gencode_linder2025_exon_centroids.json")
    basic = L.read_json("paper/data/gencode_linder2025_basic_tss.json")
    cn, bn = {}, {}
    for v in cent.values():
        cn.setdefault(v["name"], v)
    for v in basic.values():
        bn.setdefault(v["name"], v)
    kept = []
    for r in (r for r in rows if r["distance"] <= cap):
        gid = (r["gene_id"] or "").split(".")[0]
        mid = int(r["enhancer_mid"])
        c = cent.get(gid) or cn.get(r["gene_symbol"])
        if c is None or c["chrom"] != r["chrom"]:
            continue
        if abs(mid - c["centroid"]) > BORZOI_REACH:
            continue
        if r["distance"] <= INTERSECT_WITHIN:
            b = basic.get(gid) or bn.get(r["gene_symbol"])
            ts = b["tss"] if (b and b["chrom"] == r["chrom"]) else [int(r["tss"])]
            if min(abs(mid - t) for t in ts) > ENFORMER_REACH:
                continue
        kept.append(r)
    return kept


def signed_scores():
    """(arm, gene, enhancer) -> signed mean over the pair's 128 passes."""
    return {(r["arm"], r["gene"], r["enhancer"]): float(r["signed_mean"])
            for r in L.read_rows("paper/data/linder2025_ism_signed_per_pair.tsv.gz")}


_COHORTS = {}


def joined(stem, table, signed=None):
    """The join: the arm's score rows, in table order, whose (chrom, start, end, gene) is in the cohort of
    apply_cohort (key -> (distance, label, gene)); rows = (gene, distance, label, score)."""
    if table not in _COHORTS:
        _COHORTS[table] = {(r["chrom"], int(r["enhancer_start"]), int(r["enhancer_end"]), r["gene_id"]):
                           (r["distance"], int(r["significant"]), r["gene_id"]) for r in apply_cohort(load_table(table))}
    pairs = _COHORTS[table]
    rows = []
    for r in L.read_rows(f"scores/linder2025/linder2025-borzoi-{stem}.scores.tsv.gz"):
        hit = pairs.get((r["chrom"], int(r["start"]), int(r["end"]), r["gene"]))
        if hit is None:
            continue
        score = float(r["score"])
        if signed is not None:                       # the released code's signed reading
            score = signed.get((stem, r["gene"], r["enhancer"]), score)
        rows.append((hit[2], hit[0], hit[1], score))
    return rows


def binned(rows):
    """Per-bin (n, positives) of joined rows: bins (lo, hi] on the table's own distance."""
    return [(len(sub), sum(x[2] for x in sub)) for sub in
            ([x for x in rows if BIN_EDGES[i] < x[1] <= BIN_EDGES[i + 1]] for i in range(4))]


def bars():
    """The 24 bars, in ARMS order."""
    vals = L.read_json("paper/data/linder2025_fig4_values.json")
    signed = signed_scores()
    out = []
    for stem, cohort, table, target, arm in ARMS:
        tgt = vals["cohorts"][cohort]["arms"][target]
        rows = joined(stem, table, signed if arm == "ism" else None)
        for i in range(4):
            sub = [x for x in rows if BIN_EDGES[i] < x[1] <= BIN_EDGES[i + 1]]
            t = tgt["bins"][i]
            y = [x[2] for x in sub]
            s = [x[3] for x in sub]
            lo, hi = L.boot_ci_l25(y, s, L.ap_step)
            out.append(dict(screen=cohort, arm=arm, bin=BIN_LABELS[i], n=len(sub), pos=int(sum(y)),
                            mine=L.ap_step(y, s), mine_auc=roc_auc(y, s), ci_lo=lo, ci_hi=hi,
                            paper=t["ap"], paper_lo=t["ap_lo"], paper_hi=t["ap_hi"], paper_n=t["sites"],
                            paper_pos=t["pos"], paper_tss_ap=t["tss_ap"],
                            mine_tss_ap=L.ap_step(y, [1.0 / max(x[1], 1) for x in sub]),
                            prevalence=float(sum(y)) / len(sub), real_gap=not (lo <= t["ap"] <= hi),
                            overlap=not (hi < t["ap_lo"] or lo > t["ap_hi"])))
    return out


def counts():
    """Panel a and Supplementary Table 3's counts."""
    vals = L.read_json("paper/data/linder2025_fig4_values.json")
    out = {}
    for table, cohort, grad in (("flowfish-plus-authors", "flowfish-plus", "gradient-flowfish-plus-authors"),
                                ("gasperini2019-linder-authors", "gasperini2019",
                                 "gradient-gasperini2019-linder-authors")):
        rows = load_table(table)
        sel = apply_cohort(rows)
        tgt = vals["cohorts"][cohort]["arms"][GAUSS]["bins"]
        n, pos = [], []
        for i in range(4):
            sub = [r for r in sel if BIN_EDGES[i] < r["distance"] <= BIN_EDGES[i + 1]]
            n.append(len(sub))
            pos.append(sum(r["significant"] for r in sub))
        pub_n, pub_pos = [t["sites"] for t in tgt], [t["pos"] for t in tgt]
        neg, pub_neg = [a - b for a, b in zip(n, pos)], [a - b for a, b in zip(pub_n, pub_pos)]
        l1 = sum(abs(pos[i] - pub_pos[i]) + abs(neg[i] - pub_neg[i]) for i in range(4))
        # rows / positives of the table; the pairs the Borzoi gradient run scores,
        # labelled by the table (key: gene_id, chrom:start-end)
        truth = {(r["gene_id"], f"{r['chrom']}:{r['enhancer_start']}-{r['enhancer_end']}"): bool(r["significant"])
                 for r in rows}
        scored = [(r["gene"], r["enhancer"]) for r in
                  L.read_rows(f"scores/linder2025/linder2025-borzoi-{grad}.scores.tsv.gz")]
        out[table] = dict(n=n, pos=pos, neg=neg, pub_n=pub_n, pub_pos=pub_pos, pub_neg=pub_neg, misplaced=l1,
                          agreement_pct=100 * (1 - l1 / sum(pub_n)), cohort_rows=len(sel), rows=len(rows),
                          positives=sum(r["significant"] for r in rows), scored=len(scored),
                          scored_pos=sum(truth.get(k, False) for k in scored),
                          scored_all_in_table=float(all(k in truth for k in scored)),
                          balance=sum(neg) / max(sum(pos), 1), pub_balance=sum(pub_neg) / sum(pub_pos))
    return out


def values():
    v = {}
    bs = bars()
    for b in bs:
        k = f"bar|{b['screen']}|{b['arm']}|{b['bin']}"
        for f in ("n", "pos", "mine", "mine_auc", "ci_lo", "ci_hi", "paper", "paper_lo", "paper_hi", "paper_n",
                  "paper_pos", "paper_tss_ap", "mine_tss_ap", "prevalence"):
            v[f"{k}|{f}"] = float(b[f])
        v[f"{k}|consistent"] = float(not b["real_gap"])
        v[f"{k}|overlap"] = float(b["overlap"])
    v["tally|bars"] = float(len(bs))
    v["tally|consistent"] = float(sum(not b["real_gap"] for b in bs))
    v["gradient_bars|max_abs_gap"] = max(abs(b["mine"] - b["paper"]) for b in bs if b["arm"] != "ism")
    passes = [int(r["n_passes"]) for r in L.read_rows("paper/data/linder2025_ism_signed_per_pair.tsv.gz")]
    v["ism|passes_per_pair|min"], v["ism|passes_per_pair|max"] = float(min(passes)), float(max(passes))   # 128
    for scr in ("flowfish-plus", "gasperini2019"):
        mine = [b for b in bs if b["screen"] == scr and b["arm"] == "gradient"]
        v[f"binned_cohort|{scr}|pairs"] = float(sum(b["n"] for b in mine))       # 2,141 / 17,785
        v[f"binned_cohort|{scr}|positives"] = float(sum(b["pos"] for b in mine))  # 116 / 392
    for table, c in counts().items():
        for i, lab in enumerate(BIN_LABELS):
            for f in ("n", "pos", "neg", "pub_n", "pub_pos", "pub_neg"):
                v[f"counts|{table}|{lab}|{f}"] = float(c[f][i])
        v[f"counts|{table}|bins_1_2_max_abs_diff"] = float(max(abs(c["n"][i] - c["pub_n"][i]) for i in (0, 1)))
        v[f"counts|{table}|bins_3_4_max_abs_diff"] = float(max(abs(c["n"][i] - c["pub_n"][i]) for i in (2, 3)))
        for f in ("misplaced", "agreement_pct", "cohort_rows", "rows", "positives", "scored", "scored_pos",
                  "scored_all_in_table", "balance", "pub_balance"):
            v[f"counts|{table}|{f}"] = float(c[f])
        v[f"counts|{table}|published_pairs"] = float(sum(c["pub_n"]))
        v[f"counts|{table}|binned_positives"] = float(sum(c["pos"]))
    return v


if __name__ == "__main__":
    import time
    t0 = time.time()
    vals = values()
    print(f"{len(vals)} values in {time.time() - t0:.0f} s; tally {vals['tally|consistent']:.0f} of "
          f"{vals['tally|bars']:.0f}; max gradient gap {vals['gradient_bars|max_abs_gap']:.4f}")
