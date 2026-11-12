"""rep_avsec2021.py -- Appendix B.1 of the paper: reproducing Avsec et al. (2021), Supplementary Fig. 1.

    values() -> {label: float}

Panel b: the 40 published bars (Fig. 2b and Extended Data Fig. 7c: 2 screens x 5 attribution readings x 4 distance
bins), each with our auPRC, its bootstrap 95% interval and the published value; "37 of 40 published bars consistent"
(published value inside our interval). Panel a: reconstructed vs published pair counts per bin, misplaced pairs,
% agreement (the text's 98.2% and 98.3%), class balance. Also the figure's 1/distance and prevalence lines, the
cohort sizes Supplementary Table 3 prints, and the recount under the Enformer usage notebook's crop that Appendix B.1
describes (the bars with the gradient contribution maps cut to that crop, "38 of 40"; `crop|...` labels).
"""
from __future__ import annotations

import csv

import numpy as np

import harness
import lib_e as L

BINS = [(0, 3000), (3000, 12500), (12500, 34500), (34500, 131000)]
LABELS = ["0-3 kb", "3-12.5 kb", "12.5-34.5 kb", "34.5-131 kb"]
ED7C_BINKEY = ["0-3kb", "3-12.5kb", "12.5-34.5kb", "34.5-131kb"]
SCREEN_TBL = {"fulco": "fulco2019", "gasperini": "gasperini2019"}
GENE_KEY = {"fulco": "gene_symbol", "gasperini": "gene_id"}
FIG2B_SCREEN = {"fulco": "Fulco et al.", "gasperini": "Gasperini et al."}
S = "scores/avsec2021/avsec2021-enformer-{}.scores.tsv.gz"

# the bars of Fig. 2b and Extended Data Fig. 7c, on the tables Supplementary Fig. 1 reads
# (screen, arm, readout, table, published record, series, column)
BARS = [
    ("fulco", "attention", "cell-type-agnostic", "attention-fulco2019", "fig2b", "Enformer Attention", None),
    ("fulco", "gradient", "K562", "gradient-fulco2019", "fig2b", "Enformer Gradient x input", None),
    ("fulco", "ism", "K562", "ism-fulco2019", "fig2b", "Enformer ISM", None),
    ("gasperini", "attention", "cell-type-agnostic", "attention-gasperini2019", "fig2b", "Enformer Attention", None),
    ("gasperini", "gradient", "K562", "gradient-gasperini2019", "fig2b", "Enformer Gradient x input", None),
    ("gasperini", "ism", "K562", "ism-gasperini2019", "fig2b", "Enformer ISM", None),
    ("fulco", "ism", "cell-type-agnostic", "ism-agnostic-fulco2019", "ed7c", "Enformer ISM", "all"),
    ("gasperini", "ism", "cell-type-agnostic", "ism-agnostic-gasperini2019", "ed7c", "Enformer ISM", "all"),
    ("fulco", "gradient", "cell-type-agnostic", "gradient-agnostic-fulco2019", "ed7c", "Enformer gradient*input", "all"),
    ("gasperini", "gradient", "cell-type-agnostic", "gradient-agnostic-gasperini2019", "ed7c",
     "Enformer gradient*input", "all"),
]


def labels_for(scr):
    """(gene key, chrom, start, end) -> significant == '1'."""
    gk = GENE_KEY[scr]
    return {(r[gk], r["chrom"], r["enhancer_start"], r["enhancer_end"]): r["significant"] == "1"
            for r in L.read_rows(f"inputs/{SCREEN_TBL[scr]}.hg38.tsv.gz")}


def bars():
    """The 40 bars (a list of dicts, in BARS order)."""
    fig2b = L.read_json("paper/data/avsec2021_fig2b_values.json")
    ed7c = L.read_json("paper/data/avsec2021_ed7c_values.json")
    lab = {scr: labels_for(scr) for scr in SCREEN_TBL}
    out = []
    for scr, arm, readout, table, rec, series, col in BARS:
        if rec == "fig2b":
            paper = [fig2b[f"{FIG2B_SCREEN[scr]}|{series}"][b] for b in LABELS]
        else:
            paper = [ed7c[f"{scr.capitalize()}|{series}|{col}|{k}"] for k in ED7C_BINKEY]
        floor = [fig2b[f"{FIG2B_SCREEN[scr]}|Random"][b] for b in LABELS]
        rows = []
        for r in L.read_rows(S.format(table)):
            k = (r["gene"], r["chrom"], r["start"], r["end"])
            if k in lab[scr]:
                rows.append((abs(int(r["distance_to_tss"])), lab[scr][k], float(r["score"])))
        for (lo, hi), nm, pv, fl in zip(BINS, LABELS, paper, floor):
            sub = [t for t in rows if lo <= t[0] < hi]
            y = np.array([t[1] for t in sub])
            s = np.array([t[2] for t in sub])
            if len(y) < 5 or y.sum() == 0:
                continue
            blo, bhi, bn, bnoise, besc = L.boot_ci_adaptive_a21(y, s, pv)
            out.append(dict(screen=scr, arm=arm, readout=readout, bin=nm, n=len(y), pos=int(y.sum()),
                            mine=L.ap_tiegroup_seq(y, s), ci_lo=blo, ci_hi=bhi, paper=pv, floor=fl, boot_n=bn,
                            ci_noise=bnoise, boot_escalated=besc, real_gap=not (blo <= pv <= bhi)))
    return out


def pair_counts():
    """Panel a: per bin, [lo, hi) on the pair table's `distance`, reconstructed vs Fig. 2b's published positives and
    negatives."""
    pub = L.read_json("paper/data/avsec2021_fig2b_pair_counts.json")
    edges = pub["bin_edges_bp"]
    out = {}
    for s in ("fulco2019", "gasperini2019"):
        rows = [(int(r["distance"]), int(r["significant"])) for r in L.read_rows(f"inputs/{s}.hg38.tsv.gz")]
        P, N = [0] * 4, [0] * 4
        for d, y in rows:
            for i in range(4):
                if edges[i] <= d < edges[i + 1]:
                    (P if y else N)[i] += 1
                    break
        pp, pn = pub["screens"][s]["positives"], pub["screens"][s]["negatives"]
        tot_pub = sum(pp) + sum(pn)
        l1 = sum(abs(P[i] - pp[i]) + abs(N[i] - pn[i]) for i in range(4))
        out[s] = dict(pos=P, neg=N, pub_pos=pp, pub_neg=pn, rows=len(rows), beyond_cap=len(rows) - sum(P) - sum(N),
                      reconstructed=sum(P) + sum(N), published=tot_pub, misplaced=l1,
                      agreement_pct=100 * (1 - l1 / tot_pub), balance=sum(N) / max(sum(P), 1),
                      pub_balance=sum(pn) / sum(pp), cohort_pos=sum(y for _, y in rows))
    return out


def distance_lines():
    """The figure's two baseline lines per bin: `1/distance on our pairs` = tie-grouped
    AP of 1/max(d, 1) over the scored rows (pair-table distance, bins (lo, hi]), and prevalence."""
    out = {}
    for scr, s in SCREEN_TBL.items():
        gk = GENE_KEY[scr]
        meta = {(r[gk], r["enhancer_id"]): (int(r["significant"] in ("True", "1", "TRUE")), int(r["distance"]))
                for r in L.read_rows(f"inputs/{s}.hg38.tsv.gz")}
        rows = [meta[(r["gene"], r["enhancer"])] for r in L.read_rows(S.format(f"gradient-{s}"))
                if (r["gene"], r["enhancer"]) in meta]
        for (lo, hi), lab in zip(BINS, LABELS):
            sub = [t for t in rows if lo < t[1] <= hi]
            y = [t[0] for t in sub]
            out[(s, lab)] = dict(n=len(sub), pos=int(sum(y)), prevalence=sum(y) / len(sub),
                                 mine_tss_ap=L.ap_group_mean(y, [L.inv_distance(d) for _, d in sub]))
    return out


CROP = 57344                      # enformer-usage.ipynb: the central 896 of 3,072 pooled 128 bp bins, [TSS - CROP, TSS + CROP)


def _ap_recount(y, s):
    """The tie-grouped step-wise AP, from the precision-recall curve and np.sum."""
    o = np.argsort(-s, kind="mergesort")
    s, y = s[o], y[o]
    ends = np.flatnonzero(np.r_[np.diff(s) != 0, True])
    tp = np.cumsum(y)[ends].astype(float)
    fp, P = (ends + 1) - tp, float(y.sum())
    return float(np.sum(tp / (tp + fp) * np.diff(np.r_[0.0, tp / P])))


def crop_recount():
    """The crop recount of Appendix B.1 ("38 of 40" under the crop): the 40 bars with the two gradient readouts'
    contribution maps cut to the Enformer usage notebook's crop: a 2 kb window wholly outside
    [anchor - 57,344, anchor + 57,344) of the gene's input anchor (the Enformer-predicted major TSS) scores 0, one
    straddling the edge takes its exact in-crop sum (paper/data/avsec2021_crop_straddlers.*.tsv), one wholly inside
    keeps its score; attention and ISM are untouched; binning by the score table's |distance_to_tss| ([lo, hi)); a
    plain 2,000-replicate pair bootstrap per bar (canonical order, RandomState(1))."""
    fig2b = L.read_json("paper/data/avsec2021_fig2b_values.json")
    ed7c = L.read_json("paper/data/avsec2021_ed7c_values.json")
    meta, anc, strad = {}, {}, {}
    for scr, tbl in SCREEN_TBL.items():
        gk = GENE_KEY[scr]
        for r in L.read_rows(f"inputs/{tbl}.hg38.tsv.gz"):
            meta[(scr, r[gk], r["chrom"], r["enhancer_start"], r["enhancer_end"])] = (r["significant"] == "1",
                                                                                      int(r["enhancer_mid"]))
        with harness.open_text(f"data/enformer_major_tss.{scr}.tsv.gz") as fh:
            rows = list(csv.DictReader((x for x in fh if not x.startswith("#")), delimiter="\t"))
        anc[scr] = {r["gene_key"]: int(r["major_tss"]) for r in rows}
        for r in L.read_rows(f"paper/data/avsec2021_crop_straddlers.{scr}.tsv"):
            strad[(scr, r["readout"], r["gene"], r["chrom"], r["start"], r["end"])] = float(r["s_crop"])
    out = []
    for scr, arm, readout, table, rec, series, col in BARS:
        paper = ([fig2b[f"{FIG2B_SCREEN[scr]}|{series}"][b] for b in LABELS] if rec == "fig2b" else
                 [ed7c[f"{scr.capitalize()}|{series}|{col}|{k}"] for k in ED7C_BINKEY])
        y, s, d = [], [], []
        for r in L.read_rows(S.format(table)):
            k = (scr, r["gene"], r["chrom"], r["start"], r["end"])
            if k not in meta:
                continue
            lab, mid = meta[k]
            sc = float(r["score"])
            if arm == "gradient":
                rel = mid - anc[scr][r["gene"]]
                if rel <= -CROP - 1000 or rel >= CROP + 1000:
                    sc = 0.0
                elif -CROP - 1000 < rel < -CROP + 1000 or CROP - 1000 < rel < CROP + 1000:
                    sc = strad[(scr, "k562" if readout == "K562" else "all", r["gene"], r["chrom"], r["start"],
                                r["end"])]
            y.append(lab)
            s.append(sc)
            d.append(abs(int(r["distance_to_tss"])))
        y, s, d = np.array(y, bool), np.array(s), np.array(d)
        for b, ((lo, hi), nm) in enumerate(zip(BINS, LABELS)):
            i = (d >= lo) & (d < hi)
            yy, ss = L.canonical(y[i], s[i])
            rng = np.random.RandomState(1)
            bs = []
            for _ in range(2000):
                j = rng.randint(0, len(yy), len(yy))
                if yy[j].sum():
                    bs.append(_ap_recount(yy[j], ss[j]))
            clo, chi = np.percentile(bs, 2.5), np.percentile(bs, 97.5)
            out.append(dict(screen=scr, arm=arm, readout=readout, bin=nm, mine=_ap_recount(yy, ss), ci_lo=float(clo),
                            ci_hi=float(chi), paper=paper[b], consistent=bool(clo <= paper[b] <= chi)))
    return out


def values():
    v = {}
    cr = crop_recount()
    v["crop|tally|bars"] = float(len(cr))
    v["crop|tally|consistent"] = float(sum(b["consistent"] for b in cr))
    for b in cr:                          # every verdict; the numbers of the bars that stay outside (as recorded)
        k = f"crop|bar|{b['screen']}|{b['arm']}|{b['readout']}|{b['bin']}"
        v[f"{k}|consistent"] = float(b["consistent"])
        if not b["consistent"]:
            for f in ("mine", "ci_lo", "ci_hi"):
                v[f"{k}|{f}"] = b[f]
    bs = bars()
    for b in bs:
        k = f"bar|{b['screen']}|{b['arm']}|{b['readout']}|{b['bin']}"
        for f in ("n", "pos", "mine", "ci_lo", "ci_hi", "paper", "floor", "boot_n"):
            v[f"{k}|{f}"] = float(b[f])
        v[f"{k}|consistent"] = float(not b["real_gap"])
    v["tally|bars"] = float(len(bs))
    v["tally|consistent"] = float(sum(not b["real_gap"] for b in bs))
    v["tally|escalated_to_20000"] = float(sum(b["boot_escalated"] for b in bs))
    for s, c in pair_counts().items():
        for i, lab in enumerate(LABELS):
            v[f"counts|{s}|{lab}|positives_reconstructed"] = float(c["pos"][i])
            v[f"counts|{s}|{lab}|positives_published"] = float(c["pub_pos"][i])
            v[f"counts|{s}|{lab}|negatives_reconstructed"] = float(c["neg"][i])
            v[f"counts|{s}|{lab}|negatives_published"] = float(c["pub_neg"][i])
        for f in ("reconstructed", "published", "misplaced", "agreement_pct", "balance", "pub_balance", "rows",
                  "beyond_cap"):
            v[f"counts|{s}|{f}"] = float(c[f])
        v[f"counts|{s}|positives_reconstructed"] = float(sum(c["pos"]))
        v[f"counts|{s}|positives_published"] = float(sum(c["pub_pos"]))
        v[f"cohort|{s}|pairs"] = float(c["rows"])                 # Supplementary Table 3: Fulco 4,777, Gasperini 38,582
        v[f"cohort|{s}|positives"] = float(c["cohort_pos"])       # (194; 634)
    for (s, lab), d in distance_lines().items():
        v[f"line|{s}|{lab}|mine_tss_ap"] = d["mine_tss_ap"]
        v[f"line|{s}|{lab}|prevalence"] = d["prevalence"]
    fig2b = L.read_json("paper/data/avsec2021_fig2b_values.json")
    for s, scr in (("fulco2019", "fulco"), ("gasperini2019", "gasperini")):
        for lab in LABELS:
            v[f"line|{s}|{lab}|paper_tss_ap"] = fig2b[f"{FIG2B_SCREEN[scr]}|1/distance"][lab]
    return v


if __name__ == "__main__":
    import time
    t0 = time.time()
    vals = values()
    print(f"{len(vals)} values in {time.time() - t0:.0f} s; tally {vals['tally|consistent']:.0f} of "
          f"{vals['tally|bars']:.0f}; agreement {vals['counts|fulco2019|agreement_pct']:.1f}% / "
          f"{vals['counts|gasperini2019|agreement_pct']:.1f}%")
