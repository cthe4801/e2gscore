"""sup_settings_counts.py -- the counts in Supplementary Table 3's "Pairs and statistics" column, recomputed from
the score tables' in_window flags and the cohorts' labels.

    values() -> {label: float}

The counts: supports, their pairs and positives, rankable genes, components, the 99 sub-cohort
components, the ISM supports, the three screens, the reproduction cohorts, the Borzoi-scored and binned Linder
cohorts, and the predicted-output and input lengths the Fig. 3 row prints (from the runs' manifests). The Linder
binned cohort is the join of rep_linder2025. Also the two ABC zero counts the table's Fig. 1A settings print.
Supports are keyed (gene, enhancer) as the score tables write them; labels are keyed (gene_id, chrom:start-end) from
the Gschwind table.
"""
from __future__ import annotations

from collections import defaultdict

RUN_MANIFEST = {
    "C1g": "scores/unified-t1/unified-t1-C1g-enformer-gene-midpoint-genetssreadout.manifest.json",
    "C2Rg": "scores/unified-t1/unified-t1-C2Rg-borzoi-gene-midpoint-genetssreadout.manifest.json",
    "A4g": "scores/unified-t1/unified-t1-A4g-alphagenome-gene-midpoint-genetssreadout.manifest.json",
    "A10g": "scores/unified-t1/unified-t1-A10g-alphagenome-gene-midpoint-procap-genetssreadout.manifest.json",
}

import lib_e as L
import rep_linder2025 as R25

U1 = "scores/unified-t1/unified-t1-{}.scores.tsv.gz"
RUN = {"A1w": "A1w-enformer-single-tss", "A2wR": "A2wR-borzoi-single-tss", "A3": "A3-alphagenome-single-tss",
       "C1g": "C1g-enformer-gene-midpoint-genetssreadout", "C2Rg": "C2Rg-borzoi-gene-midpoint-genetssreadout",
       "D1R": "D1R-borzoi-single-tss-rnaseq-tssnull", "F1R": "F1R-borzoi-gene-midpoint-rnaseq-tssnull",
       "A4g": "A4g-alphagenome-gene-midpoint-genetssreadout", "A5": "A5-alphagenome-single-tss-rnaseq",
       "A6": "A6-alphagenome-gene-midpoint-rnaseq", "A8": "A8-alphagenome-single-tss-procap",
       "A10g": "A10g-alphagenome-gene-midpoint-procap-genetssreadout",
       "S3n": "S3n-borzoi-gene-midpoint-rnaseq-tssnull-logscalar-none-rafter",
       "A11": "A11-alphagenome-gene-midpoint-rnaseq-logscalar", "O3": "O3-enformer-single-tss-majortss",
       "M1E": "M1E-enformer-single-tss-ism-eulerian-dinuc",
       "M1B": "M1B-borzoi-single-tss-ism-pertrack-strandoff-eulerian-dinuc-shuffle",
       "M1A": "M1-alphagenome-ism-eulerian-dinuc", "G3P": "G3P-alphagenome-single-tss-procap-ism-eulerian",
       "G2F": "G2F-borzoi-gene-midpoint-rnaseq-tssnull-ism-eulerian",
       "G3F": "G3F-alphagenome-gene-midpoint-rnaseq-tssnull-ism-eulerian"}
GEOMETRY_RUNS = ("C1g", "C2Rg", "A4g", "A10g")
_T = {}


def table(run):
    if run not in _T:
        _T[run] = L.read_rows(U1.format(RUN[run]))
    return _T[run]


def in_window(run):
    return {(r["gene"], r["enhancer"]) for r in table(run) if r["in_window"] == "True"}


def labels():
    return {(r["gene_id"], f"{r['chrom']}:{r['enhancer_start']}-{r['enhancer_end']}"): L.truthy(r["significant"])
            for r in L.read_rows("inputs/gschwind.hg38.tsv.gz")}


def counts(keys, lab):
    keys = [k for k in keys if k in lab]
    return len(keys), sum(lab[k] for k in keys)


def rankable(keys, lab):
    """Genes with at least one positive and one negative pair, and their pairs."""
    by = defaultdict(list)
    for k in keys:
        by[k[0]].append(lab[k])
    genes = [g for g, ys in by.items() if any(ys) and not all(ys)]
    return len(genes), sum(len(by[g]) for g in genes)


def screen_counts(lab, keys):
    """The K562 benchmark's Dataset column, joined on (chrom, start, end, gene)."""
    ds = {(r["chrom"], r["chromStart"], r["chromEnd"], r["measuredGeneEnsemblId"].split(".")[0]): r["Dataset"]
          for r in L.read_rows("paper/data/epcrispr_benchmark_K562.tsv.gz")}
    out = defaultdict(lambda: [0, 0])
    for g, e in keys:
        chrom, se = e.split(":")
        s, t = se.split("-")
        d = ds.get((chrom, s, t, g.split(".")[0]))
        out[d][0] += 1
        out[d][1] += int(lab[(g, e)])
    return {k: tuple(v) for k, v in out.items()}


def values():
    v = {}
    lab = labels()
    base = table("A1w")
    allk = [(r["gene"], r["enhancer"]) for r in base]
    dist = {(r["gene"], r["enhancer"]): abs(float(r["distance_to_tss"])) for r in base}
    S = {"all": set(allk)}
    S["rf-E"], S["rf-B"], S["rf-A"] = in_window("A1w"), in_window("A2wR"), in_window("A3")
    S["fig3-E"] = S["rf-E"] & in_window("C1g")
    S["fig3-B"] = S["rf-B"] & in_window("C2Rg") & in_window("D1R") & in_window("F1R")
    S["fig3-A"] = S["rf-A"] & in_window("A4g") & in_window("A5") & in_window("A6") & in_window("A8") & in_window("A10g")
    S["fig4-B-rna"] = in_window("F1R") & in_window("S3n")
    S["fig4-A-rna"] = in_window("A6") & in_window("A11")
    S["fig6-O3"] = S["rf-E"] & in_window("O3")
    S["E-out"] = S["all"] - S["rf-E"]
    S["bin0-10"] = {k for k in allk if dist[k] < 10_000}
    S["bin10-100"] = {k for k in allk if 10_000 <= dist[k] < 100_000}
    S["bin100+"] = {k for k in allk if dist[k] >= 100_000}
    for nm, rel in (("abc>0", "scores/baselines/abc.scores.tsv.gz"), ("abcstar>0", "scores/baselines/abcstar.scores.tsv.gz")):
        sc = {(r["gene"], r["enhancer"]): float(r["score"]) for r in L.read_rows(rel)}
        v[f"support|{nm}|covers_cohort"] = float(set(sc) == S["all"])
        S[nm] = {k for k, x in sc.items() if x > 0}
        # "0 for the 1,000 pairs" (ABC) and "0 for the same 1,000 pairs" (ABC*)
        v[f"support|{nm}|zero_scores"] = float(sum(x == 0.0 for x in sc.values()))
    for k in ("all", "rf-E", "rf-B", "rf-A", "fig3-E", "fig3-B", "fig3-A", "fig4-B-rna", "fig4-A-rna", "fig6-O3",
              "E-out", "bin0-10", "bin10-100", "bin100+", "abc>0", "abcstar>0"):
        v[f"support|{k}|pairs"], v[f"support|{k}|positives"] = map(float, counts(S[k], lab))
    v["support|fig4-A-rna^rf-A|symmetric_difference"] = float(len(S["fig4-A-rna"] ^ S["rf-A"]))   # "swaps 60 pairs"
    v["support|fig4-A-rna-rf-A|pairs"] = float(len(S["fig4-A-rna"] - S["rf-A"]))                   # 60 in ...
    v["support|rf-A-fig4-A-rna|pairs"] = float(len(S["rf-A"] - S["fig4-A-rna"]))                   # ... 60 out
    # Fig. 6's pooling row prints each run's OWN reachable pairs (Borzoi F1R, AlphaGenome A6), not an intersection
    for run in ("F1R", "A6"):
        v[f"support|{run}|pairs"], v[f"support|{run}|positives"] = map(float, counts(in_window(run), lab))
    for k in ("all", "rf-A", "rf-B", "rf-E"):
        v[f"rankable|{k}|genes"], v[f"rankable|{k}|pairs"] = map(float, rankable(S[k], lab))
    for k in ("all", "rf-E", "rf-B", "rf-A"):
        v[f"components|{k}"] = float(L.n_components(S[k]))
    v["subcohort|components"] = float(int(0.15 * 663))
    for run, k in (("M1E", "rf-E"), ("M1B", "rf-B"), ("M1A", "rf-A"), ("G3P", "rf-A")):
        v[f"ism_support|{run}|equals_{k}"] = float(in_window(run) == S[k])
    v["ism_support|G2F|equals_F1R"] = float(in_window("G2F") == in_window("F1R"))
    v["ism_support|G3F|pairs"], v["ism_support|G3F|positives"] = map(float, counts(in_window("G3F"), lab))
    for name, c in screen_counts(lab, S["all"]).items():
        v[f"screen|{name}|pairs"], v[f"screen|{name}|positives"] = float(c[0]), float(c[1])
    for f in ("fulco2019", "gasperini2019", "flowfish-plus-authors", "gasperini2019-linder-authors"):
        rows = L.read_rows(f"inputs/{f}.hg38.tsv.gz")
        v[f"cohort|{f}|pairs"] = float(len(rows))
        v[f"cohort|{f}|positives"] = float(sum(L.truthy(r["significant"]) for r in rows))
    for f, stem in (("flowfish-plus-authors", "gradient-flowfish-plus-authors"),
                    ("gasperini2019-linder-authors", "gradient-gasperini2019-linder-authors")):
        truth = {(r["gene_id"], f"{r['chrom']}:{r['enhancer_start']}-{r['enhancer_end']}"): L.truthy(r["significant"])
                 for r in L.read_rows(f"inputs/{f}.hg38.tsv.gz")}
        keys = [(r["gene"], r["enhancer"]) for r in L.read_rows(f"scores/linder2025/linder2025-borzoi-{stem}.scores.tsv.gz")]
        v[f"borzoi_scored|{f}|pairs"] = float(len(keys))
        v[f"borzoi_scored|{f}|positives"] = float(sum(truth.get(k, False) for k in keys))
        v[f"borzoi_scored|{f}|all_in_table"] = float(all(k in truth for k in keys))
        b = R25.binned(R25.joined(stem, f))
        v[f"linder_binned|{f}|pairs"] = float(sum(n for n, _ in b))
        v[f"linder_binned|{f}|positives"] = float(sum(p for _, p in b))
    for run in GEOMETRY_RUNS:
        g = L.read_json(RUN_MANIFEST[run])["runtime"]["frame_guard"]["geometry"]
        out_bp = g["n_bins"] * g["bin_size"]
        v[f"geometry|{run}|output_bp"], v[f"geometry|{run}|input_bp"] = float(out_bp), float(g["seq_len"])
        v[f"geometry|{run}|centred"] = float(2 * g["crop_bins"] * g["bin_size"] + out_bp == g["seq_len"])
    return v


if __name__ == "__main__":
    import time
    t0 = time.time()
    vals = values()
    print(f"{len(vals)} values in {time.time() - t0:.0f} s")
    for k, x in vals.items():
        print(f"  {k:60s} {x:g}")
