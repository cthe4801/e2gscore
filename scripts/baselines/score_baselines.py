#!/usr/bin/env python3
"""ABC, ABC* and 1/distance on the K562 CRISPRi screen (Gschwind et al.), from shipped derived data.

    python3 scripts/baselines/score_baselines.py            # write the five tables to scores/baselines/
    python3 scripts/baselines/score_baselines.py --check    # recompute; compare with the shipped tables

Inputs (all in this repository):
    inputs/gschwind.hg38.tsv.gz              the 10,353 tested pairs
    paper/data/baselines/candidates.tsv.gz   143,676 candidate elements: class, ABC activity, its H3K27ac term
    paper/data/baselines/genes.tsv.gz        the TSS ABC and ABC* use for each of the 2,111 genes
    paper/data/baselines/hic_contact.tsv.gz  K562 Hi-C contact of each gene's TSS bin with every 5 kb bin
                                             that holds a candidate within 5 Mb (absent = 0)

Outputs, one score per tested pair, in the score-table format and row order of scores/unified-t1/:
    abc                ABC(E, G) = A_E C_EG / sum_{E' : |mid_E' - TSS_G| < 5 Mb} A_E' C_E'G
                       A = sqrt(quantile-normalised DNase x H3K27ac); C = Hi-C, rescaled to the reference
                       power law, plus a power-law pseudocount (the ENCODE ABC parameterisation)
    abcstar            the same loop with A = ABC's own H3K27ac term (the quantile-normalised H3K27ac factor
                       inside ABC's activity; no DNase) and C = ABC's distance contact at gamma = 1, i.e.
                       1 / (max(d, 5 kb) + 1)
    abc_numerator      A_E C_EG alone: ABC without its competing-element denominator
    abcstar_numerator  the same for ABC*
    distance           1 / max(|enhancer midpoint - TSS|, 1), with the screen's own TSS

In abc and abcstar a promoter candidate whose interval +-500 bp contains the gene's TSS scores 1 (ABC's
self-promoter rule). A tested element takes the maximum over the candidates it overlaps, and 0 if it overlaps none.
d is |candidate midpoint - TSS| with the midpoint (start + end) / 2.

Requires numpy. Every value is reproduced bit for bit on the platform the paper used; exp and log are the only
transcendental operations, so another CPU or numpy build may differ in the last bits (see README).
"""
from __future__ import annotations

import argparse
import csv
import gzip
import io
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join("paper", "data", "baselines")
SCREEN = os.path.join("inputs", "gschwind.hg38.tsv.gz")
TABLES = ("abc", "abcstar", "distance", "abc_numerator", "abcstar_numerator")
HEADER = ["screen", "gene", "enhancer", "chrom", "start", "end", "distance_to_tss", "in_window", "score"]

WINDOW = 5_000_000            # candidates whose midpoint lies strictly within 5 Mb of the TSS compete
HIC_BIN = 5_000               # Hi-C resolution; a candidate reads the bin holding its midpoint
TSS_SLOP = 500                # self-promoter rule
PSEUDOCOUNT_DISTANCE = 5_000  # the power law is clipped below at 5 kb
# ABC's power laws, contact(d) = exp(scale - gamma log(d + 1)) (the ABC repository's get_powerlaw_at_distance):
GAMMA_AVG, SCALE_AVG = 1.024238616787792, 5.9594510043736655   # its default "average Hi-C" fit
GAMMA_REFERENCE = 0.87                                          # its reference decay
SCALE_REFERENCE = -4.80 + 11.63 * GAMMA_REFERENCE               # its fit of scale on gamma across Hi-C maps


def powerlaw(d, gamma, scale):
    d = np.maximum(np.asarray(d, dtype=np.float64), float(PSEUDOCOUNT_DISTANCE))
    return np.exp(scale - gamma * np.log(d + 1.0))


def abc_contact(raw, d):
    """ABC's ENCODE contact: Hi-C rescaled to the reference decay, plus min(powerlaw(d), powerlaw(5 kb))."""
    pl_cell = powerlaw(d, GAMMA_AVG, SCALE_AVG)
    pl_ref = powerlaw(d, GAMMA_REFERENCE, SCALE_REFERENCE)
    pseudocount = float(powerlaw([PSEUDOCOUNT_DISTANCE], GAMMA_AVG, SCALE_AVG)[0])
    return raw * (pl_ref / pl_cell) + np.minimum(pl_cell, pseudocount)


def inverse_distance_contact(raw, d):
    """ABC's distance-based contact at gamma = 1, scale = 0: 1 / (max(d, 5 kb) + 1)."""
    return powerlaw(d, 1.0, 0.0)


def read_tsv(path):
    with io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def load(root):
    pairs = read_tsv(os.path.join(root, SCREEN))
    cand = read_tsv(os.path.join(root, DATA, "candidates.tsv.gz"))
    by_chrom = {}
    for r in cand:
        by_chrom.setdefault(r["chrom"], []).append(r)
    chroms = {}
    for c, rs in by_chrom.items():
        start = np.array([int(r["start"]) for r in rs], dtype=np.int64)
        end = np.array([int(r["end"]) for r in rs], dtype=np.int64)
        mid = (start + end) / 2.0
        o = np.argsort(mid, kind="stable")
        assert np.all(np.diff(mid[o]) > 0), f"{c}: candidate midpoints must be distinct"
        chroms[c] = dict(
            start=start[o], end=end[o], mid=mid[o], bin=(mid[o] // HIC_BIN).astype(np.int64),
            promoter=np.array([r["class"] == "promoter" for r in rs])[o],
            activity=np.array([float(r["activity"]) for r in rs])[o],
            h3k27ac=np.array([float(r["h3k27ac"]) for r in rs])[o])
    genes = {r["gene_id"]: (r["chrom"], int(r["tss"])) for r in read_tsv(os.path.join(root, DATA, "genes.tsv.gz"))}
    hic = {}
    for r in read_tsv(os.path.join(root, DATA, "hic_contact.tsv.gz")):
        hic.setdefault(r["gene_id"], {})[int(r["bin"])] = float(r["contact"])
    return pairs, chroms, genes, hic


def score_genes(chroms, genes, hic, activity, contact, normalise):
    """Per gene: (start, end, score) of every candidate in its 5 Mb window, in midpoint order."""
    out = {}
    for gene, (chrom, tss) in genes.items():
        c = chroms[chrom]
        d = np.abs(c["mid"] - tss)
        keep = d < WINDOW
        d = d[keep]
        row = hic.get(gene, {})
        raw = np.array([row.get(b, 0.0) for b in c["bin"][keep].tolist()], dtype=np.float64)
        num = c[activity][keep] * contact(raw, d)
        st, en = c["start"][keep], c["end"][keep]
        if normalise:
            tot = num.sum()
            s = num / tot if tot > 0 else np.zeros_like(num)
            own = c["promoter"][keep] & (st - TSS_SLOP < tss) & (en + TSS_SLOP > tss)
            s = np.where(own, 1.0, s)
        else:
            s = num
        out[gene] = (st, en, s)
    return out


def pair_scores(pairs, per_gene):
    """A tested element takes the maximum over the candidates it overlaps, 0 if none."""
    v = []
    for p in pairs:
        st, en, s = per_gene[p["gene_id"]]
        ov = (st < int(p["enhancer_end"])) & (en > int(p["enhancer_start"]))
        v.append(float(s[ov].max()) if ov.any() else 0.0)
    return v


def compute(root):
    pairs, chroms, genes, hic = load(root)
    assert len(pairs) == 10_353 and all(p["gene_id"] in genes for p in pairs)
    arms = {
        "abc": ("activity", abc_contact, True),
        "abcstar": ("h3k27ac", inverse_distance_contact, True),
        "abc_numerator": ("activity", abc_contact, False),
        "abcstar_numerator": ("h3k27ac", inverse_distance_contact, False),
    }
    scores = {k: pair_scores(pairs, score_genes(chroms, genes, hic, *a)) for k, a in arms.items()}
    dist = [int(p["enhancer_mid"]) - int(p["tss"]) for p in pairs]
    scores["distance"] = [1.0 / max(abs(x), 1) for x in dist]
    # rows in the order of every scores/unified-t1 table: chromosome (as text), then genes in order of first
    # appearance in the screen table, then screen-table order
    first = {}
    for i, p in enumerate(pairs):
        first.setdefault(p["gene_id"], i)
    order = sorted(range(len(pairs)), key=lambda i: (pairs[i]["chrom"], first[pairs[i]["gene_id"]], i))
    rows = [[pairs[i]["screen"], pairs[i]["gene_id"], pairs[i]["enhancer_id"], pairs[i]["chrom"],
             pairs[i]["enhancer_start"], pairs[i]["enhancer_end"], str(dist[i]), "True"] for i in order]
    return {k: [r + [repr(float(v[i]))] for r, i in zip(rows, order)] for k, v in scores.items()}


def table_bytes(rows):
    text = "\t".join(HEADER) + "\n" + "".join("\t".join(r) + "\n" for r in rows)
    buf = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=buf, mtime=0, compresslevel=9) as gz:
        gz.write(text.encode("utf-8"))
    return buf.getvalue()


def check(tables, ref_dir):
    """Compare every row with the shipped table: the eight key columns as text, the score as a double.

    PASS if the scores are bit-identical. They are on the platform the paper used; elsewhere exp and log may round
    differently in the last bit, so a table also passes if it differs only by rounding (relative 1e-12) AND ranks
    all pairs identically, ties included, which leaves every rank-based statistic (auPRC, auROC, bootstrap) unchanged.
    """
    ok = True
    for name in TABLES:
        with io.TextIOWrapper(gzip.open(os.path.join(ref_dir, f"{name}.scores.tsv.gz"), "rb"),
                              encoding="utf-8", newline="") as fh:
            ref = [line.rstrip("\r\n").split("\t") for line in fh]
        new = tables[name]
        if not (ref[0] == HEADER and len(ref) - 1 == len(new)
                and all(a[:-1] == b[:-1] for a, b in zip(ref[1:], new))):
            print(f"{name:18s} rows or keys differ from the shipped table  FAIL")
            ok = False
            continue
        x = np.array([float(a[-1]) for a in ref[1:]])
        y = np.array([float(b[-1]) for b in new])
        if np.array_equal(x, y):
            print(f"{name:18s} {len(y):,} rows; scores bit-identical  PASS")
            continue
        o = np.argsort(x, kind="stable")
        same_ranks = np.array_equal(np.sign(np.diff(x[o])), np.sign(np.diff(y[o])))
        nz = x != 0
        rel = float((np.abs(x - y)[nz] / np.abs(x[nz])).max()) if nz.any() else 0.0
        passed = same_ranks and np.array_equal(x == 0, y == 0) and rel < 1e-12
        ok &= passed
        print(f"{name:18s} {len(y):,} rows; {int((x == y).sum()):,} scores bit-identical, max relative difference "
              f"{rel:.2g}; ranks and ties {'identical' if same_ranks else 'DIFFER'}  "
              f"{'PASS (equal to rounding)' if passed else 'FAIL'}")
    return ok


def main() -> int:
    a = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    a.add_argument("--root", default=ROOT, help="repository root (default: inferred from this file)")
    a.add_argument("--out", default=None, help="output directory (default: <root>/scores/baselines)")
    a.add_argument("--check", action="store_true",
                   help="write nothing; compare with the tables in --out and exit 1 on any difference")
    args = a.parse_args()
    out = args.out or os.path.join(args.root, "scores", "baselines")
    tables = compute(args.root)
    if args.check:
        return 0 if check(tables, out) else 1
    os.makedirs(out, exist_ok=True)
    for name in TABLES:
        path = os.path.join(out, f"{name}.scores.tsv.gz")
        with open(path, "wb") as fh:
            fh.write(table_bytes(tables[name]))
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
