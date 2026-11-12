#!/usr/bin/env python3
"""Recompute ABC's Hi-C contact, paper/data/baselines/hic_contact.tsv.gz, from the ENCODE .hic file.

    python3 scripts/baselines/abc_contact_from_hic.py ENCFF621AIY.hic             # compare with the shipped table
    python3 scripts/baselines/abc_contact_from_hic.py ENCFF621AIY.hic --out FILE  # also write a recomputed table

ENCFF621AIY is ENCODE's K562 in situ Hi-C map (GRCh38, 33,783,625,697 bytes):
    https://www.encodeproject.org/files/ENCFF621AIY/@@download/ENCFF621AIY.hic

The contact ABC reads from a .hic file: the 5 kb, SCALE-normalised observed matrix, then
 1. divided by its chromosome's mean row sum (each bin's summed contacts, averaged over bins with any contact),
    which puts every chromosome on a common, roughly doubly stochastic scale;
 2. for each gene, the row of the 5 kb bin that holds its TSS (genes.tsv.gz), out to 5 Mb either side;
 3. with that bin's self-contact replaced by the larger of its two neighbours (the diagonal is dominated by
    self-ligation).
The table keeps, for each gene, the nonzero contacts at bins that hold the midpoint of a candidate within 5 Mb of the
TSS; every other bin is 0. score_baselines.py then rescales them to ABC's reference power law and adds the
pseudocount.

Requires numpy and hicstraw (pip install hicstraw). One pass over each of the 23 chromosomes: about 15 minutes on
one core and about 1 GB of memory. Exits 1 unless every value is bit-identical to the shipped table.
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
DATA = os.path.join(ROOT, "paper", "data", "baselines")
RES = 5_000
WINDOW = 5_000_000
BAND = 8_000                  # rows per query when summing whole rows, as the ABC repository walks the matrix


def read_tsv(path):
    with io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def row_mean(mat, size):
    """Mean summed contact per bin. Records are upper-triangular, so an off-diagonal value counts for both of its
    bins; a record that the neighbouring band also returns is halved."""
    per_bin = {}
    step = BAND * RES
    for s in range(0, size, step):
        e = s + step - RES
        for r in mat.getRecords(s, e, 0, size):
            v = r.counts
            if r.binX < s or r.binY > e:
                v /= 2.0
            bx, by = r.binX // RES, r.binY // RES
            per_bin[bx] = per_bin.get(bx, 0.0) + v
            if bx != by:
                per_bin[by] = per_bin.get(by, 0.0) + v
    return sum(per_bin.values()) / max(len(per_bin), 1)


def tss_row(mat, size, tss):
    """{bin: contact} for the row of the TSS bin within 5 Mb, diagonal replaced by its larger neighbour."""
    tb = tss // RES
    lo, hi = max(0, tb * RES - WINDOW), min(size, tb * RES + WINDOW)
    v = {}
    for r in mat.getRecords(tb * RES, (tb + 1) * RES - 1, lo, hi):
        other = r.binY // RES if r.binX // RES == tb else r.binX // RES
        v[other] = max(v.get(other, 0.0), r.counts)
    v[tb] = max(v.get(tb - 1, 0.0), v.get(tb + 1, 0.0))
    return v


def main() -> int:
    a = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    a.add_argument("hic", help="ENCFF621AIY.hic")
    a.add_argument("--data", default=DATA, help="directory holding genes.tsv.gz, candidates.tsv.gz, hic_contact.tsv.gz")
    a.add_argument("--out", default=None, help="write the recomputed contact table here")
    args = a.parse_args()
    import hicstraw

    genes = read_tsv(os.path.join(args.data, "genes.tsv.gz"))
    mids = {}
    for r in read_tsv(os.path.join(args.data, "candidates.tsv.gz")):
        mids.setdefault(r["chrom"], []).append((int(r["start"]) + int(r["end"])) / 2.0)
    mids = {c: np.array(m) for c, m in mids.items()}

    hic = hicstraw.HiCFile(args.hic)
    sizes = {c.name: c.length for c in hic.getChromosomes()}
    per_gene = {}
    for chrom in sorted({g["chrom"] for g in genes}, key=lambda c: (len(c), c)):
        mat = hic.getMatrixZoomData(chrom, chrom, "observed", "SCALE", "BP", RES)
        rm = row_mean(mat, sizes[chrom])
        here = [g for g in genes if g["chrom"] == chrom]
        for g in here:
            tss = int(g["tss"])
            row = tss_row(mat, sizes[chrom], tss)
            m = mids[chrom]
            bins = np.unique((m[np.abs(m - tss) < WINDOW] // RES).astype(np.int64)).tolist()
            per_gene[g["gene_id"]] = [(b, row[b] / rm) for b in bins if row.get(b, 0.0) != 0.0]
        print(f"{chrom}: mean row sum {rm:.2f}; {len(here)} genes", flush=True)
    rows = [(g["gene_id"], b, c) for g in genes for b, c in per_gene[g["gene_id"]]]

    new = {(g, b): c for g, b, c in rows}
    old = {(r["gene_id"], int(r["bin"])): float(r["contact"])
           for r in read_tsv(os.path.join(args.data, "hic_contact.tsv.gz"))}
    same_keys = set(new) == set(old)
    n_same = sum(1 for k, c in new.items() if old.get(k) == c)
    print(f"{len(new):,} (gene, bin) contacts; keys {'identical' if same_keys else 'DIFFER'} to the shipped table; "
          f"bit-identical {n_same:,}/{len(old):,}")
    if args.out:
        text = "gene_id\tbin\tcontact\n" + "".join(f"{g}\t{b}\t{c!r}\n" for g, b, c in rows)
        with open(args.out, "wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0,
                                                        compresslevel=9) as gz:
            gz.write(text.encode("utf-8"))
        print(f"wrote {args.out}")
    return 0 if same_keys and n_same == len(old) else 1


if __name__ == "__main__":
    sys.exit(main())
