#!/usr/bin/env python3
"""ism_statistic_tables.py -- reduce an ISM run's stored per-shuffle draws to any ISM statistic. No GPU, numpy only.

    python3 score_eg.py --config <cfg> [--set k=v ...] --fasta hg38.fa --out run.scores.tsv \
        --ism-draws run.ism_draws.npz
    python3 scripts/ism_statistic_tables.py --config <cfg> [--set k=v ...] --scores run.scores.tsv \
        --draws 'run*.ism_draws.npz' [--draws 'run_seed1*.ism_draws.npz'] [--strand-mask auto] \
        --prefix out/run-STAT

An ISM run reduces its draws d (shuffle x TSS x replicate x strand x track; d = u_wt - u_mut) to one score per pair
at its `ism_statistic`. `--ism-draws` keeps the draws, so every other reduction is post-processing. This script
writes one score table per statistic, `<prefix>-<statistic>.scores.tsv`, in the run's row order and with the run's
identity columns, so `scripts/evaluate.py` reads it directly:

    abs_mean_diff, abs_diff,            the engine's own reductions (score_eg.ism_reduce_draws), with the
    abs_mean_rel_diff, rel_abs_diff     run's ism_inner_mean
    log_mean_diff     |mean(d_log)|     the shuffle mean inside the absolute value
    log_abs_diff      mean(|d_log|)     the shuffle mean outside (the paper's matched ISM statistic)
    signed_mean       mean(d)           no absolute value and no log: the statistic Linder et al. (2025)'s released
                                        code uses (under log_scalar true the draws are already log fold changes)
    signed_rel_mean   mean(d / u_wt)    the relative response, no absolute value
    signed_log_mean   mean(d_log)       no absolute value

with d_log = log2(u_wt) - log2(u_mut) = -log2(1 - d / u_wt), exact rather than first-order; it is NaN where
u_wt ~ 0 or u_mut <= 0, and the means skip NaN.

--strand-mask auto keeps, for each gene, the tracks whose strand is the gene's own or unstranded, which is what the
engine's `strand_mask: auto` reads, from draws recorded at `strand_mask: off`. Each --draws pattern is one run
(its shard files hold disjoint pairs); several runs of the same pairs, for example five shuffles at `seed: 0` and
five at `seed: 1`, are concatenated along the shuffle axis in the order given. Pairs without draws (out of the
model's window) score 0, as in the run's own table.

GATES, both fatal: (1) with a single run and the full track set, the engine's reduction at the run's own
ism_statistic reproduces the run's `score` column exactly (skipped with --no-replay, e.g. for concatenated runs);
(2) every pair with draws has a gene strand and keeps at least one track under --strand-mask auto.
"""
from __future__ import annotations

import argparse
import csv
import glob
import gzip
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import score_eg as S  # noqa: E402

ID = ("screen", "gene", "enhancer", "chrom", "start", "end", "distance_to_tss", "in_window")
NATIVE = ("abs_mean_diff", "abs_diff", "abs_mean_rel_diff", "rel_abs_diff")
LOG2 = np.log(2.0)


def _log_draws(d: np.ndarray, wt: np.ndarray) -> np.ndarray:
    """-log2(1 - d / u_wt): the draw as a log-scale readout would have recorded it (u_wt ~ 0 or u_mut <= 0 -> NaN)."""
    denom = np.where(np.abs(wt) < 1e-12, np.nan, wt)
    ratio = 1.0 - d / denom
    ratio = np.where(ratio <= 0.0, np.nan, ratio)
    return -np.log(ratio) / LOG2


def log_mean_diff(d: np.ndarray, wt: np.ndarray) -> float:
    return float(np.abs(np.nanmean(_log_draws(d, wt))))


def log_abs_diff(d: np.ndarray, wt: np.ndarray) -> float:
    return float(np.nanmean(np.abs(_log_draws(d, wt))))


def signed_log_mean(d: np.ndarray, wt: np.ndarray) -> float:
    return float(np.nanmean(_log_draws(d, wt)))


def signed_mean(d: np.ndarray, wt: np.ndarray) -> float:
    return float(np.mean(d))


def signed_rel_mean(d: np.ndarray, wt: np.ndarray) -> float:
    denom = np.where(np.abs(wt) < 1e-12, np.nan, wt)
    return float(np.nanmean(d / denom))


DERIVED = {"log_mean_diff": log_mean_diff, "log_abs_diff": log_abs_diff, "signed_mean": signed_mean,
           "signed_rel_mean": signed_rel_mean, "signed_log_mean": signed_log_mean}


def _open(path):
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path)


def load_draws(patterns):
    """{(gene, enhancer): (d, wt)}; runs given as separate patterns are concatenated along the shuffle axis."""
    per = {}
    for pat in patterns:
        files = sorted(glob.glob(pat))
        if not files:
            raise SystemExit(f"no draw files match {pat!r}")
        run = {}
        for f in files:
            z = np.load(f, allow_pickle=False)     # the object-typed metadata fields are never read
            shp = tuple(int(v) for v in z["draw_shape"])
            for i, (g, e) in enumerate(zip(z["gene"], z["enhancer"])):
                k = (str(g), str(e))
                d = np.asarray(z["draws"][i]).reshape(shp)
                w = np.asarray(z["wt"][i]).reshape((1,) + shp[1:])
                if k in run:                  # a pair the screen lists twice is scored twice, identically
                    if not (np.array_equal(run[k][0], d) and np.array_equal(run[k][1], w)):
                        raise SystemExit(f"{k} appears twice in {pat!r} with different draws")
                    continue
                run[k] = (d, w)
        for k, (d, w) in run.items():
            if k in per:
                d0, w0 = per[k]
                if not np.array_equal(w0, w):
                    raise SystemExit(f"{k}: the runs disagree on the wild-type readout")
                per[k] = (np.concatenate([d0, d], axis=0), w0)
            else:
                per[k] = (d, w)
    return per


def gene_strands():
    """gene id -> strand, from the annotation the engine reads, with the engine's own id aliases."""
    path = os.path.join(ROOT, "data", "gencode_v50_tssdefs.json")
    fh = S._open_data(path)
    if fh is None:
        raise SystemExit(f"missing {path}(.gz)")
    with fh:
        defs = json.load(fh)
    strand = {g: r["strand"] for g, r in defs.items() if isinstance(r, dict) and "strand" in r}
    for old, new in getattr(S, "GENE_ID_ALIASES", {}).items():
        if old not in strand and new in strand:
            strand[old] = strand[new]
    return strand


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="the ISM run's config")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="the run's overrides")
    ap.add_argument("--scores", required=True, help="the run's score table (row order and identity columns)")
    ap.add_argument("--draws", action="append", required=True, help="glob of one run's .ism_draws.npz shards")
    ap.add_argument("--strand-mask", choices=("off", "auto"), default="off")
    ap.add_argument("--statistics", default=",".join(NATIVE + tuple(DERIVED)))
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--no-replay", action="store_true")
    a = ap.parse_args(argv)

    theta = S.fill_defaults(S.apply_overrides(S.load_config(a.config), a.set))
    S.validate(theta)
    if theta["method"] != "ism":
        raise SystemExit(f"{a.config}: method is {theta['method']!r}, not ism")
    rho = S.resolve(theta)
    track_strands = list(rho.track_strands)
    with _open(a.scores) as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    per = load_draws(a.draws)

    if not a.no_replay and len(a.draws) == 1:
        shipped = {(r["gene"], r["enhancer"]): float(r["score"]) for r in rows}
        worst = max(abs(float(np.mean(S.ism_reduce_draws(d, w, theta, tail=True))) - shipped[k])
                    for k, (d, w) in per.items() if k in shipped)
        if worst != 0.0:
            raise SystemExit(f"replay gate: the draws do not reproduce the run's score (worst {worst:.3e})")
        print(f"replay gate: {len(per):,} pairs reproduce the run's score exactly", file=sys.stderr)

    sel = {k: slice(None) for k in per}
    if a.strand_mask == "auto":
        if any(d.ndim < 5 for d, _ in per.values()):
            raise SystemExit("--strand-mask auto needs draws with a per-track axis (record them at strand_mask off "
                             "and track_abs_order before)")
        strand = gene_strands()
        for k in per:
            gs = strand.get(k[0])
            keep = [i for i, st in enumerate(track_strands) if st in (None, gs)] if gs else []
            if not keep:
                raise SystemExit(f"strand gate: {k} has no gene strand or keeps no track")
            sel[k] = keep

    stats = [s.strip() for s in a.statistics.split(",") if s.strip()]
    for stat in stats:
        if stat not in NATIVE and stat not in DERIVED:
            raise SystemExit(f"unknown statistic {stat!r}")
        th = dict(theta, ism_statistic=stat)
        out = {}
        for k, s in sel.items():
            d, w = per[k]
            d, w = d[..., s], w[..., s]
            out[k] = (float(np.mean(S.ism_reduce_draws(d, w, th, tail=True))) if stat in NATIVE
                      else DERIVED[stat](d, w))
        path = f"{a.prefix}-{stat}.scores.tsv"
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", newline="") as fh:
            wr = csv.writer(fh, delimiter="\t", lineterminator="\n")
            wr.writerow(list(ID) + ["score"])
            for r in rows:
                k = (r["gene"], r["enhancer"])
                wr.writerow([r[c] for c in ID] + [out.get(k, 0.0)])
        print(f"wrote {path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
