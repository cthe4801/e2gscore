#!/usr/bin/env python3
"""
Evaluate score tables: auPRC with a block bootstrap, on several supports.

Three choices shape every number this prints.

Blocks, not pairs. A CRISPRi screen's pairs are not independent
observations: they share target genes and they share enhancers.
Resampling pairs gives intervals that are too narrow, so a replicate
here resamples whole connected components of the gene-enhancer graph.
`gene_blocks` carries the measurements behind that choice.

Several supports, side by side. The three trunks see 196 kb, 524 kb and
1 Mb of sequence, so they reach 17.3%, 38.5% and 64.5% of the cohort. A
single full-cohort auPRC therefore partly measures receptive field
rather than ranking skill, while a reachable-only auPRC hides that a
model cannot see most of the cohort. Both are real questions, so both
get reported, and the headline is the common support, where every
compared arm can see the pair.

Paired deltas. Two arms are scored on the same pairs, so most of the
bootstrap variance is shared and comparing marginal intervals is weak.
One set of resamples is drawn and every arm is scored on it, so each
difference carries an interval of its own.

    python3 scripts/evaluate.py scores/unified-t1/*.scores.tsv.gz
    python3 scripts/evaluate.py A1.scores.tsv A2.scores.tsv
    python3 scripts/evaluate.py A1.scores.tsv \\
        --column score__g2000t3-global-ms1
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

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

#: Distance bins, matching the reproductions so these numbers sit
#: beside theirs.
BINS = (("All", 0, 10 ** 12), ("0-10kb", 0, 10_000),
        ("10-100kb", 10_000, 100_000), ("100-2500kb", 100_000, 2_500_000))


def average_precision(y, s):
    """Area under the precision-recall curve.

    The sum over score thresholds of (R_n - R_{n-1}) * P_n, which is
    what `sklearn.metrics.average_precision_score` computes. Written
    out here so that evaluation needs nothing but numpy.
    """
    y = np.asarray(y, dtype=bool)
    s = np.asarray(s, dtype=float)
    if y.sum() == 0:
        return float("nan")
    o = np.argsort(-s, kind="mergesort")
    y, s = y[o], s[o]
    distinct = np.where(np.diff(s))[0]
    idx = np.r_[distinct, y.size - 1]
    tp = np.cumsum(y)[idx]
    fp = np.cumsum(~y)[idx]
    prec = tp / np.maximum(tp + fp, 1)
    rec = tp / y.sum()
    return float(np.sum(np.diff(np.r_[0.0, rec]) * prec))


def rf_half() -> dict:
    """Half the input window of each trunk, in bp.

    A pair is within a trunk's receptive field when the enhancer
    midpoint is within this distance of the anchor, which is the
    predicate `score_eg.pair_in_window` applies. Read from the model
    metadata so a metadata change cannot leave it stale.
    """
    import score_eg
    return {m: score_eg.MODEL_META[m].seq_len // 2
            for m in ("enformer", "borzoi", "alphagenome")}


def _open(path):
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path)


def load_cohort() -> dict:
    """(gene_id, enhancer_id) -> (label, distance).

    The screen, not any score table, is the source of truth for labels.
    """
    path = os.path.join(HERE, "inputs", "gschwind.hg38.tsv")
    if not os.path.exists(path):
        path += ".gz"
    coh = {}
    with _open(path) as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            coh[(r["gene_id"], r["enhancer_id"])] = (int(r["significant"]),
                                                     abs(int(r["distance"])))
    return coh


def load_arm(path: str, column: str) -> dict:
    """(gene, enhancer) -> (score, in_window).

    Refuses a table that is still an unmerged shard.
    """
    with _open(path) as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    if not rows:
        raise SystemExit(f"{path}: empty")
    if "_gene_rank" in rows[0]:
        raise SystemExit(f"{path} is an unmerged shard (carries "
                         f"_gene_rank); merge it first")
    if column not in rows[0]:
        raise SystemExit(f"{path} has no column {column!r}; "
                         f"has {sorted(rows[0])}")
    # A table without an in_window column cannot be told apart from
    # one where nothing is reachable, so the column's absence means
    # unknown rather than False.
    has_iw = "in_window" in rows[0]
    out = {}
    for r in rows:
        iw = ((r["in_window"].strip().lower() in ("true", "1"))
              if has_iw else None)
        out[(r["gene"], r["enhancer"])] = (float(r[column]), iw)
    return out


def supports(keys, coh, arms, half) -> dict:
    """name -> set of keys.

    Every support is defined on the cohort, not on any arm's scores.
    """
    sup = {"all": set(keys)}
    for model, h in sorted(half.items(), key=lambda kv: kv[1]):
        sup[f"rf-{model}"] = {k for k in keys if coh[k][1] < h}
    # per-arm reachability, and the intersection over every arm compared
    reach = {}
    for name, tab in arms.items():
        rs = {k for k in keys if tab[k][1] is True}
        if rs or any(tab[k][1] is False for k in keys):
            reach[name] = rs
    if reach:
        sup["common-reachable"] = set.intersection(*reach.values())
        if len(reach) == 1:
            sup["reachable"] = next(iter(reach.values()))
    return sup


def gene_blocks(keys, unit="component"):
    """Group row positions into exchangeable blocks.

    Returns (row order, block ids, block id -> row positions).

    Blocking by target gene fixes only half the dependency. A CRISPRi
    screen's pairs form a bipartite graph, each pair joining one gene
    to one enhancer, and enhancers are shared too: on Gschwind, 53.4%
    of enhancers appear in more than one pair and 82.2% of pairs touch
    a shared enhancer. The dependency is crossed rather than nested, so
    resampling genes leaves the enhancer dimension unaccounted for.

    The exchangeable unit is a connected component of that graph. Two
    components share no gene and no enhancer, so they are independent
    by construction, which is the property a block bootstrap needs. On
    Gschwind the 2,111 genes collapse to 663 components, the largest
    holding 982 pairs across 65 genes, leaving an effective sample size
    31% of the gene count.

    Blocking changes the interval, never the estimate. On arm A1w over
    the full cohort of 10,353 pairs and 471 positives, every unit
    returns auPRC 0.5142 over 2,000 replicates, and only the width of
    the 95% interval differs:

        pair (naive)  10,353 blocks   0.0952
        enhancer       3,941 blocks   0.0963
        gene           2,111 blocks   0.1131
        component        663 blocks   0.1321

    Gene-blocking captures most of the dependency, yet the component
    interval is another 17% wider, and 39% wider than the naive one,
    which is measured to undercover: 0.844 against a nominal 0.95.
    Enhancer-blocking on its own buys almost nothing, because the gene
    dimension carries most of the dependence. It is the crossing of the
    two that costs the 39%, so neither marginal unit substitutes for
    the component.

    The narrowing bites absolute per-arm reporting rather than the
    paired deltas, where all four units cover 0.95 to 0.98 at flat
    width: the within-component correlation the naive unit ignores is
    largely common to both arms and cancels in the difference.

    `unit` stays switchable because reporting both is more honest than
    asserting one; the gene figure is what a reader familiar with the
    usual practice expects to see.
    """
    order = sorted(keys)
    idx = {k: i for i, k in enumerate(order)}

    if unit == "pair":
        def key_of(k):
            return k
    elif unit == "gene":
        def key_of(k):
            return k[0]
    elif unit == "enhancer":
        def key_of(k):
            return k[1]
    elif unit == "component":
        # union-find over genes and enhancers jointly; a pair unions
        # its two endpoints
        parent: dict = {}

        def find(x):
            while parent.setdefault(x, x) != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for gid, eid in order:
            ra, rb = find(("g", gid)), find(("e", eid))
            if ra != rb:
                parent[ra] = rb

        def key_of(k):
            return find(("g", k[0]))
    else:
        raise SystemExit(f"unknown block unit {unit!r}")

    blocks: dict = {}
    for k in order:
        blocks.setdefault(key_of(k), []).append(idx[k])
    ids = sorted(blocks, key=lambda b: str(b))
    return order, ids, {b: np.array(v, dtype=np.int64)
                        for b, v in blocks.items()}


def evaluate(arms: dict, coh: dict, boot: int, seed: int, column: str,
             unit: str = "component", no_deltas: bool = False) -> dict:
    keys = sorted(set.intersection(*(set(t) for t in arms.values())))
    missing = {n: sorted(set(coh) - set(t))[:3] for n, t in arms.items()}
    for n, m in missing.items():
        if m:
            print(f"note: {n} is missing "
                  f"{len(set(coh) - set(arms[n])):,} cohort pairs "
                  f"(e.g. {m})", file=sys.stderr)
    keys = [k for k in keys if k in coh]
    half = rf_half()
    sup = supports(keys, coh, arms, half)

    order, block_ids, blocks = gene_blocks(keys, unit)
    y_all = np.array([coh[k][0] for k in order], dtype=np.int64)
    d_all = np.array([coh[k][1] for k in order], dtype=np.int64)
    s_all = {n: np.array([arms[n][k][0] for k in order], dtype=float)
             for n in arms}
    pos_of = {name: np.array([i for i, k in enumerate(order) if k in ks],
                             dtype=np.int64)
              for name, ks in sup.items()}

    rng = np.random.default_rng(seed)
    # One set of block resamples, reused by every arm, every support and
    # every bin. That sharing is what makes the deltas paired; drawing
    # fresh blocks per arm would silently unpair them.
    draws = [rng.integers(0, len(block_ids), len(block_ids))
             for _ in range(boot)]

    out: dict = {"column": column, "block_unit": unit,
                 "n_pairs": len(order), "n_blocks": len(block_ids),
                 "boot": boot, "seed": seed, "arms": sorted(arms),
                 "rf_half_bp": half, "results": [], "deltas": []}

    for sname in sorted(sup):
        keep = pos_of[sname]
        if keep.size == 0:
            continue
        for bname, lo, hi in BINS:
            m = keep[(d_all[keep] >= lo) & (d_all[keep] < hi)]
            if m.size == 0:
                continue
            y = y_all[m]
            npos, nneg = int(y.sum()), int((1 - y).sum())
            if npos == 0:
                continue
            inmask = np.zeros(len(order), dtype=bool)
            inmask[m] = True
            # coverage: of the cohort rows in this bin, what fraction
            # survives this support
            binall = (d_all >= lo) & (d_all < hi)
            cov_p = npos / max(1, int(y_all[binall].sum()))
            cov_n = nneg / max(1, int((1 - y_all[binall]).sum()))

            point = {n: average_precision(y, s_all[n][m]) for n in arms}
            # a replicate is a multiset of blocks; take every row of
            # each that this support keeps
            reps = {n: [] for n in arms}
            for dr in draws:
                sel = np.concatenate([blocks[block_ids[gi]] for gi in dr])
                sel = sel[inmask[sel]]
                if sel.size == 0:
                    continue
                ys = y_all[sel]
                if ys.sum() == 0 or ys.sum() == ys.size:
                    continue
                for n in arms:
                    reps[n].append(average_precision(ys, s_all[n][sel]))
            for n in arms:
                v = np.asarray(reps[n], dtype=float)
                out["results"].append({
                    "arm": n, "support": sname, "bin": bname,
                    "auprc": point[n], "n": int(m.size),
                    "n_pos": npos, "n_neg": nneg,
                    "prevalence": npos / m.size,
                    "coverage_pos": cov_p, "coverage_neg": cov_n,
                    "ci_lo": (float(np.percentile(v, 2.5))
                              if v.size else float("nan")),
                    "ci_hi": (float(np.percentile(v, 97.5))
                              if v.size else float("nan")),
                    "n_boot_used": int(v.size)})
            # Paired deltas, on the same replicates. Skippable because
            # the loop is quadratic in arms times columns, and the
            # cross product is usually not what is wanted: sweeping
            # the free grid, 65 cells is 2,080 pairwise comparisons.
            # Ask for deltas when the comparison is the question, so
            # two arms on one column, or a handful of cells against
            # each other.
            if no_deltas:
                continue
            names = sorted(arms)
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    a, b = names[i], names[j]
                    da = (np.asarray(reps[a], float)
                          - np.asarray(reps[b], float))
                    if da.size == 0:
                        continue
                    lo_, hi_ = (float(np.percentile(da, 2.5)),
                                float(np.percentile(da, 97.5)))
                    out["deltas"].append({
                        "a": a, "b": b, "support": sname, "bin": bname,
                        "delta": point[a] - point[b],
                        "ci_lo": lo_, "ci_hi": hi_,
                        # A two-sided bootstrap p: the smaller tail,
                        # doubled. Both tails count the ties, so two
                        # arms that coincide on a bin give two means of
                        # 1.0 and a p of 2.0 without the clamp.
                        "p_two_sided": float(
                            min(1.0, 2 * min((da <= 0).mean(),
                                             (da >= 0).mean()))),
                        "excludes_zero": bool(lo_ > 0 or hi_ < 0)})
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tables", nargs="+", help="score tables (globs allowed)")
    ap.add_argument("--column", default=["score"], nargs="+",
                    help="score column(s) to evaluate; a --free-grid run "
                         "also carries the score__<pooling>-<normalise> "
                         "grid. Naming several columns treats each as its "
                         "own arm, so the grid's cells get paired deltas "
                         "on shared resamples, which is the only sound "
                         "way to compare them: their marginal intervals "
                         "overlap heavily while the paired difference is "
                         "decisive")
    ap.add_argument("--boot", type=int, default=2000,
                    help="block-bootstrap replicates (default 2000)")
    ap.add_argument("--seed", type=int, default=1,
                    help="seed for the resampling draws (default 1)")
    ap.add_argument("--block", default="component",
                    choices=("component", "gene", "enhancer", "pair"),
                    help="bootstrap resampling unit. The default, "
                         "component, is a connected component of the "
                         "bipartite gene-enhancer graph, the only unit "
                         "exchangeable when both genes and enhancers are "
                         "shared across pairs (82.2%% of Gschwind pairs "
                         "touch a shared enhancer). gene blocks the gene "
                         "dimension alone and gives intervals ~14%% too "
                         "narrow; pair assumes full independence")
    ap.add_argument("--no-deltas", action="store_true",
                    help="report per-arm auPRC and interval only, "
                         "skipping the quadratic pairwise paired deltas. "
                         "Use when sweeping the free grid, where 65 cells "
                         "means 2,080 comparisons and the point estimate "
                         "and interval per cell is what a reader toggles "
                         "between")
    ap.add_argument("--json", default=None, help="write the full result here")
    a = ap.parse_args(argv)

    paths = [p for t in a.tables for p in (sorted(glob.glob(t)) or [t])]
    cols = list(a.column)
    arms = {}
    for p in paths:
        stem = (os.path.basename(p)
                .replace(".scores.tsv", "").replace(".gz", ""))
        for c in cols:
            # suffix the name only when there is something to
            # disambiguate, so the single-column case keeps the plain
            # arm name
            name = stem if len(cols) == 1 else f"{stem}:{c}"
            arms[name] = load_arm(p, c)
    if not arms:
        raise SystemExit("no tables")

    coh = load_cohort()
    res = evaluate(arms, coh, a.boot, a.seed, ",".join(cols), a.block,
                   a.no_deltas)

    print(f"\ncolumn {res['column']}   "
          f"{res['n_pairs']:,} pairs / "
          f"{res['n_blocks']:,} {res['block_unit']} blocks   "
          f"{res['boot']:,} block-bootstrap replicates\n")
    print(f"{'support':<18} {'bin':<12} {'arm':<44} {'auPRC':>7} "
          f"{'95% CI':>17} {'n':>7} {'pos':>5} {'cov+':>6}")
    print("-" * 122)
    last = None
    for r in res["results"]:
        # break on the support only: breaking on (support, bin) puts a
        # blank line between every single row whenever there is one
        # arm, which is the common case
        if last is not None and r["support"] != last:
            print()
        last = r["support"]
        print(f"{r['support']:<18} {r['bin']:<12} {r['arm'][:44]:<44} "
              f"{r['auprc']:>7.4f} [{r['ci_lo']:>6.4f}, {r['ci_hi']:>6.4f}] "
              f"{r['n']:>7,} {r['n_pos']:>5} {100 * r['coverage_pos']:>5.1f}%")

    if res["deltas"]:
        # Strip the prefix every arm shares before truncating.
        # Comparing four columns of one table otherwise prints four
        # identical labels, the differing part having fallen off the
        # end of the field.
        names = sorted(arms)
        pre = os.path.commonprefix(names) if len(names) > 1 else ""
        if len(pre) > 4 and any(len(n) > len(pre) for n in names):
            def shorten(n):
                return n[len(pre):] or n
        else:
            def shorten(n):
                return n
        if pre and shorten(names[0]) != names[0]:
            print(f"\n(arm names share the prefix {pre!r}; stripped below)")
        print("\nPAIRED delta-auPRC, same gene resamples "
              "(an interval excluding 0 is a real difference)\n")
        print(f"{'support':<18} {'bin':<12} {'A vs B':<50} "
              f"{'delta':>8} {'95% CI':>19} {'p':>7}")
        print("-" * 120)
        for d in res["deltas"]:
            star = " *" if d["excludes_zero"] else "  "
            label = f"{shorten(d['a'])} vs {shorten(d['b'])}"
            print(f"{d['support']:<18} {d['bin']:<12} {label[:50]:<50} "
                  f"{d['delta']:>+8.4f} "
                  f"[{d['ci_lo']:>+7.4f}, {d['ci_hi']:>+7.4f}] "
                  f"{d['p_two_sided']:>6.3f}{star}")

    if a.json:
        with open(a.json, "w") as fh:
            json.dump(res, fh, indent=1, sort_keys=True)
        print(f"\nwrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
