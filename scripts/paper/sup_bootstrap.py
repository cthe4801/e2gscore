"""sup_bootstrap.py -- Appendix C.1 of the paper ("Comparison of bootstrap resampling units"): Supplementary Fig. 6
(sizes of the enhancer-gene connected components) and Supplementary Fig. 5 (coverage of a nominal 95% interval under
four resampling units).

    values() -> {label: float}        values(coverage_too=False) skips the coverage experiment (~4 min on 8 processes)

Supplementary Fig. 6 / C.1 text: components of the gene-enhancer bipartite graph of the K562 (Gschwind et al.) pair
table: counts, per-bin histogram (bins 1 | 2 | 3-4 | 5-8 | ...), median size, share spanning more
than one gene, components holding a positive, the largest component (pairs, genes, enhancers, positives, share), the
ten largest components' share, components of one or two pairs.

Supplementary Fig. 5: 1,000 sub-cohorts of 99 of the 663 components (15%, drawn without replacement, seed 20260817);
in each, a 600-replicate percentile 95% interval under each resampling unit (pair, enhancer, gene, component), and
whether it contains the full-cohort value. Five targets: the marginal auPRC of Enformer, Borzoi and AlphaGenome
(matched protocol) and two paired differences, Enformer tta rc ("real effect") and region all-tss ("null"). Coverage,
its +-1.96 binomial standard error, the mean interval width, and the full-cohort values (panel A's legend).

The interval of a replicate is computed from resampled COUNTS per distinct score instead of concatenating rows: the
same tie-grouped AP, value for value (same integers, same terms, same np.sum), without sorting each replicate.
"""
from __future__ import annotations

import multiprocessing as mp
import os

import numpy as np

import lib_e as L

U1 = "scores/unified-t1/unified-t1-{}.scores.tsv.gz"
CONTRASTS = [("MARGINAL auPRC (A1w Enformer)", "A1w-enformer-single-tss", None),
             ("MARGINAL auPRC (A2wR Borzoi)", "A2wR-borzoi-single-tss", None),
             ("MARGINAL auPRC (A3 AlphaGenome)", "A3-alphagenome-single-tss", None),
             ("DELTA A1w-O4 (real effect)", "A1w-enformer-single-tss", "O4-enformer-single-tss-ttarc"),
             ("DELTA A1w-J1 (null)", "A1w-enformer-single-tss", "J1-enformer-single-tss-alltss")]
UNITS = ("pair", "enhancer", "gene", "component")              # the four units the figure draws
OUTER, INNER, FRAC, SEED = 1000, 600, 0.15, 20260817            # Supplementary Fig. 5's design


# ------------------------------------------------------------------------------------------------------------------
# Supplementary Fig. 6: component sizes
# ------------------------------------------------------------------------------------------------------------------
def log2_bins(maxv):
    """Size bins 1 | 2 | 3-4 | 5-8 | 9-16 | ...: half-open bins doubling in width."""
    edges = [0.5, 1.5, 2.5]
    k = 2
    while edges[-1] < maxv + 1:
        edges.append(2 ** k + 0.5)
        k += 1
    return edges


def bin_labels(edges):
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        lo, hi = int(np.ceil(a)), int(np.ceil(b)) - 1
        out.append(f"{lo}" if lo == hi else f"{lo}-{hi}")
    return out


def tabulate(rel):
    """One row per component (pairs, genes, enhancers, positives), largest first (-pairs, -genes; the component id
    breaks ties, which no printed number depends on)."""
    keys, y = [], []
    for r in L.read_rows(rel):
        keys.append((r["gene_id"], r["enhancer_id"]))
        y.append(int(r["significant"]))
    root = L.component_roots(keys)
    comp = {}
    for (g, e), lab in zip(keys, y):
        c = comp.setdefault(root[(g, e)], dict(pairs=0, genes=set(), enhancers=set(), positives=0))
        c["pairs"] += 1
        c["genes"].add(g)
        c["enhancers"].add(e)
        c["positives"] += lab
    tab = [dict(pairs=c["pairs"], genes=len(c["genes"]), enhancers=len(c["enhancers"]), positives=c["positives"])
           for c in comp.values()]
    tab.sort(key=lambda c: (-c["pairs"], -c["genes"]))
    return tab, len(keys), sum(y)


def component_sizes():
    v = {}
    tabs = {"k562": tabulate("inputs/gschwind.hg38.tsv.gz")}
    edges = log2_bins(max(r["pairs"] for t, _, _ in tabs.values() for r in t))
    labels = bin_labels(edges)
    for key, (tab, n_pairs, n_pos) in tabs.items():
        pairs = np.array([r["pairs"] for r in tab])
        counts, _ = np.histogram(pairs, bins=edges)
        top = tab[0]
        cum = np.cumsum(pairs) / pairs.sum()
        k = f"components|{key}"
        v[f"{k}|pairs"], v[f"{k}|positives"], v[f"{k}|n_components"] = float(n_pairs), float(n_pos), float(len(tab))
        for lab, c in zip(labels, counts):
            v[f"{k}|bin {lab}"] = float(c)
        v[f"{k}|binned_total"] = float(counts.sum())
        v[f"{k}|median_pairs"] = float(int(np.median(pairs)))
        v[f"{k}|multi_gene_pct"] = 100 * float(np.mean([r["genes"] > 1 for r in tab]))
        v[f"{k}|with_a_positive"] = float(sum(r["positives"] > 0 for r in tab))
        for f in ("pairs", "genes", "enhancers", "positives"):
            v[f"{k}|largest_{f}"] = float(top[f])
        v[f"{k}|largest_share_pct"] = 100 * top["pairs"] / n_pairs
        v[f"{k}|top10_share_pct"] = 100 * float(cum[9])
        v[f"{k}|one_or_two_pairs"] = float(sum(r["pairs"] <= 2 for r in tab))
    return v


# ------------------------------------------------------------------------------------------------------------------
# Supplementary Fig. 5: coverage of a nominal 95% interval, per resampling unit
# ------------------------------------------------------------------------------------------------------------------
_W = {}          # the contrast a worker process scores (inherited through fork)


def _arm(stem):
    """(gene, enhancer) -> score."""
    return {(r["gene"], r["enhancer"]): float(r["score"]) for r in L.read_rows(U1.format(stem))}


class _Sorted:
    """One arm's scores on a sub-cohort, sorted once (descending, stable) and cut into distinct-score groups."""

    def __init__(self, s, y):
        self.order = np.argsort(-s, kind="mergesort")
        ss = s[self.order]
        self.starts = np.r_[0, np.flatnonzero(np.diff(ss) != 0) + 1]
        self.ypos = y[self.order].astype(np.int64)

    def ap(self, cnt, P):
        """The tie-grouped average precision of each resample (row r holds element i cnt[r, i] times), for all
        replicates at once. Per distinct score, the counts of rows and of positives; their running sums are the
        integers tp and tp + fp the row-level function forms at each threshold present in the resample. A score
        absent from a resample (count 0) repeats the previous tp, so its recall step is exactly 0 and the steps of the
        present scores are exactly the row-level ones; those terms, in the same order, go to the same np.sum."""
        c = cnt[:, self.order]
        gtot = np.add.reduceat(c, self.starts, axis=1)
        gpos = np.add.reduceat(c * self.ypos, self.starts, axis=1)
        tp = np.cumsum(gpos, axis=1)
        prec = tp / np.maximum(np.cumsum(gtot, axis=1), 1)
        rec = tp / P[:, None]
        terms = np.diff(rec, axis=1, prepend=0.0) * prec
        keep = gtot > 0
        return np.array([np.sum(terms[r][keep[r]]) for r in range(len(cnt))])


def _first_appearance(labels):
    """Block index of each element, blocks numbered in order of first appearance."""
    idx, out = {}, np.empty(len(labels), dtype=np.int64)
    for j, b in enumerate(labels):
        out[j] = idx.setdefault(b, len(idx))
    return out, len(idx)


def _rep(task):
    """One outer replicate: the sub-cohort `take`, then one interval per unit."""
    rep, take = task
    W = _W
    rows = np.concatenate([W["comps"][W["comp_ids"][t]] for t in take])
    rows.sort()
    y = W["y"][rows]
    if y.sum() < 5:
        return rep, None
    sub_keys = [W["keys"][i] for i in rows]
    arms = [_Sorted(W["sa"][rows], y)] + ([_Sorted(W["sb"][rows], y)] if W["sb"] is not None else [])
    yi = y.astype(np.int64)
    res = {}
    for u in UNITS:
        if u == "pair":
            lab = sub_keys
        elif u == "gene":
            lab = [k[0] for k in sub_keys]
        elif u == "enhancer":
            lab = [k[1] for k in sub_keys]
        else:                      # the sub-cohort's components are the drawn ones: same partition, any root
            lab = [W["root"][k] for k in sub_keys]
        block, n = _first_appearance(lab)
        rng = np.random.default_rng(SEED + rep)                    # fresh per unit
        cnt_block = np.empty((INNER, n), dtype=np.int64)
        for i in range(INNER):                                     # one draw of n blocks per replicate
            cnt_block[i] = np.bincount(rng.integers(0, n, size=n), minlength=n)
        cnt = cnt_block[:, block]                                  # rows each replicate holds, per element
        P = cnt @ yi                                               # positives per replicate (ys.sum())
        size = cnt.sum(axis=1)                                     # rows per replicate (ys.size)
        ok = (P != 0) & (P != size)
        vals = arms[0].ap(cnt[ok], P[ok])
        if len(arms) == 2:
            vals = vals - arms[1].ap(cnt[ok], P[ok])
        if vals.size < INNER // 2:
            res[u] = None
            continue
        res[u] = (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))
    return rep, res


def coverage_contrast(fa, fb, coh, procs):
    A = _arm(fa)
    B = _arm(fb) if fb else None
    keys = sorted((set(A) & set(coh)) if B is None else (set(A) & set(B) & set(coh)))
    y = np.array([coh[k] for k in keys], dtype=bool)
    sa = np.array([A[k] for k in keys])
    sb = np.array([B[k] for k in keys]) if B is not None else None
    truth = L.ap_step(y, sa) if sb is None else L.ap_step(y, sa) - L.ap_step(y, sb)
    root = L.component_roots(keys)
    comps = {}
    for i, k in enumerate(keys):
        comps.setdefault(root[k], []).append(i)
    comp_ids = sorted(comps, key=str)
    rng = np.random.default_rng(SEED)
    takes = [rng.choice(len(comp_ids), size=int(len(comp_ids) * FRAC), replace=False) for _ in range(OUTER)]
    _W.clear()
    _W.update(keys=keys, y=y, sa=sa, sb=sb, root=root, comps={c: np.array(v) for c, v in comps.items()},
              comp_ids=comp_ids)
    tasks = list(enumerate(takes))
    if procs > 1:
        with mp.get_context("fork").Pool(procs) as pool:
            done = dict(pool.imap_unordered(_rep, tasks, chunksize=8))
    else:
        done = dict(map(_rep, tasks))
    hits = {u: 0 for u in UNITS}
    tried = {u: 0 for u in UNITS}
    widths = {u: [] for u in UNITS}
    for rep in range(OUTER):                                       # in replicate order
        res = done[rep]
        if res is None:
            continue
        for u in UNITS:
            ci = res[u]
            if ci is None:
                continue
            tried[u] += 1
            widths[u].append(ci[1] - ci[0])
            hits[u] += ci[0] <= truth <= ci[1]
    units = {u: dict(coverage=hits[u] / max(tried[u], 1), n_reps=tried[u],
                     mean_width=float(np.mean(widths[u])) if widths[u] else L.NAN) for u in UNITS}
    return dict(n_pairs=len(keys), n_pos=int(y.sum()), n_components=len(comp_ids), population=truth, units=units)


def coverage(procs=None):
    procs = procs or int(os.environ.get("E2G_PROCS", min(8, os.cpu_count() or 1)))
    coh = {(r["gene_id"], r["enhancer_id"]): int(r["significant"]) for r in L.read_rows("inputs/gschwind.hg38.tsv.gz")}
    v = {}
    for name, fa, fb in CONTRASTS:
        c = coverage_contrast(fa, fb, coh, procs)
        k = f"coverage|{name}"
        v[f"{k}|population"] = c["population"]
        v[f"{k}|n_pairs"], v[f"{k}|n_pos"], v[f"{k}|n_components"] = float(c["n_pairs"]), float(c["n_pos"]), \
            float(c["n_components"])
        for u, d in c["units"].items():
            v[f"{k}|{u}|coverage"] = d["coverage"]
            v[f"{k}|{u}|n_reps"] = float(d["n_reps"])
            v[f"{k}|{u}|mean_width"] = d["mean_width"]
            v[f"{k}|{u}|se95"] = 1.96 * float(np.sqrt(d["coverage"] * (1 - d["coverage"]) / d["n_reps"]))
    v["coverage|design|sub_cohort_components"] = float(int(663 * FRAC))
    return v


def values(coverage_too=True, procs=None):
    v = component_sizes()
    if coverage_too:
        v.update(coverage(procs))
    return v


if __name__ == "__main__":
    import sys
    import time
    t0 = time.time()
    vals = values(coverage_too="--no-coverage" not in sys.argv)
    print(f"{len(vals)} values in {time.time() - t0:.0f} s")
    for k, x in vals.items():
        if k.endswith("coverage") or "largest" in k or "top10" in k:
            print(f"  {k:70s} {x:.4f}")
