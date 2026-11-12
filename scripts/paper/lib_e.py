"""lib_e.py -- shared helpers for the modules of the paper's appendix: the three reproductions, the all-pairs
attribution figure, Appendix C.1's bootstrap and component-size figures, Supplementary Table 3's counts.

Every input is opened through harness.open_text / open_binary by its repository-relative path. numpy + stdlib only.
Each estimator below returns the same floating-point value, bit for bit, as the computation behind the paper's
figures: the four average-precision functions differ only in summation order, and the paper's values were computed
with each one in its place.
"""
from __future__ import annotations

import csv
import json

import numpy as np

import harness

NAN = float("nan")


# ------------------------------------------------------------------------------------------------------------------
# I/O
# ------------------------------------------------------------------------------------------------------------------
def read_rows(rel):
    """A tab-separated repository file as a list of dicts (csv.DictReader)."""
    with harness.open_text(rel) as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def read_json(rel):
    with harness.open_text(rel) as fh:
        return json.load(fh)


def truthy(v):
    """str(significant).strip().lower() in ("true", "1", "yes")."""
    return str(v).strip().lower() in ("true", "1", "yes")


# ------------------------------------------------------------------------------------------------------------------
# Average precision: four variants (identical values up to summation order)
# ------------------------------------------------------------------------------------------------------------------
def ap_tiegroup_seq(y, s):
    """Step-wise AP at distinct score thresholds, accumulated left to right (the Avsec et al. 2021 reproduction's).
    A loop `ap += p * (r - prev)` and this cumulative sum add the same terms in the same order."""
    s = np.asarray(s, float)
    y = np.asarray(y, bool)
    if s.size == 0:
        return NAN
    if not np.all(np.isfinite(s)):
        raise ValueError("ap_tiegroup_seq: non-finite score")
    o = np.argsort(-s, kind="mergesort")
    s, y = s[o], y[o]
    ends = np.flatnonzero(np.r_[np.diff(s) != 0, True])
    tp = np.cumsum(y)[ends]
    prec = tp / (ends + 1).astype(float)
    rec = tp / max(int(y.sum()), 1)
    return float(np.cumsum(prec * np.diff(np.r_[0.0, rec]))[-1])


def ap_step(y, s):
    """The same AP, summed with np.sum. Used by the Linder 2025 and Avsec 2026 reproductions and by the coverage
    experiment."""
    y = np.asarray(y, dtype=bool)
    s = np.asarray(s, dtype=float)
    if y.sum() == 0:
        return NAN
    o = np.argsort(-s, kind="mergesort")
    y, s = y[o], s[o]
    distinct = np.where(np.diff(s))[0]
    idx = np.r_[distinct, y.size - 1]
    tp = np.cumsum(y)[idx]
    fp = np.cumsum(~y)[idx]
    prec = tp / np.maximum(tp + fp, 1)
    rec = tp / y.sum()
    return float(np.sum(np.diff(np.r_[0.0, rec]) * prec))


def ap_sklearn(y, s):
    """sklearn.metrics.average_precision_score (scikit-learn 1.6.1; binary labels, pos_label 1, no weights), ported:
    _binary_clf_curve -> precision_recall_curve(drop_intermediate=False) -> -sum(diff(recall) * precision[:-1]).
    Supplementary Fig. 4's Enformer and Borzoi bars use it."""
    y_true = np.asarray(y) == 1
    y_score = np.asarray(s, dtype=float)
    desc = np.argsort(y_score, kind="mergesort")[::-1]
    y_score, y_true = y_score[desc], y_true[desc]
    thr = np.r_[np.where(np.diff(y_score))[0], y_true.size - 1]
    tps = np.cumsum(y_true * 1.0, dtype=np.float64)[thr]
    fps = 1 + thr - tps
    ps = tps + fps
    precision = np.zeros_like(tps)
    np.divide(tps, ps, out=precision, where=(ps != 0))
    recall = np.ones_like(tps) if tps[-1] == 0 else tps / tps[-1]
    precision = np.hstack((precision[::-1], 1))
    recall = np.hstack((recall[::-1], 0))
    return float(max(0.0, -np.sum(np.diff(recall) * np.array(precision)[:-1])))


def ap_group_mean(y, s):
    """Sum over tie groups of precision x group positives, / P
    (the figures' `1/distance on our pairs` lines)."""
    y = np.asarray(y, dtype=float)
    s = np.asarray(s, dtype=float)
    if len(y) == 0 or y.sum() == 0:
        return NAN
    o = np.argsort(-s, kind="mergesort")
    y, s = y[o], s[o]
    total = y.sum()
    ap, seen, tp, i = 0.0, 0, 0.0, 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        gp = y[i:j + 1].sum()
        tp += gp
        seen += j + 1 - i
        if gp:
            ap += (tp / seen) * gp
        i = j + 1
    return float(ap / total)


def inv_distance(d):
    """The model-free ranking score of every reproduction: 1 / max(d, 1)."""
    return 1.0 / max(int(d), 1)


# ------------------------------------------------------------------------------------------------------------------
# Bootstrap intervals over pairs, one per reproduction (they differ in generator, guard and row order)
# ------------------------------------------------------------------------------------------------------------------
def canonical(y, s):
    """Reorder by (score descending, label) so a resample is data-determined."""
    y = np.asarray(y, bool)
    s = np.asarray(s, float)
    o = np.lexsort((y, -s))
    return y[o], s[o]


def boot_ci_a21(y, s, n=2000, seed=1):
    """Avsec et al. (2021) reproduction: RandomState(seed), canonical order, replicates without a positive skipped,
    percentile 95% interval; (nan, nan) for a zero-width interval."""
    rng = np.random.RandomState(seed)
    y, s = canonical(y, s)
    if len(y) == 0 or y.sum() == 0:
        return NAN, NAN
    v = []
    for _ in range(n):
        i = rng.randint(0, len(y), len(y))
        if y[i].sum():
            v.append(ap_tiegroup_seq(y[i], s[i]))
    if not v:
        return NAN, NAN
    lo, hi = float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))
    if lo == hi:
        return NAN, NAN
    return lo, hi


def boot_ci_adaptive_a21(y, s, paper, n0=2000, n1=20000, k=4.0):
    """Adaptive interval: four 2,000-replicate probes (seeds 1, 11, 101, 1009); if the published value lies within
    k x the endpoint noise of probe 1's interval, the bar is recomputed at 20,000 replicates, seed 7919.
    Returns (lo, hi, n_used, noise, escalated)."""
    probes = [boot_ci_a21(y, s, n=n0, seed=sd) for sd in (1, 11, 101, 1009)]
    if not all(np.isfinite(p[0]) and np.isfinite(p[1]) for p in probes):
        return probes[0][0], probes[0][1], n0, NAN, False
    lo1, hi1 = probes[0]
    noise = max(float(np.std([p[0] for p in probes], ddof=1)), float(np.std([p[1] for p in probes], ddof=1)))
    margin = min(abs(paper - lo1), abs(hi1 - paper))
    if margin < k * noise:
        lo, hi = boot_ci_a21(y, s, n=n1, seed=7919)
        return lo, hi, n1, noise, True
    return lo1, hi1, n0, noise, False


def boot_ci_l25(y, s, fn, n=1000, seed=1):
    """Linder et al. (2025) reproduction: RandomState(seed) on the rows in the order
    they were joined (no canonical reordering), replicates without a positive skipped."""
    y, s = np.asarray(y), np.asarray(s, float)
    if len(y) == 0 or y.sum() == 0:
        return NAN, NAN
    rng = np.random.RandomState(seed)
    v = []
    for _ in range(n):
        i = rng.randint(0, len(y), len(y))
        if y[i].sum():
            v.append(fn(y[i], s[i]))
    return (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) if v else (NAN, NAN)


def boot_ci_a26(y, s, fn, n=2000, seed=1):
    """Avsec et al. (2026) reproduction: numpy Generator default_rng(seed), integers(0, n, n); a replicate with no
    positive or no negative is skipped."""
    y, s = np.asarray(y), np.asarray(s, float)
    if len(y) == 0 or y.sum() == 0:
        return NAN, NAN
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].sum() == 0 or y[i].sum() == len(i):
            continue
        vals.append(fn(y[i], s[i]))
    if not vals:
        return NAN, NAN
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


# ------------------------------------------------------------------------------------------------------------------
# The enhancer-gene graph
# ------------------------------------------------------------------------------------------------------------------
def component_roots(keys):
    """Connected components of the bipartite gene-enhancer graph; {key: root}. The union-find of
    evaluate.gene_blocks (path halving; the gene's root is attached under the enhancer's). The root of a
    component is a ('g', gene) or ('e', enhancer) tuple; the coverage experiment orders components by its str()."""
    parent = {}

    def find(x):
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for gid, eid in keys:
        ra, rb = find(("g", gid)), find(("e", eid))
        if ra != rb:
            parent[ra] = rb
    return {k: find(("g", k[0])) for k in keys}


def n_components(keys):
    return len(set(component_roots(keys).values()))


# ------------------------------------------------------------------------------------------------------------------
# Small formatting helpers shared by the modules (the figures' own rounding)
# ------------------------------------------------------------------------------------------------------------------
def reach_label(cov, n):
    """Fig 1's reachable-row label: '>99%' at a 1-dp share >= 99.5, else the 1-dp share printed with no decimals;
    None when every pair is reached."""
    reach = round(100 * cov / n, 1)
    if cov >= n:
        return reach, None
    return reach, (">99%" if reach >= 99.5 else f"{reach:.0f}%")
