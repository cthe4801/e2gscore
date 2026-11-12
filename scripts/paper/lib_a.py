"""lib_a.py -- shared code for Fig. 1, Fig. 2 and the two "by support" supplementary figures (Supplementary Table 2
families fig1, fig2, sfig-support, sfig-wg-support). numpy + stdlib only.

Every input is read through harness.open_text by its repository path. The conventions:

  labels, Cohort        the cohort is the Enformer table's rows in file order, kept when the pair is in the
                        Gschwind cohort; a pair is positive when `significant` is true / 1 / yes; d = |distance_to_tss|.
  components            union-find over ("g", gene) and ("e", enhancer) nodes; components numbered by first appearance
                        in cohort order; members ascending.
  resamples             every figure of these families draws the same stream: default_rng(20260816), then per
                        replicate rng.integers(0, 663, 663), members concatenated in pick order; 10,000 replicates.
  ap_tie                the paper's tie-aware step-wise AP (one threshold per distinct score).
  boot_ap               each resample is restricted to the group's pairs, and a resample without a positive there is
                        dropped (`idx = [i for i in idx if y[i].sum() > 0]`).
  ci / pval             2.5 / 97.5 percentiles (numpy's default linear interpolation);
                        p = min(1, 2 * min(mean(d <= 0), mean(d >= 0))).
  auroc, within_gene    macro within-gene auROC, component draws.
  reach_label           the "reachable" row rule.
  bh_q                  BH within a family; a p below 2e-4 enters at 2e-4.

Speed only: the per-arm bootstraps run in parallel worker processes (E2G_WORKERS, default 6; set 1 for serial).
Each worker draws the same resample stream, so the result does not depend on the number of workers.
"""
from __future__ import annotations

import csv
import math
import os

import numpy as np

import harness

NBOOT, SEED = 10000, 20260816
WORKERS = int(os.environ.get("E2G_WORKERS", "6"))
COHORT = "inputs/gschwind.hg38.tsv.gz"
#: the matched protocol's score tables (Supplementary Table 3)
TABLE = {"Enformer": "scores/unified-t1/unified-t1-A1w-enformer-single-tss.scores.tsv.gz",
         "Borzoi": "scores/unified-t1/unified-t1-A2wR-borzoi-single-tss.scores.tsv.gz",
         "AlphaGenome": "scores/unified-t1/unified-t1-A3-alphagenome-single-tss.scores.tsv.gz",
         "ABC": "scores/baselines/abc.scores.tsv.gz",
         "ABC*": "scores/baselines/abcstar.scores.tsv.gz",
         "1/distance": None}            # 1 / max(d, 1) of the cohort table, as Fig. 1 computes it
DISTANCE_TABLE = "scores/baselines/distance.scores.tsv.gz"     # Fig. 2 reads the baseline's own table
ARMS = ["Enformer", "Borzoi", "AlphaGenome", "ABC", "ABC*", "1/distance"]
TRUNKS = ["Enformer", "Borzoi", "AlphaGenome"]
REF = "ABC"
#: Fig. 1A's four groups (all pairs and three distance bins): label -> (lo, hi) on d = |distance_to_tss|
FIG1_GROUPS = [("all pairs", 0, 10 ** 9), ("0–10 kb", 0, 10_000), ("10–100 kb", 10_000, 100_000),
               (">100 kb", 100_000, 10 ** 9)]
#: the by-support figure's supports: pairs with d below each receptive half-width
SUPPORTS = [("all", 10 ** 9), ("rf-alphagenome", 524288), ("rf-borzoi", 262144), ("rf-enformer", 98304)]
#: the within-gene figures' supports: every pair, or the pairs inside a trunk's window (in_window flag)
WG_SUPPORTS = [("all", None), ("rf-AlphaGenome", "AlphaGenome"), ("rf-Borzoi", "Borzoi"), ("rf-Enformer", "Enformer")]
#: values() labels: a support as the figures print it, and a paired contrast's statistics
SUPPORT_NAME = {"all": "all pairs", "rf-alphagenome": "rf-AlphaGenome", "rf-borzoi": "rf-Borzoi",
                "rf-enformer": "rf-Enformer", "rf-AlphaGenome": "rf-AlphaGenome", "rf-Borzoi": "rf-Borzoi",
                "rf-Enformer": "rf-Enformer"}
STAT_NAMES = [("delta", "delta"), ("lo", "95% CI low"), ("hi", "95% CI high"), ("p", "p")]


# ------------------------------------------------------------------------------------------------------- inputs
def read_rows(rel: str) -> list:
    with harness.open_text(rel) as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


_LAB = None


def labels() -> dict:
    """(gene_id, 'chrom:start-end') -> positive?"""
    global _LAB
    if _LAB is None:
        _LAB = {}
        for r in read_rows(COHORT):
            _LAB[(r["gene_id"], f"{r['chrom']}:{r['enhancer_start']}-{r['enhancer_end']}")] = \
                str(r["significant"]).strip().lower() in ("true", "1", "yes")
    return _LAB


def components(key: list):
    """Connected components of the gene-enhancer graph; also returns each pair's component index."""
    par = {}

    def find(x):
        par.setdefault(x, x)
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x
    for g, e in key:
        p, q = find(("g", g)), find(("e", e))
        if p != q:
            par[p] = q
    ci = {}
    cidx = np.array([ci.setdefault(find(("g", g))[1], len(ci)) for g, _ in key])
    return cidx, [np.where(cidx == i)[0] for i in range(len(ci))]


class Cohort:
    """The 10,353-pair cohort in the Enformer table's row order, its labels, distances and bootstrap components."""

    def __init__(self):
        lab = labels()
        rows = [r for r in read_rows(TABLE["Enformer"]) if (r["gene"], r["enhancer"]) in lab]
        self.key = [(r["gene"], r["enhancer"]) for r in rows]
        self.y = np.array([1 if lab[k] else 0 for k in self.key])
        self.d = np.array([abs(int(r["distance_to_tss"])) for r in rows])
        self.g = np.array([k[0] for k in self.key])
        self.cidx, self.mem = components(self.key)
        self._arm = {}

    @property
    def n(self) -> int:
        return len(self.y)

    def arm(self, name: str):
        """(score, in_window) aligned to the cohort; a pair missing from a table scores (0.0, False)."""
        if name not in self._arm:
            if TABLE[name] is None:
                self._arm[name] = (1.0 / np.maximum(self.d, 1), np.ones(self.n, bool))
            else:
                dd = {(r["gene"], r["enhancer"]): (float(r["score"]), str(r.get("in_window")).lower() == "true")
                      for r in read_rows(TABLE[name])}
                self._arm[name] = (np.array([dd.get(k, (0.0, False))[0] for k in self.key]),
                                   np.array([dd.get(k, (0.0, False))[1] for k in self.key]))
        return self._arm[name]

    def score(self, name: str) -> np.ndarray:
        return self.arm(name)[0]

    def reached(self, name: str) -> np.ndarray:
        """The pairs an arm scores: a trunk's window, a baseline's nonzero scores; 1/distance
        is never zero."""
        s, w = self.arm(name)
        return w if name in TRUNKS else (s != 0)

    def group(self, name: str) -> np.ndarray:
        """Named pair masks: Fig. 1A's groups, the distance supports, and the pairs outside Enformer's window."""
        for lab, lo, hi in FIG1_GROUPS:
            if name == lab:
                return (self.d >= lo) & (self.d < hi)
        for lab, thr in SUPPORTS:
            if name == lab:
                return self.d < thr
        if name == "outside Enformer's window":
            return ~self.arm("Enformer")[1]
        raise KeyError(name)


_COH = None


def cohort() -> Cohort:
    global _COH
    if _COH is None:
        _COH = Cohort()
    return _COH


# ------------------------------------------------------------------------------------------------------ metrics
def ap_tie(v, yy) -> float:
    """Tie-aware step-wise AP with one threshold per distinct score (scores first, labels second)."""
    v = np.asarray(v, dtype=float)
    yy = np.asarray(yy)
    P = yy.sum()
    if P == 0:
        return float("nan")
    o = np.argsort(-v, kind="mergesort")
    s, y = v[o], yy[o]
    last = np.r_[np.nonzero(np.diff(s))[0], len(s) - 1]
    tp = np.cumsum(y)[last]
    return float(np.sum(np.diff(np.r_[0.0, tp / P]) * (tp / (last + 1))))


def auroc(v, yy) -> float:
    """Tie-corrected Mann-Whitney AUC (average ranks within a tied run)."""
    P, N = int(yy.sum()), int((yy == 0).sum())
    if P == 0 or N == 0:
        return float("nan")
    o = np.argsort(v, kind="stable")
    s = v[o]
    r = np.empty(len(s), float)
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        r[i:j + 1] = (i + j) / 2.0 + 1.0
        i = j + 1
    rr = np.empty(len(s), float)
    rr[o] = r
    return float((rr[yy == 1].sum() - P * (P + 1) / 2.0) / (P * N))


def ci(a) -> tuple:
    a = np.asarray(a, float)
    a = a[~np.isnan(a)]
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))


def pval(a) -> float:
    """Two-sided bootstrap p: twice the smaller share of replicates on either side of 0."""
    a = np.asarray(a, float)
    a = a[~np.isnan(a)]
    return min(1.0, 2 * min(float((a <= 0).mean()), float((a >= 0).mean())))


# ---------------------------------------------------------------------------------------------------- bootstrap
def resamples(mem: list):
    """The 10,000 component resamples: (pick, member indices concatenated in pick order)."""
    rng = np.random.default_rng(SEED)
    for _ in range(NBOOT):
        pick = rng.integers(0, len(mem), len(mem))
        yield pick, np.concatenate([mem[t] for t in pick])


def _boot_one(job):
    """One arm over several groups, on every resample: the arm's AP (or, for score None, the resample's prevalence,
    y[i].mean()) on the resample restricted to the group; a resample without a positive in the group is dropped."""
    y, mem, masks, s = job
    out = {g: [] for g, _ in masks}
    for _, ix in resamples(mem):
        for g, m in masks:
            i = ix[m[ix]]
            if y[i].sum() > 0:
                out[g].append(y[i].mean() if s is None else ap_tie(s[i], y[i]))
    return {g: np.array(v) for g, v in out.items()}


def _map(fn, jobs: list) -> list:
    n = min(len(jobs), WORKERS)
    if n > 1:
        import multiprocessing as mp
        if "fork" in mp.get_all_start_methods():
            with mp.get_context("fork").Pool(n) as pool:
                return pool.map(fn, jobs)
    return [fn(j) for j in jobs]


_BOOT = {}
PREVALENCE = "__prevalence__"


def boot_ap(groups: list, arms: list) -> dict:
    """{(group, arm): (point AP on the group's pairs, AP on each kept resample)}, memoised. `arms` may include
    PREVALENCE (the group's positive rate instead of an AP)."""
    coh = cohort()
    todo = {}
    for a in arms:
        miss = [g for g in groups if (g, a) not in _BOOT]
        if miss:
            todo[a] = miss
    jobs = [(coh.y, coh.mem, [(g, coh.group(g)) for g in gs], None if a == PREVALENCE else coh.score(a))
            for a, gs in todo.items()]
    for (a, gs), res in zip(todo.items(), _map(_boot_one, jobs)):
        for g in gs:
            m = coh.group(g)
            pt = float(coh.y[m].mean()) if a == PREVALENCE else ap_tie(coh.score(a)[m], coh.y[m])
            _BOOT[(g, a)] = (pt, res[g])
    return {(g, a): _BOOT[(g, a)] for g in groups for a in arms}


def contrast(B: dict, group: str, a: str, b: str, rounded_points: bool = False) -> dict:
    """Paired contrast a - b on one group. delta: the unrounded point difference, or with rounded_points the
    difference of the 4-dp points (`round(pa, 4) - round(pb, 4)`, which Fig. 1A's brackets print)."""
    (pa, ra), (pb, rb) = B[(group, a)], B[(group, b)]
    dv = ra - rb
    lo, hi = ci(dv)
    delta = (round(pa, 4) - round(pb, 4)) if rounded_points else (pa - pb)
    return {"delta": delta, "lo": lo, "hi": hi, "p": pval(dv), "reps": len(dv)}


# -------------------------------------------------------------------------------------------------- within gene
_WG = {}


def within_gene(support: str) -> dict:
    """Macro within-gene auROC on one support (its genes, components and draws), for the six arms: per-gene
    auROCs, macro auROC, the macro auROC on each resample. Memoised."""
    if support in _WG:
        return _WG[support]
    coh = cohort()
    y, g, cidx = coh.y, coh.g, coh.cidx
    trunk = dict(WG_SUPPORTS)[support]
    m = np.ones(coh.n, bool) if trunk is None else coh.arm(trunk)[1]
    NC = len(coh.mem)
    comp_of_gene = {gg: cidx[np.where(g == gg)[0][0]] for gg in np.unique(g)}
    genes = [gg for gg in np.unique(g[m])
             if y[m & (g == gg)].sum() >= 1 and (y[m & (g == gg)] == 0).sum() >= 1]
    gidx = [np.where(m & (g == gg))[0] for gg in genes]
    gcomp = np.array([comp_of_gene[gg] for gg in genes])
    bycomp = {c: np.where(gcomp == c)[0] for c in np.unique(gcomp)}
    AUC = {a: np.array([auroc(coh.score(a)[ix], y[ix]) for ix in gidx]) for a in ARMS}
    rng = np.random.default_rng(SEED)
    picks = [rng.integers(0, NC, NC) for _ in range(NBOOT)]
    ba = {a: [] for a in ARMS}
    for pick in picks:
        sel = [bycomp[t] for t in pick if t in bycomp]
        gi = np.concatenate(sel) if sel else None
        for a in ARMS:
            ba[a].append(float(np.nanmean(AUC[a][gi])) if gi is not None else np.nan)
    _WG[support] = {"mask": m, "genes": genes, "gidx": gidx, "auc": AUC,
                    "macro": {a: float(np.nanmean(AUC[a])) for a in ARMS},
                    "boot": {a: np.array(v) for a, v in ba.items()}}
    return _WG[support]


def wg_contrast(W: dict, a: str, b: str = REF) -> dict:
    """Against the reference: delta of the macro auROCs, percentile CI and p of the per-resample difference (NaN
    resamples dropped). The paper prints p to 4 dp; it is a multiple of 2e-4 already."""
    dl = W["boot"][a] - W["boot"][b]
    lo, hi = ci(dl)
    return {"delta": W["macro"][a] - W["macro"][b], "lo": lo, "hi": hi, "p": pval(dl)}


# ------------------------------------------------------------------------------------------------------- labels
def reach_label(cov: int, n: int):
    """The "reachable" row: the 1-dp share, and the printed label (None when the arm reaches every pair)."""
    reach = round(100 * cov / n, 1)
    if cov >= n:
        return reach, None
    return reach, (">99%" if reach >= 99.5 else f"{reach:.0f}%")


def bh_q(tests: list, resolution: float = 2e-4, alpha: float = 0.05) -> dict:
    """Benjamini-Hochberg on ONE family: id -> (q, q_bound, rejected). A p below the bootstrap resolution enters at
    the resolution and its q is an upper bound; ranks break ties by name."""
    P = lambda p: resolution if p < resolution else float(p)                          # noqa: E731
    m = len(tests)
    order = sorted(range(m), key=lambda i: (P(tests[i]["p"]), tests[i]["name"]))
    k = 0
    for r, i in enumerate(order, 1):
        if P(tests[i]["p"]) <= r / m * alpha:
            k = r
    out, run = {}, 1.0
    for r in range(m, 0, -1):
        i = order[r - 1]
        run = min(run, P(tests[i]["p"]) * m / r)
        out[tests[i]["id"]] = [run, tests[i]["p"] < resolution - 1e-12, None]
    for r, i in enumerate(order, 1):
        out[tests[i]["id"]][2] = r <= k
    return {i: tuple(v) for i, v in out.items()}


def q_text(q: float, bound: bool) -> str:
    """The printed q: '$q = 0.0343$', or '$q < 0.0006$' for an upper bound, rounded up."""
    v = f"{math.ceil(q * 1e4 - 1e-9) / 1e4:.4f}" if bound else f"{q:.4f}"
    return f"$q {'<' if bound else '='} {v}$"
