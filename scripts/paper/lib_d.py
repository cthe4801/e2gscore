"""lib_d.py -- shared code for the families of Appendix D and E.

Families: screen-Gasperini2019 / screen-Nasser2021 / screen-Schraivogel2020 (the screens figure, one family per screen
column), sfig-ism (ISM under the matched protocol), sfig-cpg (ISM perturbation modes), sfig-ismdraws (ISM draws).
numpy + stdlib only; every input is opened through harness. Used only by fam_screen_*.py, fam_sfig_ism.py,
fam_sfig_cpg.py, fam_sfig_ismdraws.py.

The machinery every one of these figures shares:
  * cohort: the 10,353 Gschwind K562 pairs, keyed (gene_id, chrom:start-end), labelled by `significant`
    (inputs/gschwind.hg38.tsv), in the row order of the Enformer matched-protocol table (A1w; every unified-t1 table
    shares that order and those distances);
  * bootstrap unit: the connected components of the enhancer-gene graph (663), found by union-find in that order;
  * resamples: numpy.random.default_rng(20260816), then 10,000 x rng.integers(0, 663, 663), each replicate the
    concatenation of the drawn components' rows; every figure masks the SAME replicates to its own support;
  * tie-aware step-wise average precision (one threshold per distinct score).
"""
from __future__ import annotations

import csv
import json
import math
import multiprocessing as mp
import os

import numpy as np

import harness

SEED, NBOOT = 20260816, 10000
COHORT_TSV = "inputs/gschwind.hg38.tsv.gz"
BASE = "scores/unified-t1/unified-t1-A1w-enformer-single-tss.scores.tsv.gz"
SCREEN_LABELS = "paper/data/epcrispr_benchmark_K562.tsv.gz"   # the raw EPCrisprBenchmark K562 file
PERDRAW = "paper/data/ism_matched_perdraw.npz"
SCREENS = ("Gasperini2019", "Nasser2021", "Schraivogel2020")
RESOLUTION = 2e-4                # a two-sided bootstrap p below this is "< 2e-4" (10,000 replicates)


def unified(stem: str) -> str:
    return f"scores/unified-t1/unified-t1-{stem}.scores.tsv.gz"


# ------------------------------------------------------------------------------------------------------------- tables
_TAB: dict = {}


def read_table(rel: str, cols: tuple) -> dict:
    """{(gene, enhancer): [str value of each requested column]} in file order. A column a table lacks reads as None
    (callers supply the default, e.g. in_window -> "true")."""
    ck = (rel, tuple(cols))
    if ck not in _TAB:
        with harness.open_text(rel) as fh:
            rd = csv.reader(fh, delimiter="\t")
            hdr = next(rd)
            ix = {c: i for i, c in enumerate(hdr)}
            gi, ei = ix["gene"], ix["enhancer"]
            sel = [ix.get(c) for c in cols]
            _TAB[ck] = {(r[gi], r[ei]): [None if j is None else r[j] for j in sel] for r in rd}
    return _TAB[ck]


def has_column(rel: str, col: str) -> bool:
    with harness.open_text(rel) as fh:
        return col in fh.readline().rstrip("\n").split("\t")


def inwin(s) -> bool:
    """str(r.get("in_window", "true")).lower() == "true"."""
    return str("true" if s is None else s).lower() == "true"


# ------------------------------------------------------------------------------------------------------------- cohort
def components(key):
    """Union-find over the enhancer-gene graph; component ids in order of first appearance."""
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
    """The cohort: key, y, |distance|, screen label, components; columns and windows by table."""

    def __init__(self):
        lab = {}
        with harness.open_text(COHORT_TSV) as fh:
            for r in csv.DictReader(fh, delimiter="\t"):
                k = (r["gene_id"], f"{r['chrom']}:{r['enhancer_start']}-{r['enhancer_end']}")
                lab[k] = (str(r["significant"]).strip().lower() in ("true", "1", "yes"),
                          (r["chrom"], r["enhancer_start"], r["enhancer_end"], r["gene_id"].split(".")[0]))
        base = read_table(BASE, ("distance_to_tss",))
        self.key = [k for k in base if k in lab]
        self.pos = {k: i for i, k in enumerate(self.key)}
        self.y = np.array([1 if lab[k][0] else 0 for k in self.key])
        self.d = np.array([abs(int(base[k][0])) for k in self.key])
        # THE SCREEN LABEL: the raw EPCrisprBenchmark's `Dataset`, joined on
        # (chrom, chromStart, chromEnd, Ensembl id without version); the cohort file's own `screen` is constant
        sm = {}
        with harness.open_text(SCREEN_LABELS) as fh:
            for r in csv.DictReader(fh, delimiter="\t"):
                sm[(r["chrom"], r["chromStart"], r["chromEnd"], r["measuredGeneEnsemblId"].split(".")[0])] = r["Dataset"]
        self.screen = np.array([sm.get(lab[k][1], "?") for k in self.key])
        self.gene = np.array([k[0] for k in self.key])
        self.cidx, self.mem = components(self.key)

    def column(self, rel, col="score"):
        t = read_table(rel, (col,))
        return np.array([float(t[k][0]) if k in t else 0.0 for k in self.key])

    def window(self, rel):
        t = read_table(rel, ("in_window",))
        return np.array([inwin(t[k][0]) if k in t else False for k in self.key])


_COH = {}


def cohort() -> Cohort:
    if "c" not in _COH:
        _COH["c"] = Cohort()
    return _COH["c"]


_PICKS = {}


def picks(nb: int):
    """The paper's resamples: one rng.integers(0, nb, nb) per replicate from default_rng(20260816) (every figure
    of these families draws exactly these)."""
    if nb not in _PICKS:
        rng = np.random.default_rng(SEED)
        _PICKS[nb] = [rng.integers(0, nb, nb) for _ in range(NBOOT)]
    return _PICKS[nb]


# ---------------------------------------------------------------------------------------------- average precision
def ap_tie(v, yy) -> float:
    """Tie-aware step-wise AP, scores first."""
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


def value_groups(v, m):
    """Each pair of mask m -> the rank of its score among the distinct scores of m, highest first (-1 off the mask)."""
    vals = v[m]
    if np.isnan(vals).any():
        raise ValueError("NaN score: value_groups assumes none (ap_tie would rank each NaN apart)")
    u = np.unique(vals)
    g = np.full(len(v), -1, dtype=np.int64)
    g[m] = len(u) - 1 - np.searchsorted(u, vals)
    return g, len(u)


def ap_grouped(gs, ys, ng, npos) -> float:
    """ap_tie of a resample from its score groups: the same integers (positives and rows up to the end of each
    distinct score, highest first) in the same arithmetic, so the result is bit-identical to ap_tie on the resample
    (ap_tie reads a score column only through its order and ties). O(n + groups) instead of a sort per call."""
    tot = np.bincount(gs, minlength=ng)
    pos = np.bincount(gs[ys == 1], minlength=ng)
    keep = tot > 0
    cnt = np.cumsum(tot)[keep]
    tp = np.cumsum(pos)[keep]
    return float(np.sum(np.diff(np.r_[0.0, tp / npos]) * (tp / cnt)))


# ------------------------------------------------------------------------------------------------------ bootstrap
_W: dict = {}
CHUNK = 250


def nprocs() -> int:
    """Worker processes (fork) for the bootstrap: E2G_PROCS, else min(8, CPUs). 1 = serial."""
    n = os.environ.get("E2G_PROCS")
    return max(1, int(n)) if n else max(1, min(8, os.cpu_count() or 1))


def _work(span):
    a, b = span
    W = _W
    mem, pk, y = W["mem"], W["picks"], W["y"]
    out = np.full((W["ncell"], b - a), np.nan)
    for j in range(b - a):
        idx = np.concatenate([mem[t] for t in pk[a + j]])
        for m, cells in W["plan"]:
            sub = idx[m[idx]]
            if sub.size == 0:
                continue
            ys = y[sub]
            npos = ys.sum()
            if npos == 0:
                continue
            for ci, g, ng in cells:
                out[ci, j] = ap_grouped(g[sub], ys, ng, npos)
    return a, out


def bootstrap(coh: Cohort, plan):
    """plan: [(mask, [score column, ...]), ...] -> for every mask, for every column, the AP of each of the 10,000
    replicates masked to that support (NaN where the masked replicate is empty or has no positive). Every figure of
    these families masks the same replicates."""
    wplan, ci = [], 0
    for m, cols in plan:
        cells = []
        for v in cols:
            g, ng = value_groups(np.asarray(v, dtype=float), m)
            cells.append((ci, g, ng))
            ci += 1
        wplan.append((m, cells))
    _W.update(mem=coh.mem, picks=picks(len(coh.mem)), y=coh.y, plan=wplan, ncell=ci)
    spans = [(i, min(i + CHUNK, NBOOT)) for i in range(0, NBOOT, CHUNK)]
    res = np.full((ci, NBOOT), np.nan)
    try:
        n = nprocs()
        if n > 1 and "fork" not in mp.get_all_start_methods():
            n = 1                                       # the workers read the plan from this process (fork only)
        if n > 1:
            with mp.get_context("fork").Pool(n) as pool:
                for a, out in pool.imap_unordered(_work, spans):
                    res[:, a:a + out.shape[1]] = out
        else:
            for sp in spans:
                a, out = _work(sp)
                res[:, a:a + out.shape[1]] = out
    finally:
        _W.clear()
    it = iter(range(ci))
    return [[res[next(it)] for _v in cols] for _m, cols in plan]


def two_sided_p(dv) -> float:
    """min(1, 2 min(P(d <= 0), P(d >= 0))) over the replicates (every figure here)."""
    return min(1.0, 2 * min(float((dv <= 0).mean()), float((dv >= 0).mean())))


# ------------------------------------------------------------------------------------------------- per-draw ISM
_PD = {}


def perdraw() -> dict:
    """paper/data/ism_matched_perdraw.npz: for each ISM arm and statistic, the statistic of every scored pair
    evaluated on ONE stored draw at a time (strand slice applied)."""
    if "z" not in _PD:
        with harness.open_binary(PERDRAW) as fh:
            z = np.load(fh, allow_pickle=False)
            _PD["z"] = {k: z[k] for k in z.files}
    return _PD["z"]


def ism_column(coh: Cohort, arm: str, stat: str, ndraws=None):
    """(score, in_window) over the cohort for one ISM arm: the statistic over the first `ndraws` stored draws (all by
    default). Every statistic the paper uses (abs_diff, rel_abs_diff, log_abs_diff) averages |.| OVER the draws, so the
    K-draw value is the mean of the first K per-draw values (score_eg.ism_reduce_draws). Pairs without stored draws
    are out of window and score 0.0, as every table imputes."""
    z = perdraw()
    pi = z[f"{arm}__pair"]
    vals = z[f"{arm}__{stat}"]
    k = vals.shape[1] if ndraws is None else ndraws
    if k > vals.shape[1]:
        raise ValueError(f"{arm}: {k} draws requested, {vals.shape[1]} stored")
    score = vals[:, :k].mean(axis=1)
    v = np.zeros(len(coh.key))
    w = np.zeros(len(coh.key), bool)
    for i, s in zip(pi, score):
        j = coh.pos[(str(z["pair_gene"][i]), str(z["pair_enhancer"][i]))]
        v[j], w[j] = s, True
    return v, w


# ------------------------------------------------------------------------------------------------- BH (values only)
def bh_q(tests, floor=RESOLUTION, alpha=0.05):
    """Benjamini-Hochberg within one family, the paper's convention: rank by (p, name), a p
    below the resolution enters at 2e-4, q = running minimum of p m / rank from the bottom. tests: [(name, p)].
    Returns ({name: q}, k, line)."""
    P = lambda p: floor if (p is None or p < floor) else float(p)          # noqa: E731
    m = len(tests)
    order = sorted(range(m), key=lambda i: (P(tests[i][1]), tests[i][0]))
    k = 0
    for r, i in enumerate(order, 1):
        if P(tests[i][1]) <= r / m * alpha:
            k = r
    q, run = {}, 1.0
    for r in range(m, 0, -1):
        i = order[r - 1]
        run = min(run, P(tests[i][1]) * m / r)
        q[tests[i][0]] = run
    return q, k, k / m * alpha


def family_rejected(rows):
    """{name: rejected} under the paper's BH within one family (rank <= k). rows: [(name, p)]."""
    _q, k, _line = family_q(rows)
    P = lambda p: RESOLUTION if p < RESOLUTION else p                       # noqa: E731
    order = sorted(range(len(rows)), key=lambda i: (P(round(rows[i][1], 4)), rows[i][0]))
    return {rows[i][0]: r <= k for r, i in enumerate(order, 1)}


def family_q(rows):
    """q-values of one family as the paper prints them: every test enters BH with its exact bootstrap p (a multiple
    of 2e-4, so round(p, 4) is exact), the comparisons quoted in the text included.
    rows: [(Supplementary Table 2 name, p)] -> ({name: q}, k, line)."""
    return bh_q([(n, round(p, 4)) for n, p in rows])


# =====================================================================================================
# THE SCREENS FIGURE (screen-Gasperini2019, screen-Nasser2021, screen-Schraivogel2020)
# =====================================================================================================
# the screens figure (Supplementary Fig. 16), all 49 rows x 4 strata in one pass; shared by
# fam_screen_Gasperini2019 / _Nasser2021 / _Schraivogel2020.
#
# The rows, labels and supports of the figure; the bootstrap loop and the per-stratum summary (paired by replicate,
# replicates without a positive dropped, two-sided p, 2.5/97.5 percentiles); tie-aware AP for every point estimate
# and replicate; Fig. 3's gene-midpoint CAGE / PRO-cap cells read at the TSS (C1g, C2Rg, A4g, A10g, supports
# included); Borzoi at RAO=after (A2wR, D1R, F1R, NB1R); the 24 Fig. 4 input-gradient specs read at sigma 300
# (s300_columns / gene_nu below).
#
# INPUTS. Score tables (scores/unified-t1/), the 12 sigma-300 sources' manifests
# (scores/unified-t1/<stem>.manifest.json); paper/data/epcrispr_benchmark_K562.tsv.gz (the raw benchmark: the
# screen of each pair, its `Dataset` column); paper/data/ism_matched_perdraw.npz (the twelve ISM rows' per-pair
# statistics, from the stored draws).

# ---------------------------------------------------------------- arms: figure name -> repository table
#: The figure's arm names and the table each is read from (the Borzoi names without R read the RAO=after tables).
TABLE = {
    "A1w-enformer-single-tss": "A1w-enformer-single-tss",
    "C1g-enformer-gene-midpoint-genetssreadout": "C1g-enformer-gene-midpoint-genetssreadout",
    "N1-enformer-single-tss-agnostic": "N1-enformer-single-tss-agnostic",
    "A2w-borzoi-single-tss": "A2wR-borzoi-single-tss",                                 # RAO=after
    "D1-borzoi-single-tss-rnaseq": "D1R-borzoi-single-tss-rnaseq-tssnull",
    "F1-borzoi-gene-midpoint-rnaseq": "F1R-borzoi-gene-midpoint-rnaseq-tssnull",
    "C2Rg-borzoi-gene-midpoint-genetssreadout": "C2Rg-borzoi-gene-midpoint-genetssreadout",
    "NB1-borzoi-single-tss-agnostic": "NB1R-borzoi-single-tss-agnostic",
    # S2 / S3 (replicate order before) are read only for their in_window, which equals that of the sigma-300 runs
    # S2n / S3n (the windows do not depend on the kernel)
    "S2-borzoi-single-tss-logscalar": "S2n-borzoi-single-tss-logscalar-none-rafter",
    "S3-borzoi-gene-midpoint-rnaseq-tssnull-logscalar": "S3n-borzoi-gene-midpoint-rnaseq-tssnull-logscalar-none-rafter",
    # the LOGSCALAR2x2 table holds A1w / N4's cells on A1w's rows and in_window
    "LOGSCALAR2x2-enformer": "A1w-enformer-single-tss",
}
for _s in ("A3-alphagenome-single-tss", "A4g-alphagenome-gene-midpoint-genetssreadout",
           "A5-alphagenome-single-tss-rnaseq", "A6-alphagenome-gene-midpoint-rnaseq", "A8-alphagenome-single-tss-procap",
           "A10g-alphagenome-gene-midpoint-procap-genetssreadout", "A9-alphagenome-single-tss-logscalar",
           "A11-alphagenome-gene-midpoint-rnaseq-logscalar", "A12-alphagenome-single-tss-procap-logscalar",
           "NA1-alphagenome-single-tss-agnostic", "N4-enformer-single-tss-logscalar"):
    TABLE[_s] = _s

# ---------------------------------------------------------------- rows
LS = "LOGSCALAR2x2-enformer"
BN, BG = "score__g2000t3-none-ms1-rafter", "score__g2000t3-global-ms1-rafter"
AN, AG = "score__g2000t3-none-ms1", "score__g2000t3-global-ms1"
ENF_PANEL = ["A1w-enformer-single-tss", "C1g-enformer-gene-midpoint-genetssreadout"]
BOR_PANEL = ["A2w-borzoi-single-tss", "D1-borzoi-single-tss-rnaseq", "C2Rg-borzoi-gene-midpoint-genetssreadout",
             "F1-borzoi-gene-midpoint-rnaseq"]
AG_PANEL = ["A3-alphagenome-single-tss", "A5-alphagenome-single-tss-rnaseq",
            "A4g-alphagenome-gene-midpoint-genetssreadout", "A6-alphagenome-gene-midpoint-rnaseq",
            "A8-alphagenome-single-tss-procap", "A10g-alphagenome-gene-midpoint-procap-genetssreadout"]
DIST = {"enformer": 98_304, "borzoi": 262_144, "alphagenome": 524_288}

ROWS = [
    ("Fig3", ("panel", ENF_PANEL), "Enformer CAGE: anchor single-TSS vs gene-mid", ENF_PANEL[0], ENF_PANEL[1]),
    ("Fig3", ("panel", BOR_PANEL), "Borzoi CAGE: anchor single-TSS vs gene-mid",
     "A2w-borzoi-single-tss", "C2Rg-borzoi-gene-midpoint-genetssreadout"),
    ("Fig3", ("panel", AG_PANEL), "AlphaGenome CAGE: anchor single-TSS vs gene-mid",
     "A3-alphagenome-single-tss", "A4g-alphagenome-gene-midpoint-genetssreadout"),
    ("Fig3", ("panel", AG_PANEL), "AlphaGenome PRO-cap: anchor single-TSS vs gene-mid",
     "A8-alphagenome-single-tss-procap", "A10g-alphagenome-gene-midpoint-procap-genetssreadout"),
    ("Fig3", ("panel", BOR_PANEL), "Borzoi RNA-seq: anchor single-TSS vs gene-mid",
     "D1-borzoi-single-tss-rnaseq", "F1-borzoi-gene-midpoint-rnaseq"),
    ("Fig3", ("panel", AG_PANEL), "AlphaGenome RNA-seq: anchor single-TSS vs gene-mid",
     "A5-alphagenome-single-tss-rnaseq", "A6-alphagenome-gene-midpoint-rnaseq"),
    ("Fig3", ("panel", BOR_PANEL), "Borzoi: RNA-seq protocol vs CAGE protocol",
     "F1-borzoi-gene-midpoint-rnaseq", "A2w-borzoi-single-tss"),
    ("Fig3", ("panel", AG_PANEL), "AlphaGenome: RNA-seq protocol vs CAGE protocol",
     "A6-alphagenome-gene-midpoint-rnaseq", "A3-alphagenome-single-tss"),
    ("Fig3", ("panel", AG_PANEL), "AlphaGenome: PRO-cap protocol vs CAGE protocol",
     "A8-alphagenome-single-tss-procap", "A3-alphagenome-single-tss"),
    ("Fig3", ("panel", AG_PANEL), "AlphaGenome: RNA-seq protocol vs PRO-cap protocol",
     "A6-alphagenome-gene-midpoint-rnaseq", "A8-alphagenome-single-tss-procap"),
]
#: Fig 4's 2x2 per series: (label, panel, none, log-scalar, map-mean, both)
GRID = [
    ("Enformer IG CAGE", [LS], f"{LS}:dT_none", f"{LS}:dlogT_none", f"{LS}:dT_global", f"{LS}:dlogT_global"),
    ("Borzoi IG CAGE", ["A2w-borzoi-single-tss", "S2-borzoi-single-tss-logscalar"],
     f"A2w-borzoi-single-tss:{BN}", f"S2-borzoi-single-tss-logscalar:{BN}",
     f"A2w-borzoi-single-tss:{BG}", f"S2-borzoi-single-tss-logscalar:{BG}"),
    ("Borzoi IG RNA-seq", ["F1-borzoi-gene-midpoint-rnaseq", "S3-borzoi-gene-midpoint-rnaseq-tssnull-logscalar"],
     f"F1-borzoi-gene-midpoint-rnaseq:{BN}", f"S3-borzoi-gene-midpoint-rnaseq-tssnull-logscalar:{BN}",
     f"F1-borzoi-gene-midpoint-rnaseq:{BG}", f"S3-borzoi-gene-midpoint-rnaseq-tssnull-logscalar:{BG}"),
    ("AlphaGenome IG CAGE", ["A3-alphagenome-single-tss", "A9-alphagenome-single-tss-logscalar"],
     f"A3-alphagenome-single-tss:{AN}", f"A9-alphagenome-single-tss-logscalar:{AN}",
     f"A3-alphagenome-single-tss:{AG}", f"A9-alphagenome-single-tss-logscalar:{AG}"),
    ("AlphaGenome IG RNA-seq", ["A6-alphagenome-gene-midpoint-rnaseq", "A11-alphagenome-gene-midpoint-rnaseq-logscalar"],
     f"A6-alphagenome-gene-midpoint-rnaseq:{AN}", f"A11-alphagenome-gene-midpoint-rnaseq-logscalar:{AN}",
     f"A6-alphagenome-gene-midpoint-rnaseq:{AG}", f"A11-alphagenome-gene-midpoint-rnaseq-logscalar:{AG}"),
    ("AlphaGenome IG PRO-cap", ["A8-alphagenome-single-tss-procap", "A12-alphagenome-single-tss-procap-logscalar"],
     f"A8-alphagenome-single-tss-procap:{AN}", f"A12-alphagenome-single-tss-procap-logscalar:{AN}",
     f"A8-alphagenome-single-tss-procap:{AG}", f"A12-alphagenome-single-tss-procap-logscalar:{AG}"),
]
for _nm, _panel, _none, _ls, _gl, _both in GRID:
    ROWS += [("Fig4", ("panel", _panel), f"{_nm}: gene-readout divisor vs none", _ls, _none),
             ("Fig4", ("panel", _panel), f"{_nm}: map-mean divisor vs none", _gl, _none),
             ("Fig4", ("panel", _panel), f"{_nm}: both divisors vs none", _both, _none),
             ("Fig4", ("panel", _panel), f"{_nm}: gene-readout vs map-mean", _ls, _gl)]
#: ISM's divisor ladder: STAT-table prefix -> per-draw arm
ISM_TRUNKS = [("Enformer ISM CAGE", "M1ESTAT10A-", "enformer_eulerian"),
              ("Borzoi ISM CAGE", "M1BSTAT10A-", "borzoi_eulerian"),
              ("Borzoi ISM RNA-seq", "G2FSTAT-A-", "borzoi_rnaseq"),
              ("AlphaGenome ISM CAGE", "M1STAT10A-", "alphagenome_eulerian"),
              ("AlphaGenome ISM RNA-seq", "G3FSTAT-A-", "alphagenome_rnaseq"),
              ("AlphaGenome ISM PRO-cap", "G3PSTAT-A-", "alphagenome_procap")]
ISM_ARM = {px: arm for _n, px, arm in ISM_TRUNKS}
for _nm, _px, _arm in ISM_TRUNKS:
    _panel = [f"{_px}abs_diff", f"{_px}rel_abs_diff", f"{_px}log_abs_diff"]
    ROWS += [("Fig4", ("panel", _panel), f"{_nm}: gene-readout divisor vs none", f"{_px}rel_abs_diff", f"{_px}abs_diff"),
             ("Fig4", ("panel", _panel), f"{_nm}: log ratio vs none", f"{_px}log_abs_diff", f"{_px}abs_diff")]
ROWS += [
    ("Fig5", ("dist", DIST["enformer"]), "Enformer: K562 tracks vs all-cell-type tracks",
     "A1w-enformer-single-tss", "N1-enformer-single-tss-agnostic"),
    ("Fig5", ("dist", DIST["borzoi"]), "Borzoi: K562 tracks vs all-cell-type tracks",
     "A2w-borzoi-single-tss", "NB1-borzoi-single-tss-agnostic"),
    ("Fig5", ("dist", DIST["alphagenome"]), "AlphaGenome: K562 tracks vs all-cell-type tracks",
     "A3-alphagenome-single-tss", "NA1-alphagenome-single-tss-agnostic"),
]
STRATA = ("pooled",) + SCREENS

# ---------------------------------------------------------------- the 24 sigma-300 input-gradient specs
#: (arm, the spec's free-grid column name) -> (sigma-300 source, Borzoi?, 0 none / 1 global)
S300_SRC = {(LS, "dT_none"): ("A1w-enformer-single-tss", False, 0),
            (LS, "dT_global"): ("A1w-enformer-single-tss", False, 1),
            (LS, "dlogT_none"): ("N4-enformer-single-tss-logscalar", False, 0),
            (LS, "dlogT_global"): ("N4-enformer-single-tss-logscalar", False, 1)}
for _a, _src in (("A2w-borzoi-single-tss", "A2wR-borzoi-single-tss"),
                 ("S2-borzoi-single-tss-logscalar", "S2n-borzoi-single-tss-logscalar-none-rafter"),
                 ("F1-borzoi-gene-midpoint-rnaseq", "F1R-borzoi-gene-midpoint-rnaseq-tssnull"),
                 ("S3-borzoi-gene-midpoint-rnaseq-tssnull-logscalar",
                  "S3n-borzoi-gene-midpoint-rnaseq-tssnull-logscalar-none-rafter")):
    S300_SRC[(_a, BN)], S300_SRC[(_a, BG)] = (_src, True, 0), (_src, True, 1)
for _a in ("A3-alphagenome-single-tss", "A9-alphagenome-single-tss-logscalar", "A6-alphagenome-gene-midpoint-rnaseq",
           "A11-alphagenome-gene-midpoint-rnaseq-logscalar", "A8-alphagenome-single-tss-procap",
           "A12-alphagenome-single-tss-procap-logscalar"):
    S300_SRC[(_a, AN)], S300_SRC[(_a, AG)] = (_a, False, 0), (_a, False, 1)


def manifest(stem):
    """The run's config: scores/unified-t1/<stem>.manifest.json."""
    with harness.open_text(f"scores/unified-t1/unified-t1-{stem}.manifest.json") as fh:
        return json.load(fh)["config"]


def gene_nu(rel, borzoi):
    """Per-gene nu_global-ms1 (the mean |map| over the window) from its column
    (first finite in-window value), else the none/global ratio of the free-grid cells (one constant per gene)."""
    if has_column(rel, "nu_global-ms1"):
        t = read_table(rel, ("in_window", "nu_global-ms1"))
        nu = {}
        for (g, _e), (w, v) in t.items():
            if inwin(w) and v not in ("", "nan", "NaN"):
                x = float(v)
                if math.isfinite(x):
                    nu.setdefault(g, x)
        return nu
    sfx = "-rafter" if borzoi else ""
    t = read_table(rel, ("in_window", f"score__g2000t3-none-ms1{sfx}", f"score__g2000t3-global-ms1{sfx}"))
    per = {}
    for (g, _e), (w, n, gl) in t.items():
        if inwin(w):
            n, gl = float(n), float(gl)
            if math.isfinite(n) and math.isfinite(gl) and abs(gl) > 0:
                per.setdefault(g, []).append(n / gl)
    out = {}
    for g, v in per.items():
        v = sorted(v)
        med = v[len(v) // 2]
        assert (v[-1] - v[0]) <= 1e-9 * abs(med), (g, v[0], v[-1])          # one constant per gene
        out[g] = med
    return out


_S300 = {}


def s300_columns(stem, borzoi):
    """(none, global) sigma-300 values keyed by pair, from the table's
    headline `score` at its own normalisation and the per-gene constant nu; out of window both are 0."""
    if stem not in _S300:
        rel, cfg = unified(stem), manifest(stem)
        # the headline `score` is the protocol kernel (Gaussian sigma 300, truncate 4, mean-subtracted) at the run's
        # own `normalise`; the other normalisation is exact through nu
        assert cfg.get("enh_agg") == "gaussian" and cfg.get("gaussian_sigma") == 300 \
            and cfg.get("gaussian_truncate") == 4 and cfg.get("mean_subtract") is True, stem
        if borzoi:
            assert cfg.get("replicate_abs_order") == "after", stem
        norm = cfg.get("normalise")
        assert norm in ("global", "none"), (stem, norm)
        nu = gene_nu(rel, borzoi)
        none, glob = {}, {}
        for k, (s, w) in read_table(rel, ("score", "in_window")).items():
            s = float(s)
            if not inwin(w):
                assert s == 0.0, (stem, k, s)
                none[k] = glob[k] = 0.0
                continue
            v = nu[k[0]]
            none[k], glob[k] = (s * v, s) if norm == "global" else (s, s / v)
        _S300[stem] = (none, glob)
    return _S300[stem]


def column(coh, spec):
    """Score column of a figure spec ("arm" or "arm:column"): sigma-300 IG cells, ISM statistics from the stored
    draws, otherwise the arm's `score`."""
    arm, _, cn = spec.partition(":")
    if (arm, cn) in S300_SRC:
        src, brz, which = S300_SRC[(arm, cn)]
        d = s300_columns(src, brz)[which]
        return np.array([d.get(k, 0.0) for k in coh.key])
    for px, ism_arm in ISM_ARM.items():
        if arm.startswith(px):
            return ism_column(coh, ism_arm, arm[len(px):])[0]
    if cn:
        raise KeyError(spec)
    return coh.column(unified(TABLE[arm]))


def window(coh, arm):
    for px, ism_arm in ISM_ARM.items():
        if arm.startswith(px):
            return ism_column(coh, ism_arm, arm[len(px):])[1]
    return coh.window(unified(TABLE[arm]))


def support(coh, sp):
    kind, val = sp
    if kind == "dist":
        return coh.d < val
    m = np.ones(len(coh.y), bool)
    for a in val:
        m &= window(coh, a)
    return m


_RESULT = {}


def compute_all():
    """{label: {stratum: dict(n, pos, reps, a, b, delta, lo, hi, p)}} for the 49 rows, unrounded (the figure prints
    four decimals). Computed once per process."""
    if "r" in _RESULT:
        return _RESULT["r"]
    coh = cohort()
    strata = [("pooled", np.ones(len(coh.y), bool))] + [(s, coh.screen == s) for s in SCREENS]
    sups, masks, specs = {}, {}, {}
    for _f, sp, _lab, a, b in ROWS:
        key = (sp[0], sp[1] if sp[0] == "dist" else tuple(sp[1]))
        if key not in sups:
            sups[key] = support(coh, key)
        for nm, sm in strata:
            masks[(key, nm)] = sups[key] & sm
            specs.setdefault((key, nm), [])
            for s in (a, b):
                if s not in specs[(key, nm)]:
                    specs[(key, nm)].append(s)
    S = {}
    for lst in specs.values():
        for s in lst:
            if s not in S:
                S[s] = column(coh, s)
    order = list(specs)
    boots = bootstrap(coh, [(masks[k], [S[s] for s in specs[k]]) for k in order])
    AP = {(s, k): boots[i][j] for i, k in enumerate(order) for j, s in enumerate(specs[k])}
    out = {}
    for _f, sp, lab, a, b in ROWS:
        key = (sp[0], sp[1] if sp[0] == "dist" else tuple(sp[1]))
        out[lab] = {}
        for nm, _sm in strata:
            m = masks[(key, nm)]
            da, db = AP[(a, (key, nm))], AP[(b, (key, nm))]
            ok = ~(np.isnan(da) | np.isnan(db))
            dv = (da - db)[ok]
            pa, pb = ap_tie(S[a][m], coh.y[m]), ap_tie(S[b][m], coh.y[m])
            out[lab][nm] = dict(n=int(m.sum()), pos=int(coh.y[m].sum()), reps=int(ok.sum()), a=pa, b=pb,
                                delta=pa - pb, lo=float(np.percentile(dv, 2.5)), hi=float(np.percentile(dv, 97.5)),
                                p=two_sided_p(dv))
    counts = {"pairs": int(len(coh.y)), "positives": int(coh.y.sum()), "screens": {}}
    for nm, m in strata:
        counts["screens"][nm] = {"pairs": int(m.sum()), "positives": int(coh.y[m].sum()),
                                 "frac_pairs": float(m.sum()) / len(coh.y),
                                 "frac_positives": float(coh.y[m].sum()) / float(coh.y.sum())}
    _RESULT["r"] = (out, counts)
    return _RESULT["r"]


def family(screen):
    """compute() of one screen family: the 49 Supplementary Table 2 tests of that screen column."""
    out, _c = compute_all()
    return [{"id": f"screen-{screen}:{lab} [{screen}]", "delta": v[screen]["delta"], "lo": v[screen]["lo"],
             "hi": v[screen]["hi"], "p": v[screen]["p"]} for lab, v in ((r[2], out[r[2]]) for r in ROWS)]


def family_values(screen):
    """Every number the figure prints for this screen column (delta and q; the header's positives, pairs
    and shares), the column's AP point estimates and supports, its BH line, and the pooled column the figure prints
    beside it (pooled delta; whether a pooled cell is significant is decided in its main figure's family).
    Labels: '<stratum>|<row label>|n|pos|reps|a|b|delta|lo|hi|p' for stratum = pooled and this screen (pairs and
    positives on the row's support, bootstrap replicates used, auPRC of arm a and arm b, a - b, its 95% CI, two-sided
    p); '<screen>|<row label>|q' and '<screen>|family|k|line' (the paper's BH within this screen's family);
    'cohort|<stratum>|pairs|positives|frac_pairs|frac_positives' (the column header); 'single-family line 0.008|n|pooled_sig'
    and 'single-family line 0.008|<screen>|same_sign|same_sign_sig' (the same tallies at one paper-wide line, p <= 0.008).
    The per-family tallies need the pooled verdicts of other families: screen_tallies()."""
    out, counts = compute_all()
    v = {}
    q, k, line = family_q([(f"{lab} [{screen}]", out[lab][screen]["p"]) for lab in out])
    for lab, st in out.items():
        for nm in ("pooled", screen):
            for fld in ("n", "pos", "reps", "a", "b", "delta", "lo", "hi", "p"):
                v[f"{nm}|{lab}|{fld}"] = st[nm][fld]
        v[f"{screen}|{lab}|q"] = q[f"{lab} [{screen}]"]
    v[f"{screen}|family|k"] = k
    v[f"{screen}|family|line"] = line
    for nm in ("pooled", screen):
        for fld, x in counts["screens"][nm].items():
            v[f"cohort|{nm}|{fld}"] = x
    for k_, x in tallies_at(OLD_LINE).items():
        if isinstance(x, dict):
            if k_ == screen:
                for f, n in x.items():
                    v[f"single-family line {OLD_LINE}|{screen}|{f}"] = n
        else:
            v[f"single-family line {OLD_LINE}|{k_}"] = x
    return v


#: one paper-wide p line for the tallies of tallies_at()
OLD_LINE = 0.008


def _as_printed(out):
    """The rows as the figure prints them (delta and p rounded to 4 dp): both tally rules read those."""
    return {lab: {nm: {"delta": round(v[nm]["delta"], 4), "p": round(v[nm]["p"], 4)} for nm in STRATA}
            for lab, v in out.items()}


def tallies_at(line):
    """Pooled rows with p <= line, and per screen how many of them keep the pooled sign there (same_sign) and are
    also p <= line there (same_sign_sig)."""
    R = _as_printed(compute_all()[0])
    sig = [lab for lab in R if R[lab]["pooled"]["p"] <= line]
    same = lambda lab, s: (R[lab][s]["delta"] >= 0) == (R[lab]["pooled"]["delta"] >= 0)      # noqa: E731
    out = {"n": len(R), "pooled_sig": len(sig)}
    for s in SCREENS:
        out[s] = {"same_sign": sum(same(lab, s) for lab in sig),
                  "same_sign_sig": sum(same(lab, s) and R[lab][s]["p"] <= line for lab in sig)}
    return out


def pooled_test_name(fig, lab):
    """Supplementary Table 2 name of a row's POOLED test, which belongs to its main figure's family (Fig. 3, Fig. 4
    with the ISM cells of Supplementary Fig. 13, Fig. 5)."""
    if fig == "Fig3":
        if ": anchor single-TSS vs gene-mid" in lab:
            return lab.replace(": anchor single-TSS vs gene-mid", ": single-TSS vs gene-midpoint anchor")
        return {"Borzoi: RNA-seq protocol vs CAGE protocol": "Borzoi: RNA-seq@exons/gene-mid vs CAGE@TSS/single-TSS",
                "AlphaGenome: RNA-seq protocol vs CAGE protocol":
                    "AlphaGenome: RNA-seq@exons/gene-mid vs CAGE@TSS/single-TSS",
                "AlphaGenome: PRO-cap protocol vs CAGE protocol": "AlphaGenome: PRO-cap@TSS vs CAGE@TSS",
                "AlphaGenome: RNA-seq protocol vs PRO-cap protocol":
                    "AlphaGenome: RNA-seq@exons/gene-mid vs PRO-cap@TSS/single-TSS"}[lab]
    if fig == "Fig4" and " ISM " in lab:
        model, rest = lab.split(" ISM ")
        readout, stat = rest.split(": ")
        trunk = model if readout == "CAGE" else f"{model} {readout}"
        st = "relative change (/ wild type)" if stat.startswith("gene-readout") else "log ratio"
        return f"{trunk} ISM: {st} vs none (perturbation outside)"
    if fig == "Fig4":
        return {"Enformer IG CAGE: map-mean divisor vs none": "Enformer IG CAGE: gradient-map divisor vs none"}.get(
            lab, lab.replace("map-mean", "gradient-map"))
    return lab


def screen_tallies(pooled_rejected):
    """Appendix E's per-family tallies: pooled_rejected = {row label: whether its pooled test (Supplementary Table 2
    name pooled_test_name(fig, label)) is rejected in its main figure's family} -- an input,
    because those families (fig3, fig4, fig5) are computed by their own modules. Each screen cell's significance is
    BH within that screen's family (computed here)."""
    out = compute_all()[0]
    R = _as_printed(out)
    fam_sig = {s: family_rejected([(lab, out[lab][s]["p"]) for lab in out]) for s in SCREENS}
    sig_rows = [lab for lab in out if pooled_rejected[lab]]
    t = {"pooled": len(sig_rows)}
    for s in SCREENS:
        same = [lab for lab in sig_rows if (R[lab][s]["delta"] >= 0) == (R[lab]["pooled"]["delta"] >= 0)]
        t[s] = {"same_sign": len(same), "same_sign_sig": sum(fam_sig[s][lab] for lab in same),
                "sig_any": sum(fam_sig[s].values()),
                "sig_here_not_pooled": sum(fam_sig[s][lab] and not pooled_rejected[lab] for lab in out)}
    return t
