"""lib_b.py -- shared helpers for the families fig3, sfig-allpairs-anchor and fig4.

Numpy + stdlib only. Every file is opened through harness.open_text (repository-relative paths). Nothing here reads
Supplementary Table 2 or the paper's printed values.

THE ESTIMATOR (the paper's):
  * cohort: the 10,353 pairs of inputs/gschwind.hg38.tsv, labelled by `significant`; the row ORDER is the A1w score
    table's, except the two ABC-denominator contrasts, which follow the cohort file's order;
  * resampling unit: connected components of the enhancer-gene bipartite graph (union-find, numbered by first
    appearance in row order);
  * draws: default_rng(20260816), 10,000 replicates, one rng.integers(0, n_comp, n_comp) per replicate, the support
    mask applied to the concatenated resample, replicates with no positive inside the support dropped;
  * AP: tie-aware step-wise AP, one threshold per distinct score (ap_tie);
  * a contrast a - b: delta = AP_a - AP_b on the support; CI = np.percentile(2.5, 97.5) of the paired per-replicate
    differences; p = min(1, 2 min(mean(d <= 0), mean(d >= 0))).

SPEED, NOT A DIFFERENT ESTIMATOR. A replicate is the multiset of rows of the drawn components, i.e. every row of
component t weighted by how often t was drawn. Tie-aware AP depends on the multiset only, so each replicate's AP is
computed from per-distinct-score counts (np.bincount with those weights) instead of sorting the concatenated index
array. The counts are exact integers in float64 and the final sum runs over the same array as ap_tie's, so the
result is bit-for-bit ap_tie's (self_test() checks this against a direct ap_tie on concatenated indices).
"""
from __future__ import annotations

import json
import math
import multiprocessing as mp
import os

import numpy as np

import harness

NBOOT, SEED = 10000, 20260816
COHORT = "inputs/gschwind.hg38.tsv.gz"
UT = "scores/unified-t1/unified-t1-{}.scores.tsv.gz"
MANIFEST = "scores/unified-t1/unified-t1-{}.manifest.json"
BASE_ARM = "A1w-enformer-single-tss"
ABC = "scores/baselines/abc.scores.tsv.gz"
ABCSTAR = "scores/baselines/abcstar.scores.tsv.gz"
PROCS = max(1, min(8, int(os.environ.get("E2G_PROCS", "4"))))

_CACHE: dict = {}


# ------------------------------------------------------------------------------------------------ reading
def read_columns(rel: str, cols: tuple, key=("gene", "enhancer")) -> tuple:
    """(row keys in file order, {col: list of str}) for a TSV read through the harness. Missing `in_window` reads
    "true" (r.get("in_window", "true"))."""
    ck = (rel, tuple(cols), tuple(key))
    if ck in _CACHE:
        return _CACHE[ck]
    with harness.open_text(rel) as fh:
        head = fh.readline().rstrip("\n").split("\t")
        pos = {c: i for i, c in enumerate(head)}
        miss = [c for c in cols if c not in pos and c != "in_window"]
        if miss:
            raise KeyError(f"{rel}: no column {miss}")
        ki = [pos[k] for k in key]
        ci = [pos.get(c) for c in cols]
        keys, out = [], {c: [] for c in cols}
        for line in fh:
            f = line.rstrip("\n").split("\t")
            keys.append(tuple(f[i] for i in ki))
            for c, i in zip(cols, ci):
                out[c].append("true" if i is None else f[i])
    _CACHE[ck] = (keys, out)
    return keys, out


def read_json(rel: str) -> dict:
    if ("json", rel) not in _CACHE:
        with harness.open_text(rel) as fh:
            _CACHE[("json", rel)] = json.load(fh)
    return _CACHE[("json", rel)]


def is_true(s: str) -> bool:
    return str(s).lower() == "true"


# ------------------------------------------------------------------------------------------------ cohort
def labels() -> dict:
    """(gene_id, chrom:start-end) -> significant, and the cohort file's key order."""
    if "labels" not in _CACHE:
        keys, c = read_columns(COHORT, ("gene_id", "chrom", "enhancer_start", "enhancer_end", "significant"),
                               key=("gene_id",))
        lab, order = {}, []
        for g, ch, s, e, sig in zip(c["gene_id"], c["chrom"], c["enhancer_start"], c["enhancer_end"],
                                    c["significant"]):
            k = (g, f"{ch}:{s}-{e}")
            lab[k] = str(sig).strip().lower() in ("true", "1", "yes")
            order.append(k)
        _CACHE["labels"] = (lab, order)
    return _CACHE["labels"]


def components(keys: list) -> np.ndarray:
    """Component id per row: connected components of the gene-enhancer graph, numbered by first appearance."""
    par = {}

    def find(x):
        par.setdefault(x, x)
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x
    for g, e in keys:
        p, q = find(("g", g)), find(("e", e))
        if p != q:
            par[p] = q
    ci = {}
    return np.array([ci.setdefault(find(("g", g))[1], len(ci)) for g, _ in keys])


class Cohort:
    """Row order, labels, components and the 10,000 draws for one row order ("a1w" or "cohort")."""

    def __init__(self, order: str = "a1w"):
        lab, corder = labels()
        if order == "a1w":
            k, _ = read_columns(UT.format(BASE_ARM), ())
            self.key = [x for x in k if x in lab]
        else:
            self.key = list(corder)
        self.pos = {k: i for i, k in enumerate(self.key)}
        self.n = len(self.key)
        self.y = np.array([1 if lab[k] else 0 for k in self.key])
        self.comp = components(self.key)
        self.nb = int(self.comp.max()) + 1
        rng = np.random.default_rng(SEED)
        self.draws = np.array([rng.integers(0, self.nb, self.nb) for _ in range(NBOOT)])

    def mem(self, t: int) -> np.ndarray:
        return np.where(self.comp == t)[0]

    def column(self, rel: str, col: str = "score", key=("gene", "enhancer")) -> tuple:
        """(scores, in_window) aligned to this cohort; a key the table lacks is (0.0, False)."""
        keys, c = read_columns(rel, (col, "in_window"), key=key)
        s, w = np.zeros(self.n), np.zeros(self.n, bool)
        for k, v, iw in zip(keys, c[col], c["in_window"]):
            i = self.pos.get(k)
            if i is not None:
                s[i] = float(v)
                w[i] = is_true(iw)
        return s, w

    def arm(self, slug: str, col: str = "score") -> tuple:
        return self.column(UT.format(slug), col)


_COH: dict = {}


def cohort(order: str = "a1w") -> Cohort:
    if order not in _COH:
        _COH[order] = Cohort(order)
    return _COH[order]


# ------------------------------------------------------------------------------------------------ AP
def ap_tie(v, yy) -> float:
    """Tie-aware step-wise AP with one threshold per distinct score."""
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


def levels(s: np.ndarray) -> tuple:
    """Dense rank of each score, 0 = highest; equal floats share a level (as ap_tie's np.diff(s) != 0)."""
    u, inv = np.unique(-s, return_inverse=True)
    return inv, len(u)


def ap_counts(inv, nlev, w, wy) -> float:
    """ap_tie of the multiset in which row i appears w[i] times (exactly ap_tie's arithmetic)."""
    tot = np.bincount(inv, weights=w, minlength=nlev)
    pos = np.bincount(inv, weights=wy, minlength=nlev)
    nz = tot > 0
    C = np.cumsum(tot)[nz]
    TP = np.cumsum(pos)[nz]
    P = TP[-1]
    if P == 0:
        return float("nan")
    return float(np.sum(np.diff(np.r_[0.0, TP / P]) * (TP / C)))


# ------------------------------------------------------------------------------------------------ bootstrap
_G: dict = {}


def _work(chunk):
    coh, prep = _G["coh"], _G["prep"]
    out = {}
    for sup, (R, compR, yR, cols) in prep.items():
        A = np.full((len(chunk), len(cols)), np.nan)
        for j, b in enumerate(chunk):
            cnt = np.bincount(coh.draws[b], minlength=coh.nb)
            w = cnt[compR].astype(float)
            wy = w * yR
            if wy.sum() == 0:
                continue
            for c, (inv, nlev) in enumerate(cols):
                A[j, c] = ap_counts(inv, nlev, w, wy)
        out[sup] = A
    return chunk, out


def bootstrap(coh: Cohort, plan: dict, procs: int = None) -> dict:
    """plan: {support: (mask, {name: scores})} -> {support: {"n", "pos", "pt": {name: AP},
    "reps": {name: array(NBOOT), NaN where the replicate has no positive in the support}}}.
    Every support and column is scored on the same 10,000 draws (paired by replicate index)."""
    procs = PROCS if procs is None else procs
    prep, names = {}, {}
    for sup, (m, cols) in plan.items():
        R = np.flatnonzero(m)
        names[sup] = list(cols)
        for c in cols:
            assert np.isfinite(np.asarray(cols[c], float)[R]).all(), (sup, c)    # ap_tie's sort assumes no NaN
        prep[sup] = (R, coh.comp[R], coh.y[R].astype(float), [levels(np.asarray(cols[c], float)[R]) for c in cols])
    _G.update(coh=coh, prep=prep)
    chunks = [list(range(i, min(i + 250, NBOOT))) for i in range(0, NBOOT, 250)]
    reps = {sup: np.full((NBOOT, len(names[sup])), np.nan) for sup in plan}
    if procs > 1:
        with mp.get_context("fork").Pool(procs) as pool:
            for chunk, out in pool.imap_unordered(_work, chunks):
                for sup, A in out.items():
                    reps[sup][chunk] = A
    else:
        for ch in chunks:
            chunk, out = _work(ch)
            for sup, A in out.items():
                reps[sup][chunk] = A
    res = {}
    for sup, (m, cols) in plan.items():
        y = coh.y[m]
        res[sup] = {"n": int(m.sum()), "pos": int(y.sum()),
                    "pt": {c: ap_tie(np.asarray(cols[c], float)[m], y) for c in cols},
                    "reps": {c: reps[sup][:, i] for i, c in enumerate(names[sup])}}
    return res


def contrast(B: dict, sup: str, a: str, b: str) -> dict:
    """a - b on support `sup`."""
    S = B[sup]
    da, db = S["reps"][a], S["reps"][b]
    ok = ~(np.isnan(da) | np.isnan(db))
    dv = (da - db)[ok]
    p = min(1.0, 2 * min(float((dv <= 0).mean()), float((dv >= 0).mean())))
    return {"delta": S["pt"][a] - S["pt"][b], "lo": float(np.percentile(dv, 2.5)),
            "hi": float(np.percentile(dv, 97.5)), "p": p, "reps": int(ok.sum()),
            "a": S["pt"][a], "b": S["pt"][b], "n": S["n"], "pos": S["pos"]}


def interval(B: dict, sup: str, a: str) -> dict:
    """AP and its percentile interval (Fig. 4's bars; Section 4.4's marginal auPRCs)."""
    S = B[sup]
    ds = S["reps"][a]
    ds = ds[~np.isnan(ds)]
    return {"v": S["pt"][a], "lo": float(np.percentile(ds, 2.5)), "hi": float(np.percentile(ds, 97.5))}


def flip(t: dict) -> dict:
    """The same contrast quoted the other way round (b - a)."""
    return dict(t, delta=-t["delta"], lo=-t["hi"], hi=-t["lo"])


def row(test_id: str, t: dict) -> dict:
    return {"id": test_id, "delta": float(t["delta"]), "lo": float(t["lo"]), "hi": float(t["hi"]),
            "p": float(t["p"])}


RESOLUTION = 2e-4            # two-sided p from 10,000 paired replicates


def p_entered(p: float, text: bool) -> float:
    """The p a test enters Benjamini-Hochberg with: its exact bootstrap p (a multiple of 2e-4; the paper's values
    store it to 4 decimals), for a comparison quoted in the text as for a figure's (`text` does not change it)."""
    return round(p, 4)


def bh_q(rows: list) -> dict:
    """{id: (q, q_bound)}: Benjamini-Hochberg within the family, the paper's procedure (floor = the resolution): a p
    below 2e-4 enters at 2e-4 and its q is an upper bound, printed '<' and rounded up. Tied p share one q, so
    breaking ties by name cannot change any q."""
    m = len(rows)
    P = [max(p_entered(r["p"], r["id"].startswith("text:")), RESOLUTION) for r in rows]
    order = sorted(range(m), key=lambda i: (P[i], rows[i]["id"].split(":", 1)[1]))
    q, run = {}, 1.0
    for r in range(m, 0, -1):
        i = order[r - 1]
        run = min(run, P[i] * m / r)
        q[rows[i]["id"]] = (run, p_entered(rows[i]["p"], rows[i]["id"].startswith("text:")) < RESOLUTION - 1e-12)
    return q


def q_values(rows: list) -> dict:
    """values() entries for the q (and whether it is an upper bound) every test of a family prints."""
    v = {}
    for tid, (q, bound) in bh_q(rows).items():
        v[f"q/{tid}"], v[f"q_bound/{tid}"] = q, float(bound)
    return v


# ------------------------------------------------------------------------------------------------ sigma 300
def s300_columns(coh: Cohort, slug: str, borzoi: bool) -> tuple:
    """(none, global, in_window) at the protocol kernel (Gaussian sigma 300 / truncate 4), aligned to `coh`.

    The columns Fig. 4 and Section 4.4 read. The arm's headline `score` is the protocol kernel at the run's own
    `normalise`; the other normalisation is exact through the per-gene constant nu = mean |map| over the window
    (nu_global-ms1, or where a table lacks it the per-gene ratio of its free-grid g2000t3 none/global cells):
    none = score * nu for a `global` run, global = score / nu for a `none` run. Out-of-window pairs are 0.0 in
    both."""
    ck = ("s300", slug)
    if ck in _CACHE:
        return _CACHE[ck]
    cfg = read_json(MANIFEST.format(slug))["config"]
    assert cfg.get("enh_agg") == "gaussian" and cfg.get("gaussian_sigma") == 300 \
        and cfg.get("gaussian_truncate") == 4 and cfg.get("mean_subtract") is True, slug
    if borzoi:
        assert cfg.get("replicate_abs_order") == "after", slug
    norm = cfg.get("normalise")
    assert norm in ("global", "none"), (slug, norm)
    sfx = "-rafter" if borzoi else ""
    rel = UT.format(slug)
    with harness.open_text(rel) as fh:
        has_nu = "nu_global-ms1" in fh.readline().rstrip("\n").split("\t")
    cols = ("score", "in_window") + (("nu_global-ms1",) if has_nu else
                                     (f"score__g2000t3-none-ms1{sfx}", f"score__g2000t3-global-ms1{sfx}"))
    keys, c = read_columns(rel, cols)
    inw = [is_true(v) for v in c["in_window"]]
    nu = {}
    if has_nu:
        for (g, _e), iw, v in zip(keys, inw, c["nu_global-ms1"]):
            if iw and v not in ("", "nan", "NaN"):
                x = float(v)
                if math.isfinite(x):
                    nu.setdefault(g, x)
    else:
        per = {}
        for (g, _e), iw, n_, g_ in zip(keys, inw, c[cols[2]], c[cols[3]]):
            if iw:
                n_, g_ = float(n_), float(g_)
                if math.isfinite(n_) and math.isfinite(g_) and abs(g_) > 0:
                    per.setdefault(g, []).append(n_ / g_)
        for g, v in per.items():
            v = sorted(v)
            med = v[len(v) // 2]
            assert (v[-1] - v[0]) <= 1e-9 * abs(med), (slug, g)          # one constant per gene
            nu[g] = med
    none, glob, win = np.zeros(coh.n), np.zeros(coh.n), np.zeros(coh.n, bool)
    for k, iw, sc in zip(keys, inw, c["score"]):
        i = coh.pos.get(k)
        s = float(sc)
        if not iw:
            assert s == 0.0, (slug, k, s)
            continue
        v = nu[k[0]]
        a, b = (s * v, s) if norm == "global" else (s, s / v)
        if i is not None:
            none[i], glob[i], win[i] = a, b, True
    _CACHE[ck] = (none, glob, win)
    return _CACHE[ck]


# ------------------------------------------------------------------------------------------------ Fig. 3
#: Fig. 3's panels as the paper draws them: Borzoi at replicate_abs_order = after (A2wR, D1R, F1R, C2Rg), and the
#: gene-midpoint CAGE / PRO-cap cells read at the TSS (C1g, C2Rg, A4g, A10g).
#: cell = (anchor row, readout column).
FIG3 = [
    ("Enformer", {(0, 0): "A1w-enformer-single-tss",
                  (1, 0): "C1g-enformer-gene-midpoint-genetssreadout"}),
    ("Borzoi", {(0, 0): "A2wR-borzoi-single-tss",
                (0, 1): "D1R-borzoi-single-tss-rnaseq-tssnull",
                (1, 0): "C2Rg-borzoi-gene-midpoint-genetssreadout",
                (1, 1): "F1R-borzoi-gene-midpoint-rnaseq-tssnull"}),
    ("AlphaGenome", {(0, 0): "A3-alphagenome-single-tss",
                     (0, 1): "A5-alphagenome-single-tss-rnaseq",
                     (1, 0): "A4g-alphagenome-gene-midpoint-genetssreadout",
                     (1, 1): "A6-alphagenome-gene-midpoint-rnaseq",
                     (0, 2): "A8-alphagenome-single-tss-procap",
                     (1, 2): "A10g-alphagenome-gene-midpoint-procap-genetssreadout"}),
]
READOUTS = ["CAGE@TSS", "RNA-seq@exons", "PRO-cap@TSS"]
ANCHORS = ["single-TSS", "gene-midpoint"]


def fig3_panels(all_pairs: bool, extra: dict = None) -> tuple:
    """Fig. 3's panels (FIG3), or with all_pairs its all-pairs twin (Supplementary Fig. 10).

    Each panel is scored on the pairs in window for EVERY arm of that panel (`common`), or on all 10,353 pairs in
    the all-pairs twin; moves in the figure's order (anchor per readout, readout per anchor, CAGE -> PRO-cap per
    anchor), the corner-to-corner total when the four corners exist, and in all-pairs mode each move's delta on the
    common support (`delta_common`, point estimate only, as the figure prints it). `extra` adds supports to the
    same bootstrap pass (same draws). Returns (panels, bootstrap result)."""
    coh = cohort("a1w")
    plan, sup = {}, {}
    for name, arms in FIG3:
        S, W = {}, {}
        for cell, slug in arms.items():
            S[cell], W[cell] = coh.arm(slug)
        common = np.ones(coh.n, bool)
        for c in W:
            common &= W[c]
        m = np.ones(coh.n, bool) if all_pairs else common
        plan[name] = (m, S)
        sup[name] = (common, S, W)
    plan.update(extra or {})
    B = bootstrap(coh, plan)
    out = []
    for name, arms in FIG3:
        common, S, W = sup[name]
        moves = []

        def move(src, dst, kind, label):
            if src not in arms or dst not in arms:
                return
            t = contrast(B, name, dst, src)
            t.update(kind=kind, label=label, src=list(src), dst=list(dst))
            if all_pairs:
                t["delta_common"] = ap_tie(S[dst][common], coh.y[common]) - ap_tie(S[src][common], coh.y[common])
                t["n_common"] = int(common.sum())
            moves.append(t)
        for r in range(3):
            move((0, r), (1, r), "anchor", READOUTS[r])
        for a in range(2):
            move((a, 0), (a, 1), "readout", ANCHORS[a])
        for a in range(2):
            move((a, 0), (a, 2), "readout-alt", READOUTS[2])
        total = None
        if all(c in arms for c in ((0, 0), (0, 1), (1, 0), (1, 1))):
            total = contrast(B, name, (1, 1), (0, 0))
        P = B[name]
        out.append({"model": name, "n": P["n"], "pos": P["pos"], "n_common": int(common.sum()),
                    "prev": 100.0 * P["pos"] / P["n"],
                    "reach": {c: 100.0 * W[c].mean() for c in arms},
                    "cells": dict(P["pt"]), "moves": moves, "total": total})
    return out, B


def fig3_move_name(model: str, mv: dict) -> str:
    """The Supplementary Table 2 name of a drawn move."""
    src, dst = mv["src"], mv["dst"]
    if mv["kind"] == "anchor":
        return f"{model} {READOUTS[src[1]]}: gene-midpoint vs single-TSS anchor"
    return f"{model}: {READOUTS[dst[1]]} vs {READOUTS[src[1]]} readout, {ANCHORS[src[0]]} anchor"


def cell_label(cell) -> str:
    """'single-TSS anchor, RNA-seq@exons auPRC' for cell (0, 1). Labels of values() are stable:
    paper/expected/values.json is keyed by them."""
    return f"{ANCHORS[cell[0]]} anchor, {READOUTS[cell[1]]} auPRC"


def move_label(kind: str, src, dst) -> str:
    if kind == "anchor":
        return f"anchor move at {READOUTS[src[1]]}, gene-midpoint minus single-TSS"
    return f"readout move at the {ANCHORS[src[0]]} anchor, {READOUTS[dst[1]]} minus {READOUTS[src[1]]}"


TOTAL_LABEL = "protocol move, gene-midpoint RNA-seq@exons minus single-TSS CAGE@TSS"
#: what a move's fields are called in values(): the figure's delta, its 95% CI and the bootstrap p, and in the
#: all-pairs twin the matched-support delta printed under the arrow
FIELDS = {"delta": "delta", "lo": "lo", "hi": "hi", "p": "p", "delta_common": "delta on the matched support"}


def fig3_values(panels: list, prefix: str) -> dict:
    """What the figure prints, per panel: n, positives, each cell's auPRC and each move's delta / CI / p (the
    figure prints q; Supplementary Table 2 holds p). All-pairs: the matched-support delta under each move."""
    v = {}
    for p in panels:
        base = f"{prefix}/{p['model']}"
        v[f"{base}/pairs"], v[f"{base}/positives"] = p["n"], p["pos"]
        for c, ap in p["cells"].items():
            v[f"{base}/{cell_label(c)}"] = ap
        for mv in p["moves"]:
            k = f"{base}/{move_label(mv['kind'], mv['src'], mv['dst'])}"
            for f in ("delta", "lo", "hi", "p") + (("delta_common",) if "delta_common" in mv else ()):
                v[f"{k}/{FIELDS[f]}"] = mv[f]
        if p["total"]:
            for f in ("delta", "lo", "hi", "p"):
                v[f"{base}/{TOTAL_LABEL}/{f}"] = p["total"][f]
    return v


# ------------------------------------------------------------------------------------------------ self-test
def self_test(nrep: int = 25) -> None:
    """ap_counts on component weights == ap_tie on the concatenated, masked index array (bit for bit)."""
    coh = cohort("a1w")
    s, w = coh.arm("A3-alphagenome-single-tss")
    m = w
    inv, nlev = levels(s[m])
    R = np.flatnonzero(m)
    mem = [coh.mem(t) for t in range(coh.nb)]
    keep = np.zeros(coh.n, bool)
    keep[m] = True
    for b in range(nrep):
        idx = np.concatenate([mem[t] for t in coh.draws[b]])
        idx = idx[keep[idx]]
        ref = ap_tie(s[idx], coh.y[idx])
        cnt = np.bincount(coh.draws[b], minlength=coh.nb)
        ww = cnt[coh.comp[R]].astype(float)
        got = ap_counts(inv, nlev, ww, ww * coh.y[R])
        assert got == ref, (b, got, ref)
    print(f"lib_b self-test: {nrep} replicates bit-identical to ap_tie on concatenated indices")


if __name__ == "__main__":
    self_test()
