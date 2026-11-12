"""lib_c.py -- shared helpers for the Fig. 5 / Fig. 6 reproduction modules.

Every number comes from this repository's score tables through harness.open_text; nothing here reads the paper's
printed values. The estimators:

  Fig. 5 and its all-pairs twin
    each model's K562-specific, random non-K562 and all-cell-type arms (Borzoi at replicate_abs_order = after) and
    the distance floor, every AP tie-aware.
  Fig. 6
    every row on A1w's matched protocol, the pairs both runs reach, seed 20260816, B = 10,000, with the cohort,
    arms, supports and component blocks of scripts/evaluate.py (load_cohort(), load_arm(), supports(),
    gene_blocks(unit="component")) and its average_precision.
  q-values
    Benjamini-Hochberg within a family; a p below the bootstrap resolution of 2e-4 enters at 2e-4 and its q is an
    upper bound.

THE BOOTSTRAP ENGINE. Both estimators score each replicate on the CONCATENATED rows of the drawn components (a
component drawn k times contributes its rows k times). Tie-aware AP depends only on the multiset of (score, label),
so a replicate is fully described by the per-component draw counts: here every row carries its component's count
as a weight and the AP is taken from weighted group sums. The arithmetic is the row-level AP's, operation for
operation (integer-valued cumulative counts at each distinct-score group, then
sum(diff([0, tp / P]) * tp / n) over the groups present), so every replicate's AP is bit-identical to the
row-level computation's, not merely close.
"""
from __future__ import annotations

import csv
import math

import numpy as np

import harness

SEED = 20260816          # every Fig. 5 / Fig. 6 bootstrap
NBOOT = 10000
RESOLUTION = 2e-4        # two-sided bootstrap p from 10,000 paired replicates
ALPHA = 0.05

COHORT = "inputs/gschwind.hg38.tsv.gz"


def unified(stem: str) -> str:
    """Repository path of a unified-t1 score table."""
    return f"scores/unified-t1/unified-t1-{stem}.scores.tsv.gz"


_MEMO: dict = {}


def memo(key, fn):
    if key not in _MEMO:
        _MEMO[key] = fn()
    return _MEMO[key]


# ================================================================================================================
# reading
# ================================================================================================================
def read_columns(rel: str, cols) -> dict:
    """{column: [str, ...]} for the named columns of a TSV (csv.DictReader semantics). A file is parsed once per
    batch of columns not read before; ask for every column you need in one call."""
    cache = _MEMO.setdefault(("cols", rel), {})
    need = [c for c in cols if c not in cache]
    if need:
        with harness.open_text(rel) as fh:
            rd = csv.reader(fh, delimiter="\t")
            head = next(rd)
            miss = [c for c in need if c not in head]
            if miss:
                raise KeyError(f"{rel}: no column(s) {miss}")
            ix = [head.index(c) for c in need]
            got = {c: [] for c in need}
            for row in rd:
                if not row:                      # DictReader skips empty rows
                    continue
                for c, i in zip(need, ix):
                    got[c].append(row[i])
        cache.update(got)
    return {c: cache[c] for c in cols}


def labels_fig15() -> dict:
    """Fig. 5's labels: (gene_id, chrom:start-end) -> significant (bool)."""
    def go():
        c = read_columns(COHORT, ("gene_id", "chrom", "enhancer_start", "enhancer_end", "significant"))
        return {(g, f"{ch}:{a}-{b}"): str(s).strip().lower() in ("true", "1", "yes")
                for g, ch, a, b, s in zip(c["gene_id"], c["chrom"], c["enhancer_start"], c["enhancer_end"],
                                          c["significant"])}
    return memo("labels_fig15", go)


def cohort_eval() -> dict:
    """As evaluate.load_cohort: (gene_id, enhancer_id) -> (label, |distance|)."""
    def go():
        c = read_columns(COHORT, ("gene_id", "enhancer_id", "significant", "distance"))
        return {(g, e): (int(s), abs(int(d)))
                for g, e, s, d in zip(c["gene_id"], c["enhancer_id"], c["significant"], c["distance"])}
    return memo("cohort_eval", go)


def load_arms(rel: str, columns=("score",)) -> dict:
    """As evaluate.load_arm, one read for every column: {column: {(gene, enhancer): (score, in_window)}}."""
    c = read_columns(rel, ("gene", "enhancer", "in_window") + tuple(columns))
    out = {}
    for col in columns:
        out[col] = memo(("arm", rel, col), lambda col=col: {
            (g, e): (float(v), w.strip().lower() in ("true", "1"))
            for g, e, w, v in zip(c["gene"], c["enhancer"], c["in_window"], c[col])})
    return out


def load_arm(rel: str, column: str = "score") -> dict:
    return load_arms(rel, (column,))[column]


# ================================================================================================================
# average precision
# ================================================================================================================
def ap_tie(s, y) -> float:
    """Tie-aware step-wise AP, one threshold per distinct score (the arithmetic of
    evaluate.average_precision and sklearn's average_precision_score). Scores first, labels second."""
    s = np.asarray(s, dtype=float)
    y = np.asarray(y)
    P = y.sum()
    if P == 0:
        return float("nan")
    o = np.argsort(-s, kind="mergesort")
    ss, yy = s[o], y[o]
    last = np.r_[np.nonzero(np.diff(ss))[0], len(ss) - 1]
    tp = np.cumsum(yy)[last]
    return float(np.sum(np.diff(np.r_[0.0, tp / P]) * (tp / (last + 1))))


def draws(n_blocks: int, seed: int = SEED, boot: int = NBOOT) -> np.ndarray:
    """(boot x n_blocks) draw COUNTS of each block. The draws are made one call per replicate,
    [rng.integers(0, C, C) for _ in range(boot)], as Fig. 5 and evaluate.evaluate / subset_eval make them;
    a single call of size boot*C would not give the same stream."""
    def go():
        rng = np.random.default_rng(seed)
        return np.array([np.bincount(rng.integers(0, n_blocks, n_blocks), minlength=n_blocks)
                         for _ in range(boot)], dtype=float)
    return memo(("draws", n_blocks, seed, boot), go)


class Boot:
    """The replicates of one component bootstrap restricted to one support.

    comp  block index (0..C-1) of each SUPPORT row; y  its 0/1 label; counts  (B x C) draw counts.
    rule  'fig15': keep a replicate with at least one positive (Fig. 5: `idx = [i for i in idx if y[i].sum()>0]`);
          'eval' : skip an empty replicate and one whose labels are all 0 or all 1 (evaluate.evaluate,
                   subset_eval).
    """

    CELLS = 200_000      # replicates x groups per chunk: small enough to stay in cache

    def __init__(self, comp, y, counts, rule):
        self.comp = np.asarray(comp, dtype=np.int64)
        self.y = np.asarray(y, dtype=np.int64)
        C = counts.shape[1]
        rows = np.bincount(self.comp, minlength=C).astype(float)
        pos = np.bincount(self.comp, weights=self.y, minlength=C)
        tot, npos = (counts * rows).sum(1), (counts * pos).sum(1)   # exact: integer-valued floats < 2**53
        if rule == "fig15":
            ok = npos > 0
        elif rule == "eval":
            ok = (tot > 0) & (npos > 0) & (npos < tot)
        else:
            raise ValueError(rule)
        self.counts = np.ascontiguousarray(counts[ok])         # B' x C
        self.n_used = int(ok.sum())
        self.C = C

    def aps(self, s) -> np.ndarray:
        """AP of score vector `s` (over the support rows) on every kept replicate."""
        s = np.asarray(s, dtype=float)
        if np.isnan(s).any():
            raise ValueError("NaN score")
        _, g = np.unique(-s, return_inverse=True)       # group 0 = highest score; equal scores share a group
        key = g.astype(np.int64) * self.C + self.comp
        uk, inv = np.unique(key, return_inverse=True)   # super-rows: (group, block), in group order
        mult = np.bincount(inv).astype(float)           # rows of the super-row
        mpos = np.bincount(inv, weights=self.y)         # positives of the super-row
        blk = uk % self.C
        grp = uk // self.C
        starts = np.r_[0, np.nonzero(np.diff(grp))[0] + 1]    # first super-row of each group
        out = np.empty(self.n_used)
        chunk = max(8, self.CELLS // len(uk))
        for a in range(0, self.n_used, chunk):
            w = self.counts[a:a + chunk][:, blk]                # replicates x super-rows
            ng, tg = w * mult, w * mpos                         # rows / positives each super-row adds
            if len(starts) < len(uk):                           # a group spans several blocks: sum it
                ng, tg = np.add.reduceat(ng, starts, axis=1), np.add.reduceat(tg, starts, axis=1)
            cn = np.cumsum(ng, axis=1)                          # rows ranked at or above each group (`last + 1`)
            ct = np.cumsum(tg, axis=1)                          # positives ranked at or above it (`tp`)
            with np.errstate(invalid="ignore", divide="ignore"):
                rec = ct / ct[:, -1:]
                d = np.empty_like(rec)
                d[:, 0] = rec[:, 0]
                d[:, 1:] = rec[:, 1:] - rec[:, :-1]             # = diff([0, rec]) over the groups present
                term = d * (ct / cn)
            pres = ng > 0                                       # the groups this replicate contains
            for r in range(len(w)):
                out[a + r] = np.sum(term[r][pres[r]])           # summed in order over those, as the row-level AP
        return out


def boot_p(ds) -> float:
    """Two-sided bootstrap p: min(1, 2 min(P(d <= 0), P(d >= 0))) (Fig. 5's brackets; evaluate.evaluate)."""
    ds = np.asarray(ds, dtype=float)
    return float(min(1.0, 2 * min((ds <= 0).mean(), (ds >= 0).mean())))


def pct(ds, q) -> float:
    return float(np.percentile(np.asarray(ds, dtype=float), q))


# ================================================================================================================
# Fig. 5 (tie-aware)
# ================================================================================================================
TRUNKS = ("enformer", "borzoi", "alphagenome")
TRUNK_NAME = {"enformer": "Enformer", "borzoi": "Borzoi", "alphagenome": "AlphaGenome"}
#: (K562-specific arm, random non-K562 draws, all-cell-type arm). Borzoi: the `after` (R) tables,
#: replicate_abs_order = after. Ten draws each.
FIG5_ARMS = {
    "enformer": ("A1w-enformer-single-tss", [f"U{i}-enformer-single-tss-rand2-s{i}" for i in range(10)],
                 "N1-enformer-single-tss-agnostic"),
    "borzoi": ("A2wR-borzoi-single-tss", [f"UB{i}R-borzoi-single-tss-rand2-s{i}" for i in range(10)],
               "NB1R-borzoi-single-tss-agnostic"),
    "alphagenome": ("A3-alphagenome-single-tss", [f"UA{i}-alphagenome-single-tss-rand1-s{i}" for i in range(10)],
                    "NA1-alphagenome-single-tss-agnostic"),
}
#: |d| cap of each support
SUPPORT_CAP = {"all pairs": 10 ** 9, "rf-enformer": 98_304, "rf-borzoi": 262_144, "rf-alphagenome": 524_288}
DISTANCE = "scores/baselines/distance.scores.tsv.gz"
ABC = "scores/baselines/abc.scores.tsv.gz"


def fig15_base(trunk: str) -> dict:
    """The K562 arm's rows in the cohort, in table order: keys, labels, |distance_to_tss|, blocks."""
    def go():
        lab = labels_fig15()
        c = read_columns(unified(FIG5_ARMS[trunk][0]), ("gene", "enhancer", "distance_to_tss"))
        key, d = [], []
        for g, e, dist in zip(c["gene"], c["enhancer"], c["distance_to_tss"]):
            if (g, e) in lab:
                key.append((g, e))
                d.append(abs(int(dist)))
        y = np.array([1 if lab[k] else 0 for k in key])
        # blocks: union-find over (gene, enhancer) in key order; a block is numbered at its first row
        # (`ci.setdefault(find(("g", g))[1], len(ci))`)
        par: dict = {}

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
        ci: dict = {}
        comp = np.array([ci.setdefault(find(("g", g))[1], len(ci)) for g, _ in key], dtype=np.int64)
        return {"key": key, "y": y, "d": np.array(d), "comp": comp, "n_blocks": len(ci)}
    return memo(("fig15_base", trunk), go)


def fig15_scores(trunk: str, stem: str) -> np.ndarray:
    """The arm's score in the base key order (0.0 for a missing pair)."""
    def go():
        lab = labels_fig15()
        c = read_columns(unified(stem), ("gene", "enhancer", "score"))
        dd = {(g, e): float(v) for g, e, v in zip(c["gene"], c["enhancer"], c["score"]) if (g, e) in lab}
        return np.array([dd.get(k, 0.0) for k in fig15_base(trunk)["key"]])
    return memo(("fig15_scores", trunk, stem), go)


def support_mask(trunk: str, support: str) -> np.ndarray:
    B = fig15_base(trunk)
    return B["d"] < SUPPORT_CAP[support]


def fig15_panel(trunk: str, support: str) -> dict:
    """One Fig. 5 panel for one support, tie-aware AP, unrounded.

    Returns n, pos, k562 / N1 (point, 2.5th, 97.5th percentile), U (the ten draws' point APs), and the three
    bootstrap contrasts as [point, lo, hi, p]: specificity (K562 - U_mean), count (N1 - U_mean) and confounded
    (N1 - K562). The paper rounds each to 4 dp; Fig. 5 draws specificity, -count and -confounded."""
    def go():
        B = fig15_base(trunk)
        y, m = B["y"], support_mask(trunk, support)
        k562, us_stems, n1 = FIG5_ARMS[trunk]
        S = {"K": fig15_scores(trunk, k562), "N": fig15_scores(trunk, n1)}
        for j, st in enumerate(us_stems):
            S[f"U{j}"] = fig15_scores(trunk, st)
        nd = len(us_stems)
        boot = Boot(B["comp"][m], y[m], draws(B["n_blocks"]), "fig15")
        R = {k: boot.aps(v[m]) for k, v in S.items()}          # per-replicate APs
        pt = {k: ap_tie(v[m], y[m]) for k, v in S.items()}      # point APs

        def ciof(k):
            return [pt[k], pct(R[k], 2.5), pct(R[k], 97.5)]
        us = [pt[f"U{j}"] for j in range(nd)]
        um_rep = np.array([np.mean([R[f"U{j}"][r] for j in range(nd)]) for r in range(boot.n_used)])

        def gap(k):
            ds = R[k] - um_rep
            return [pt[k] - np.mean(us), pct(ds, 2.5), pct(ds, 97.5), boot_p(ds)]
        dsc = R["N"] - R["K"]
        return {"trunk": trunk, "support": support, "n": int(m.sum()), "pos": int(y[m].sum()),
                "n_boot_used": boot.n_used, "k562": ciof("K"), "N1": ciof("N"), "U": us,
                "specificity": gap("K"), "count": gap("N"),
                "confounded": [pt["N"] - pt["K"], pct(dsc, 2.5), pct(dsc, 97.5), boot_p(dsc)]}
    return memo(("fig15_panel", trunk, support), go)


def negate(v):
    """[pt, lo, hi, p] -> the opposite direction (bounds swap, p unchanged)."""
    return [-v[0], -v[2], -v[1], v[3]]


#: Fig. 5's three brackets: Supplementary Table 2 name -> how the drawn [delta, lo, hi, p] is made from a panel
FIG5_BRACKETS = (("K562 tracks vs all-cell-type tracks", lambda P: negate(P["confounded"])),
                 ("K562 tracks vs random non-K562 tracks", lambda P: list(P["specificity"])),
                 ("random non-K562 tracks vs all-cell-type tracks", lambda P: negate(P["count"])))


def distance_floor(support: str) -> float:
    """The distance floor, tie-aware: the distance-only baseline's AP on the panel's support (Enformer's
    A1w key order and |distance_to_tss|; the same vector for every trunk)."""
    def go():
        lab = labels_fig15()
        B = fig15_base("enformer")
        c = read_columns(DISTANCE, ("gene", "enhancer", "score"))
        dd = {(g, e): float(v) for g, e, v in zip(c["gene"], c["enhancer"], c["score"]) if (g, e) in lab}
        s = np.array([dd.get(k, 0.0) for k in B["key"]])
        m = B["d"] < SUPPORT_CAP[support]
        return ap_tie(s[m], B["y"][m])
    return memo(("floor", support), go)


def abc_on_support(trunk: str, support: str) -> float:
    """ABC (scores/baselines/abc.scores.tsv.gz) on a Fig. 5 panel's pairs."""
    def go():
        B = fig15_base(trunk)
        c = read_columns(ABC, ("gene", "enhancer", "score"))
        dd = {(g, e): float(v) for g, e, v in zip(c["gene"], c["enhancer"], c["score"])}
        s = np.array([dd[k] for k in B["key"]])
        m = support_mask(trunk, support)
        return ap_tie(s[m], B["y"][m])
    return memo(("abc", trunk, support), go)


def fig5_tests(family: str, rf: bool) -> list:
    """The nine bracket tests of Fig. 5 (rf supports) or of its all-pairs twin."""
    out = []
    for t in TRUNKS:
        P = fig15_panel(t, f"rf-{t}" if rf else "all pairs")
        for name, fn in FIG5_BRACKETS:
            d, lo, hi, p = fn(P)
            nm = f"{TRUNK_NAME[t]}: {name}" + ("" if rf else ", all pairs")
            out.append({"id": f"{family}:{nm}", "name": nm, "delta": float(d), "lo": float(lo), "hi": float(hi),
                        "p": float(p)})
    return out


def fig5_values(rf: bool, prefix: str) -> dict:
    """The numbers one Fig. 5-style figure prints (and the ones it draws without printing)."""
    v = {}
    for t in TRUNKS:
        sup = f"rf-{t}" if rf else "all pairs"
        P = fig15_panel(t, sup)
        b = f"{prefix} {TRUNK_NAME[t]} ({sup})"
        v[f"{b} pairs"] = float(P["n"])
        v[f"{b} positives"] = float(P["pos"])
        for arm, k in (("K562-specific", "k562"), ("all-cell-types", "N1")):
            v[f"{b} {arm} auPRC"] = P[k][0]
            v[f"{b} {arm} CI lo"] = P[k][1]
            v[f"{b} {arm} CI hi"] = P[k][2]
        for j, u in enumerate(P["U"]):
            v[f"{b} random draw {j} auPRC"] = u
        r4 = [round(u, 4) for u in P["U"]]               # the figure prints the band from the 4-dp draws
        v[f"{b} random mean (printed, of the 4-dp draws)"] = float(np.mean(r4))
        v[f"{b} random band min"] = min(P["U"])
        v[f"{b} random band max"] = max(P["U"])
        v[f"{b} distance floor auPRC"] = distance_floor(sup)
        for name, fn in FIG5_BRACKETS:
            d, lo, hi, p = fn(P)
            v[f"{b} bracket {name}: delta"] = d
    return v


# ================================================================================================================
# Fig. 6 (subset_eval: the computation of scripts/evaluate.py)
# ================================================================================================================
def gene_blocks(order: list):
    """evaluate.gene_blocks(keys, "component") on already-sorted keys: block index of each row (blocks are
    ordered by str() of their union-find root, exactly as the evaluator orders them) and the number of blocks."""
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
    blocks: dict = {}
    for i, k in enumerate(order):
        blocks.setdefault(find(("g", k[0])), []).append(i)
    ids = sorted(blocks, key=lambda b: str(b))
    comp = np.empty(len(order), dtype=np.int64)
    for j, b in enumerate(ids):
        comp[blocks[b]] = j
    return comp, len(ids)


def subset_eval(arms: dict, pairs=None) -> dict:
    """evaluate's keys, blocks, support, draws and replicate skipping for support `common-reachable`, bin
    `All`, seed 20260816, 10,000 replicates, average_precision.

    arms: name -> {(gene, enhancer): (score, in_window)}. pairs default to evaluate()'s own (sorted names, i < j,
    delta = a - b). Returns points, n, n_pos, n_boot_used and deltas [{a, b, delta, lo, hi, p}]."""
    coh = cohort_eval()
    keys = sorted(set.intersection(*(set(t) for t in arms.values())))
    keys = [k for k in keys if k in coh]
    reach = []                                    # evaluate.supports: per-arm reachability, intersected
    for name, tab in arms.items():
        rs = {k for k in keys if tab[k][1] is True}
        if rs or any(tab[k][1] is False for k in keys):
            reach.append(rs)
    common = set.intersection(*reach)
    comp, n_blocks = gene_blocks(keys)
    y_all = np.array([coh[k][0] for k in keys], dtype=np.int64)
    m = np.array([i for i, k in enumerate(keys) if k in common], dtype=np.int64)
    y = y_all[m]
    s = {n: np.array([arms[n][k][0] for k in keys], dtype=float)[m] for n in arms}
    boot = Boot(comp[m], y, draws(n_blocks), "eval")
    point = {n: ap_tie(s[n], y) for n in arms}
    reps = {n: boot.aps(s[n]) for n in arms}
    if pairs is None:
        names = sorted(arms)
        pairs = [(names[i], names[j]) for i in range(len(names)) for j in range(i + 1, len(names))]
    deltas = []
    for a, b in pairs:
        da = reps[a] - reps[b]
        deltas.append({"a": a, "b": b, "delta": point[a] - point[b], "lo": pct(da, 2.5), "hi": pct(da, 97.5),
                       "p": boot_p(da)})
    return {"points": point, "n": int(m.size), "n_pos": int(y.sum()), "n_boot_used": boot.n_used, "deltas": deltas}


def find_delta(deltas, a, b):
    """(delta, lo, hi, p) of b over a (b - a), from evaluator-shaped deltas either way round."""
    for d in deltas:
        if (d["a"], d["b"]) == (a, b):
            return -d["delta"], -d["hi"], -d["lo"], d["p"]
        if (d["a"], d["b"]) == (b, a):
            return d["delta"], d["lo"], d["hi"], d["p"]
    raise KeyError((a, b))


CTRL = "A1w-enformer-single-tss"
#: Fig. 6's Enformer point rows and augmentation rows: (figure label, arm stem); the control is A1w.
FIG6_POINT = {"B1": ("multiple-TSS averaging  (B1)", "B1-enformer-each-tss"),
              "O3": ("major TSS, model picks its anchor  (O3)", "O3-enformer-single-tss-majortss"),
              "J1": ("readout at every in-window TSS  (J1)", "J1-enformer-single-tss-alltss"),
              "R2": ("enh-TSS midpoint, readout pinned  (R2)", "R2-enformer-enhancer-tss-midpoint-genetssreadout"),
              "N2": ("1 bin (128 bp)  (N2)", "N2-enformer-single-tss-tss128"),
              "L1": ("2 bins (256 bp)  (Linder 2025)", "L1-enformer-single-tss-tss256"),
              "N3": ("5 bins (640 bp)  (N3)", "N3-enformer-single-tss-tss640"),
              "O4": ("reverse complement (rc)  (O4)", "O4-enformer-single-tss-ttarc"),
              "V1": ("shift -3/0/+3 bp  (V1)", "V1-enformer-single-tss-ttashift"),
              "Z1": ("shift -43/0/+43 bp  (Z1)", "Z1-enformer-single-tss-ttashift-43bp"),
              "S1": ("rc & shift -3/0/+3 bp  (S1)", "S1-enformer-single-tss-ttarcshift"),
              "Z2": ("rc & shift -43/0/+43 bp  (Z2)", "Z2-enformer-single-tss-ttarcshift-43bp"),
              "AV8": ("8 random rc & ≤3 bp shift  (Avsec 2021)", "AV8-enformer-single-tss-ttaavsec8")}


def fig6_pair(ctl: str, arm: str) -> dict:
    """`arm` over `ctl` (arm - ctl) on the pairs both reach. The arm names (na, nb) set evaluate()'s sorted-name
    orientation, and with it the sign of every percentile."""
    def go():
        na, nb = f"unified-t1-{ctl}.run", f"unified-t1-{arm}.run"
        r = subset_eval({na: load_arm(unified(ctl)), nb: load_arm(unified(arm))})
        d, lo, hi, p = find_delta(r["deltas"], na, nb)
        return {"delta": d, "lo": lo, "hi": hi, "p": p, "n": r["n"], "n_pos": r["n_pos"],
                "n_boot_used": r["n_boot_used"]}
    return memo(("fig6_pair", ctl, arm), go)


def fig6_row(tag: str) -> dict:
    """A Fig. 6 point / augmentation row against the matched control A1w."""
    return fig6_pair(CTRL, FIG6_POINT[tag][1])


G12 = [f"g{s}t{t}" for s in (500, 1000, 2000, 3000) for t in (1, 2, 3)]
W4 = (500, 1000, 2000, 3000)
#: the range rows' runs: each compares reductions of ONE run, on its own in_window set
FIG6_RANGE = {"E": (CTRL, ""),
              "B": ("F1R-borzoi-gene-midpoint-rnaseq-tssnull", "-rafter"),
              "A": ("A6-alphagenome-gene-midpoint-rnaseq", "")}
RANGE_LABEL = {"gc": "gradient correction", "ks": "gaussian kernel sweep, 12 cells",
               "gv": "gaussian vs average pooling"}
TRUNK_LETTER = {"E": "Enformer", "B": "Borzoi", "A": "AlphaGenome"}


def range_cells(suf: str) -> dict:
    """The cells of each range row as (hypothesis column, control column)."""
    return {"gc": [(f"score__{g}-global-ms1{suf}", f"score__{g}-global-ms0{suf}") for g in G12],
            "ks": [(f"score__{g}-global-ms1{suf}", "score") for g in G12],
            "gv": [(f"score__g{w}t2-global-ms1{suf}", f"score__s{w}-global-ms1{suf}") for w in W4]}


def fig6_range(trunk: str) -> dict:
    """One trunk: every cell of its three range rows, hyp - ctl."""
    def go():
        stem, suf = FIG6_RANGE[trunk]
        fams = range_cells(suf)
        cols = sorted({c for prs in fams.values() for pr in prs for c in pr})
        arms = load_arms(unified(stem), cols)
        pairs = [pr for prs in fams.values() for pr in prs]
        r = subset_eval(arms, pairs=pairs)
        out = {"n": r["n"], "n_pos": r["n_pos"], "n_boot_used": r["n_boot_used"], "families": {}}
        for f, prs in fams.items():
            out["families"][f] = [dict(next(x for x in r["deltas"] if (x["a"], x["b"]) == pr), hyp=pr[0], ctl=pr[1])
                                  for pr in prs]
        return out
    return memo(("fig6_range", trunk), go)


def range_tests(family: str, which) -> list:
    """Supplementary Table 2 tests of the range families."""
    out = []
    for t in ("E", "B", "A"):
        R = fig6_range(t)
        for f in which:
            for c in R["families"][f]:
                h = c["hyp"].replace("score__", "").replace("-global", "")
                ct = c["ctl"].replace("score__", "").replace("-global", "")
                nm = f"{TRUNK_LETTER[t]}: {RANGE_LABEL[f].split(',')[0]}, {h} vs {ct}"
                out.append({"id": f"{family}:{nm}", "name": nm, "delta": c["delta"], "lo": c["lo"], "hi": c["hi"],
                            "p": c["p"], "trunk": t, "block": f})
    return out


def point_tests(rows) -> list:
    """Point-row families: rows = [(Supplementary Table 2 id, (control stem, arm stem), quoted in the text?)]."""
    out = []
    for rid, (ctl, arm), text in rows:
        r = fig6_pair(ctl, arm)
        out.append({"id": rid, "name": rid.split(":", 1)[1], "delta": float(r["delta"]), "lo": float(r["lo"]),
                    "hi": float(r["hi"]), "p": float(r["p"]), "text": text, "n": r["n"], "n_pos": r["n_pos"]})
    return out


def point_values(tests: list, prefix: str = "Fig. 6") -> dict:
    """A point row's drawn tuple (delta [lo, hi]), its pairs, and the q the figure / text prints. Every row enters
    BH at its exact bootstrap p, the comparisons quoted in the text included."""
    v = {}
    for t in tests:
        b = f"{prefix} {t['name']}"
        for k in ("delta", "lo", "hi", "p"):
            v[f"{b}: {k}"] = t[k]
        v[f"{b}: pairs"] = float(t["n"])                   # the figure notes O3's 1,787 common-reachable pairs
    v.update(q_values(tests, prefix=f"{prefix} q"))
    return v


def range_values(tests: list, family: str) -> dict:
    """Each range row as Fig. 6 draws it: [min, max] of its cells' deltas, and "k/n sig" where k counts the cells
    that survive BH within the block's family."""
    tt = [dict(t) for t in tests]
    bh(tt)
    v = {}
    for t in ("E", "B", "A"):
        R = fig6_range(t)
        v[f"Fig. 6 {TRUNK_LETTER[t]} range rows: pairs"] = float(R["n"])
        v[f"Fig. 6 {TRUNK_LETTER[t]} range rows: positives"] = float(R["n_pos"])
        for f in sorted({x["block"] for x in tt}):
            cells = [x for x in tt if x["trunk"] == t and x["block"] == f]
            b = f"Fig. 6 {TRUNK_LETTER[t]} {RANGE_LABEL[f].split(',')[0]}"
            v[f"{b}: range lo"] = min(c["delta"] for c in cells)
            v[f"{b}: range hi"] = max(c["delta"] for c in cells)
            v[f"{b}: cells significant (BH within {family})"] = float(sum(c["rejected"] for c in cells))
            v[f"{b}: cells"] = float(len(cells))
    return v


# ================================================================================================================
# Benjamini-Hochberg within a family (a p below 2e-4 enters at 2e-4)
# ================================================================================================================
def bh(tests: list, floor: float = RESOLUTION) -> dict:
    """In place: q, rank, rejected, q_bound for each test of ONE family; returns m, k and the family's line."""
    def P(p):
        return floor if (p is None or p < floor) else float(p)
    m = len(tests)
    order = sorted(range(m), key=lambda i: (P(tests[i]["p"]), tests[i]["name"]))
    k = 0
    for r, i in enumerate(order, 1):
        if P(tests[i]["p"]) <= r / m * ALPHA:
            k = r
    run = 1.0
    for r in range(m, 0, -1):
        i = order[r - 1]
        run = min(run, P(tests[i]["p"]) * m / r)
        tests[i]["q"] = run
    for r, i in enumerate(order, 1):
        tests[i]["rank"] = r
        tests[i]["rejected"] = r <= k
        tests[i]["q_bound"] = bool(tests[i]["p"] is None or tests[i]["p"] < RESOLUTION - 1e-12)
    return {"m": m, "k": k, "line": k / m * ALPHA}


def q_printed(t: dict) -> float:
    """The q as Supplementary Table 2, the text and the figures print it (four decimals; an upper bound rounded up)."""
    if t["q_bound"]:
        return math.ceil(t["q"] * 1e4 - 1e-9) / 1e4
    return round(t["q"], 4)


def q_values(tests: list, prefix: str = "q") -> dict:
    """BH within the family; {'<prefix> <name>': printed q} plus '<prefix> <name> significant' (1/0)."""
    tt = [dict(t) for t in tests]
    bh(tt)
    v = {}
    for t in tt:
        v[f"{prefix} {t['name']}"] = q_printed(t)
        v[f"{prefix} {t['name']} significant"] = 1.0 if t["rejected"] else 0.0
    return v
