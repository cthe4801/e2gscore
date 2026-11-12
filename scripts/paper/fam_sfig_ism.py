"""fam_sfig_ism.py -- family sfig-ism: ISM under the matched protocol (Supplementary Fig. 9, Appendix D.2), 11 tests:
each model's ISM auPRC minus ABC's, on all pairs and in three distance bins (Enformer reaches no pair beyond 100 kb,
so it has no bracket there).

Every AP tie-aware; ISM statistic log_abs_diff; replicate_abs_order = after; ABC and ABC* are this repository's
baselines (scripts/baselines/score_baselines.py).
  * scores: each model's matched ISM statistic, log_abs_diff over the 10 stored Eulerian dinucleotide-shuffle draws
    with the strand slice (the M1ESTAT10A / M1BSTAT10A / M1STAT10A-log_abs_diff tables), from
    paper/data/ism_matched_perdraw.npz; ABC / ABC* from their score tables; 1/distance = 1 / max(|distance|, 1);
  * every pair of a bin is scored (unreachable pairs imputed 0); a model with no reachable pair in a bin has no bar;
  * paired component bootstrap: the 10,000 full-cohort resamples (seed 20260816) masked to the bin;
  * bracket delta = (bar height, rounded to 4 dp) minus ABC's bar height, as the figure prints it; CI = nan-percentiles
    of the replicate differences; p two-sided over the replicates where both are defined.
values(): every bar (auPRC, CI, reach), the bracket tuples, the bin headers and panel B (macro within-gene AUROC).
"""
import numpy as np

import lib_d as L

BINS = [("all pairs", 0, 10 ** 9), ("0–10 kb", 0, 10_000), ("10–100 kb", 10_000, 100_000),
        (">100 kb", 100_000, 10 ** 9)]
ORDER = ["Enformer", "Borzoi", "AlphaGenome", "ABC (Fulco)", "ABC*", "1/distance"]
TRUNKS = ["Enformer", "Borzoi", "AlphaGenome"]
REF = "ABC (Fulco)"
ISM_ARM = {"Enformer": "enformer_eulerian", "Borzoi": "borzoi_eulerian", "AlphaGenome": "alphagenome_eulerian"}
BASELINE_TABLE = {"ABC (Fulco)": "scores/baselines/abc.scores.tsv.gz", "ABC*": "scores/baselines/abcstar.scores.tsv.gz"}

_R = {}


def auroc(v, yy):
    """Tie-corrected Mann-Whitney AUC."""
    P, N = int(yy.sum()), int((yy == 0).sum())
    if P == 0 or N == 0:
        return float("nan")
    o = np.argsort(v, kind="stable")
    ss = v[o]
    r = np.empty(len(ss), float)
    i = 0
    while i < len(ss):
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]:
            j += 1
        r[i:j + 1] = (i + j) / 2.0 + 1.0
        i = j + 1
    rr = np.empty(len(ss), float)
    rr[o] = r
    return float((rr[yy == 1].sum() - P * (P + 1) / 2.0) / (P * N))


def _arms(coh):
    S, W = {}, {}
    for nm in ORDER:
        if nm in ISM_ARM:
            S[nm], W[nm] = L.ism_column(coh, ISM_ARM[nm], "log_abs_diff")
        elif nm in BASELINE_TABLE:
            S[nm] = coh.column(BASELINE_TABLE[nm])
            W[nm] = S[nm] != 0                         # a baseline "reaches" the pairs it scores nonzero
        else:
            S[nm] = 1.0 / np.maximum(coh.d, 1)
            W[nm] = np.ones(len(coh.y), bool)
    return S, W


def _run():
    if "r" in _R:
        return _R["r"]
    coh = L.cohort()
    y, d = coh.y, coh.d
    S, W = _arms(coh)
    masks = [(d >= lo) & (d < hi) for _lbl, lo, hi in BINS]
    boots = L.bootstrap(coh, [(m, [S[nm] for nm in ORDER]) for m in masks])
    bins = []
    for (lbl, _lo, _hi), m, bb in zip(BINS, masks, boots):
        rec = {"label": lbl, "n": int(m.sum()), "pos": int(y[m].sum()), "prev": 100 * y[m].mean(), "arms": {},
               "vs_ref": {}}
        B = dict(zip(ORDER, bb))
        for nm in ORDER:
            mm = m & W[nm]
            pt = L.ap_tie(S[nm][m], y[m]) if (y[m].sum() and mm.sum()) else float("nan")
            lo, hi = np.nanpercentile(B[nm], [2.5, 97.5])
            rec["arms"][nm] = {"pt": None if np.isnan(pt) else pt, "lo": float(min(lo, pt)), "hi": float(max(hi, pt)),
                               "cov": int(mm.sum()), "reach": 100 * mm.sum() / max(m.sum(), 1)}
        for nm in TRUNKS:
            if rec["arms"][nm]["pt"] is None:
                rec["vs_ref"][nm] = None
                continue
            dl = B[nm] - B[REF]
            # the bracket prints the difference of the two printed bar heights (each rounded to 4 dp)
            delta = round(rec["arms"][nm]["pt"], 4) - round(rec["arms"][REF]["pt"], 4)
            lo, hi = np.nanpercentile(dl, [2.5, 97.5])
            rec["vs_ref"][nm] = (delta, float(lo), float(hi), L.two_sided_p(dl[~np.isnan(dl)]))
        bins.append(rec)
    _R["r"] = (coh, S, bins)
    return _R["r"]


def compute():
    _coh, _S, bins = _run()
    out = []
    for rec in bins:
        for nm, v in rec["vs_ref"].items():
            if v is not None:
                out.append({"id": f"sfig-ism:{nm} vs ABC (ISM), {rec['label']}", "delta": v[0], "lo": v[1],
                            "hi": v[2], "p": v[3]})
    return out


def macro(coh, S):
    """Panel B: macro within-gene AUROC over the genes with >= 1 positive and >= 1 negative, its component bootstrap
    over the same resamples, and the mean over the genes where ABC is nonzero on every pair."""
    y, gene = coh.y, coh.gene
    genes = [g for g in np.unique(gene) if y[gene == g].sum() >= 1 and (y[gene == g] == 0).sum() >= 1]
    gi = [np.where(gene == g)[0] for g in genes]
    gcomp = np.array([coh.cidx[k[0]] for k in gi])
    bycomp = {c: np.where(gcomp == c)[0] for c in np.unique(gcomp)}
    clean = [k for k in gi if (S[REF][k] != 0).all()]
    AUC = {nm: np.array([auroc(S[nm][k], y[k]) for k in gi]) for nm in ORDER}
    CLEAN = {nm: float(np.nanmean([auroc(S[nm][k], y[k]) for k in clean])) for nm in ORDER}
    bm = {nm: [] for nm in ORDER}
    for pk in L.picks(len(coh.mem)):
        sel = [bycomp[t] for t in pk if t in bycomp]
        if not sel:
            continue
        ix = np.concatenate(sel)
        for nm in ORDER:
            bm[nm].append(float(np.nanmean(AUC[nm][ix])))
    out = {"n_genes": len(genes), "n_clean": len(clean), "arms": {}}
    for nm in ORDER:
        lo, hi = np.nanpercentile(np.array(bm[nm]), [2.5, 97.5])
        out["arms"][nm] = {"auroc": float(np.nanmean(AUC[nm])), "lo": float(lo), "hi": float(hi), "clean": CLEAN[nm]}
    return out


def values():
    """Labels: '<bin>|n|pos|prev' (pairs, positives, prevalence %); '<bin>|<arm>|pt|lo|hi' (auPRC bar and its 95% CI;
    pt None where the arm reaches no pair of the bin), '<bin>|<arm>|cov|reach' (pairs reached, % of the bin);
    '<bin>|<model> vs ABC|delta|lo|hi|p' (the brackets); 'macro|n_genes|n_clean', 'macro|<arm>|auroc|lo|hi|clean'
    (panel B); 'q|<Supplementary Table 2 id>' and 'family|k|line' (the paper's Benjamini-Hochberg within this
    family)."""
    coh, S, bins = _run()
    v = {}
    for rec in bins:
        b = rec["label"]
        for f in ("n", "pos", "prev"):
            v[f"{b}|{f}"] = rec[f]
        for nm, a in rec["arms"].items():
            for f in ("pt", "lo", "hi", "cov", "reach"):
                v[f"{b}|{nm}|{f}"] = a[f]
        for nm, t in rec["vs_ref"].items():
            if t is not None:
                for f, x in zip(("delta", "lo", "hi", "p"), t):
                    v[f"{b}|{nm} vs ABC|{f}"] = x
    tests = compute()
    q, k, line = L.family_q([(t["id"].split(":", 1)[1], t["p"]) for t in tests])
    for t in tests:
        v[f"q|{t['id']}"] = q[t["id"].split(":", 1)[1]]
    v["family|k"], v["family|line"] = k, line
    M = macro(coh, S)
    v["macro|n_genes"], v["macro|n_clean"] = M["n_genes"], M["n_clean"]
    for nm, a in M["arms"].items():
        for f in ("auroc", "lo", "hi", "clean"):
            v[f"macro|{nm}|{f}"] = a[f]
    return v
