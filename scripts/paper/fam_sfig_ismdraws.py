"""fam_sfig_ismdraws.py -- family sfig-ismdraws: the number of ISM perturbation draws (Supplementary Fig. 15,
Appendix D.5), 3 tests: per model, auPRC at 10 draws minus auPRC at 3 draws.

Computed on the ISM matched protocol, every AP tie-aware.
  * panel: each model's reachable pairs (the in_window rows of its M?STAT10-log_abs_diff table);
  * scores: log_abs_diff over the FIRST K stored Eulerian draws (K = 1, 3, 5, 7, 10; five draws at seed 0, five at
    seed 1), strand slice, from paper/data/ism_matched_perdraw.npz (the K-draw statistic is the mean of the first K
    per-draw values);
  * resamples: the 10,000 full-cohort component resamples (seed 20260816), masked to the panel; NaN where a resample
    has no positive;
  * step a -> b: delta = (auPRC at b, rounded to 4 dp) - (auPRC at a, rounded), as the figure prints it; CI =
    percentiles of the paired replicate differences; p two-sided, floored at 2 / 10,000.
"""
import numpy as np

import lib_d as L

TRUNKS = [("Enformer", "enformer_eulerian"), ("Borzoi", "borzoi_eulerian"), ("AlphaGenome", "alphagenome_eulerian")]
PS = [1, 3, 5, 7, 10]

_R = {}


def _run():
    if "r" in _R:
        return _R["r"]
    coh = L.cohort()
    out = {}
    for trunk, arm in TRUNKS:
        cols = {K: L.ism_column(coh, arm, "log_abs_diff", ndraws=K) for K in PS}
        panel = cols[10][1]
        (boot,) = L.bootstrap(coh, [(panel, [cols[K][0] for K in PS])])
        boots = dict(zip(PS, boot))
        y = coh.y[panel]
        pan = {"n": int(panel.sum()), "pos": int(y.sum()), "prev": 100 * float(y.mean()), "components": len(coh.mem),
               "max_abs_distance": int(coh.d[panel].max()), "points": {}, "steps": {}}
        for K in PS:
            pt = L.ap_tie(cols[K][0][panel], y)
            lo, hi = np.nanpercentile(boots[K], [2.5, 97.5])
            pan["points"][K] = {"auprc": pt, "lo": float(min(lo, pt)), "hi": float(max(hi, pt))}

        def paired(a, b):
            dl = boots[b] - boots[a]
            dl = dl[~np.isnan(dl)]
            delta = round(pan["points"][b]["auprc"], 4) - round(pan["points"][a]["auprc"], 4)
            lo, hi = np.percentile(dl, [2.5, 97.5])
            return (delta, float(lo), float(hi), max(2.0 / L.NBOOT, L.two_sided_p(dl)))
        for a, b in zip(PS, PS[1:]):
            pan["steps"][f"{a}->{b}"] = paired(a, b)
        pan["steps"]["3->10"] = paired(3, 10)
        for b in PS:
            pan["steps"][f"1->{b}"] = paired(1, b) if b != 1 else (0.0, 0.0, 0.0, 1.0)
        out[trunk] = pan
    _R["r"] = out
    return out


def compute():
    res = _run()
    out = []
    for trunk, _arm in TRUNKS:
        d, lo, hi, p = res[trunk]["steps"]["3->10"]
        out.append({"id": f"text:ISM 10 vs 3 perturbation draws ({trunk})", "delta": d, "lo": lo, "hi": hi, "p": p})
    return out


def values():
    """Labels: '<model>|n|pos|prev|components|max_abs_distance' (the panel); '<model>|P<K>|auprc|lo|hi' (auPRC at K
    draws and its 95% CI, K = 1, 3, 5, 7, 10); '<model>|<a>-><b>|delta|lo|hi|p' (every step the figure object holds);
    'q|<Supplementary Table 2 id>' and 'family|k|line' (the paper's BH within this family)."""
    res = _run()
    v = {}
    for trunk, pan in res.items():
        for f in ("n", "pos", "prev", "components", "max_abs_distance"):
            v[f"{trunk}|{f}"] = pan[f]
        for K, x in pan["points"].items():
            for f in ("auprc", "lo", "hi"):
                v[f"{trunk}|P{K}|{f}"] = x[f]
        for s, x in pan["steps"].items():
            for f, val in zip(("delta", "lo", "hi", "p"), x):
                v[f"{trunk}|{s}|{f}"] = val
    tests = compute()
    q, k, line = L.family_q([(t["id"].split(":", 1)[1], t["p"]) for t in tests])
    for t in tests:
        v[f"q|{t['id']}"] = q[t["id"].split(":", 1)[1]]
    v["family|k"], v["family|line"] = k, line
    return v
