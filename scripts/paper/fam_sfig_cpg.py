"""fam_sfig_cpg.py -- family sfig-cpg: ISM perturbation modes (Supplementary Fig. 14, Appendix D.4), 18 tests:
per model, CpG-preserving vs CpG-destroying modes (the three comparisons quoted in the text) and each of the five
other modes against the Eulerian dinucleotide shuffle (15 rows).

Computed on the ISM matched protocol: every arm scored by log_abs_diff over ALL its stored draws (10; zero-replace is
deterministic and stored at 1) with the strand slice (lib_d.ism_column), every AP tie-aware. The 15 mode rows are the
Supplementary Table 2 ids "sfig-cpg:ISM <mode> vs Eulerian dinucleotide shuffle (<model>)" (MODE_LABEL).
  * support: the pairs every one of a model's six arms reaches (each arm's scored pairs);
  * resamples: the 10,000 full-cohort component resamples (seed 20260816) masked to the support; a resample with no
    positive is dropped;
  * per mode: delta = auPRC(mode) - auPRC(Eulerian), CI and two-sided p of the paired replicate differences;
  * group contrast (excl. zero-replace): mean over CpG-preserving {genomic replace, Eulerian dinucleotide shuffle,
    2-mer block shuffle} minus mean over CpG-destroying {mononucleotide shuffle, random replace}, per replicate.
Panel A of the figure (CpG obs/exp and GC of each mode on 200 genomic windows) is a sequence measurement: it needs the
hg38 FASTA and the engine's perturb() and is NOT recomputed here.
"""
import numpy as np

import lib_d as L

TRUNKS = ["Enformer", "Borzoi", "AlphaGenome"]
#: the figure's modes, zero-replace last; the per-draw arm of each mode
MODES = [("genomic-replace", "genomic_replace"), ("eulerian-dinuc", "eulerian"), ("dinuc (block)", "dinuc_block"),
         ("mononuc", "mononuc"), ("random-replace", "random_replace"), ("zero-replace", "zero_replace")]
BASELINE, ZERO = "eulerian-dinuc", "zero-replace"
PRESERVE = {"genomic-replace", "eulerian-dinuc", "dinuc (block)"}
MODE_LABEL = {"genomic-replace": "genomic replace", "dinuc (block)": "2-mer block shuffle",
              "mononuc": "mononucleotide shuffle", "random-replace": "random replace", "zero-replace": "zero replace"}

_R = {}


def _summary(ds):
    return {"lo": float(np.percentile(ds, 2.5)), "hi": float(np.percentile(ds, 97.5)), "p": L.two_sided_p(ds)}


def _run():
    if "r" in _R:
        return _R["r"]
    coh = L.cohort()
    y = coh.y
    res = {}
    for trunk in TRUNKS:
        arms = {lbl: L.ism_column(coh, f"{trunk.lower()}_{arm}", "log_abs_diff") for lbl, arm in MODES}
        m = np.ones(len(y), bool)
        for _v, w in arms.values():
            m &= w
        (boot,) = L.bootstrap(coh, [(m, [v for v, _w in arms.values()])])
        reps = dict(zip(arms, boot))
        ok = ~np.isnan(reps[BASELINE])                 # drop the resamples with no positive in the support
        assert all(np.array_equal(~np.isnan(r), ok) for r in reps.values())
        reps = {k: r[ok] for k, r in reps.items()}
        pt = {k: L.ap_tie(v[m], y[m]) for k, (v, _w) in arms.items()}
        modes = {}
        for k in arms:
            if k == BASELINE:
                modes[k] = {"delta": 0.0, "lo": 0.0, "hi": 0.0, "p": 1.0, "auprc": pt[k]}
                continue
            modes[k] = dict(_summary(reps[k] - reps[BASELINE]), delta=pt[k] - pt[BASELINE], auprc=pt[k])
        kp = [k for k in arms if k in PRESERVE]
        de = [k for k in arms if k not in PRESERVE and k != ZERO]
        group = {}
        for tag, dd in (("excl_zero", de), ("incl_zero", de + [ZERO])):
            K = np.mean(np.stack([reps[k] for k in kp]), axis=0)
            D = np.mean(np.stack([reps[k] for k in dd]), axis=0)
            group[tag] = dict(_summary(K - D), delta=float(np.mean([pt[k] for k in kp]) - np.mean([pt[k] for k in dd])))
        res[trunk] = {"n": int(m.sum()), "pos": int(y[m].sum()), "prev": float(100 * y[m].mean()), "modes": modes,
                      "group": group}
    _R["r"] = res
    return res


def compute():
    res = _run()
    out = []
    for trunk in ("Borzoi", "AlphaGenome", "Enformer"):          # the text's order (Appendix D.4)
        g = res[trunk]["group"]["excl_zero"]
        out.append({"id": f"text:ISM CpG-preserving vs CpG-destroying ({trunk})", "delta": g["delta"], "lo": g["lo"],
                    "hi": g["hi"], "p": g["p"]})
    for trunk in TRUNKS:
        for md, lab in MODE_LABEL.items():
            v = res[trunk]["modes"][md]
            out.append({"id": f"sfig-cpg:ISM {lab} vs Eulerian dinucleotide shuffle ({trunk})", "delta": v["delta"],
                        "lo": v["lo"], "hi": v["hi"], "p": v["p"]})
    return out


def values():
    """Labels: '<model>|n|pos|prev' (the support: pairs, positives, prevalence %); '<model>|<mode>|auprc|delta|lo|hi|p'
    (each mode's auPRC and its paired difference from the Eulerian dinucleotide shuffle); '<model>|group|excl_zero|...'
    and '<model>|group|incl_zero|...' (CpG-preserving minus CpG-destroying, without / with zero replace);
    'q|<Supplementary Table 2 id>' and 'family|k|line' (the paper's BH within this family)."""
    res = _run()
    v = {}
    for trunk, r in res.items():
        for f in ("n", "pos", "prev"):
            v[f"{trunk}|{f}"] = r[f]
        for md, x in r["modes"].items():
            for f in ("auprc", "delta", "lo", "hi", "p"):
                v[f"{trunk}|{md}|{f}"] = x[f]
        for tag, x in r["group"].items():
            for f in ("delta", "lo", "hi", "p"):
                v[f"{trunk}|group|{tag}|{f}"] = x[f]
    tests = compute()
    q, k, line = L.family_q([(t["id"].split(":", 1)[1], t["p"]) for t in tests])
    for t in tests:
        v[f"q|{t['id']}"] = q[t["id"].split(":", 1)[1]]
    v["family|k"], v["family|line"] = k, line
    return v
