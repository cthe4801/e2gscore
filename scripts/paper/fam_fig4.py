"""fam_fig4.py -- Supplementary Table 2 family "fig4" (Fig. 4, Section 4.4, the ISM divisor figure): 56 tests.

    python3 scripts/reproduce_paper.py fig4

Fig. 4 asks whether cross-gene normalization of input gradients changes the pooled ranking. Every number uses the
paper's estimator (lib_b: tie-aware AP; 10,000 component draws, seed 20260816; the A1w row order; one draw set
shared by every contrast). The 56 tests:

  A. 18 input-gradient cells + 6 gene-readout vs gradient-map contrasts (6 series x {gene readout, gradient map,
     both} vs none, and gene readout vs gradient map). Three of the 18 are quoted in Section 4.4 (Enformer gene
     readout, Enformer gradient map, Borzoi CAGE gene readout); one, Enformer "both", has scores identical to its
     gradient-map cell. Each series is scored on its own support: the pairs in window for both arms of the series
     (Enformer: A1w's window, 1,796 pairs; Borzoi CAGE 3,988; Borzoi RNA-seq 3,973; AlphaGenome 6,677).
     KERNEL: SIGMA 300, NOT g2000t3. Every cell uses the protocol kernel (Gaussian sigma 300 / truncate 4): each
     arm's headline `score` at its own `normalise`, the other normalisation through the per-gene constant nu
     (lib_b.s300_columns). Reading the free-grid columns score__g2000t3-{none,global}-ms1[-rafter] instead gives
     e.g. AlphaGenome CAGE /map-mean 0.7857 where the figure prints 0.7796.
     gene readout = the run with log_scalar: true (N4, S2n, S3n, A9, A11, A12) at normalise none; gradient map =
     the log_scalar: false run (A1w, A2wR, F1R, A3, A6, A8) at normalise global; both = the log_scalar run at
     global; none = the log_scalar: false run at none.
  B. 24 ISM cells: 6 ISM arms x {relative change (/ wild type), log ratio} vs none x perturbation-mean inside /
     outside the absolute value, each on the arm's own in-window pairs, from the six ISM statistic tables per
     arm (strand_mask auto, Eulerian dinucleotide shuffle, 10 draws).
  C. Section 4.4's other comparisons (against our ABC): Borzoi CAGE vs Enformer CAGE unnormalized
     and Enformer (gene readout) vs ABC on rf-Enformer pairs (A1w's window); AlphaGenome CAGE unnormalized and
     normalized vs ABC on rf-AlphaGenome pairs (A3's window).
  D. ABC and ABC* without their competing-element denominators vs with (all pairs; the cohort file's row order).
  E. AlphaGenome CAGE and RNA-seq with gene-readout normalization against the reproduced published protocol
     (Section 4.4): all 10,353 pairs, out-of-window pairs at 0, the same draws. The reproduction is the Avsec et
     al. (2026) gradient run (gene-midpoint, RNA-seq@exons, log_scalar true, normalise global, GENCODE v46).

Test names follow Supplementary Table 2, where the "map-mean" divisor is printed as "gradient-map".
"""
from __future__ import annotations

import numpy as np

import lib_b as L

#: (series, [(arm slug, Borzoi?) for the log_scalar: false run, the log_scalar: true run])
SERIES = [
    ("Enformer IG CAGE", ("A1w-enformer-single-tss", False), ("N4-enformer-single-tss-logscalar", False)),
    ("Borzoi IG CAGE", ("A2wR-borzoi-single-tss", True), ("S2n-borzoi-single-tss-logscalar-none-rafter", True)),
    ("Borzoi IG RNA-seq", ("F1R-borzoi-gene-midpoint-rnaseq-tssnull", True),
     ("S3n-borzoi-gene-midpoint-rnaseq-tssnull-logscalar-none-rafter", True)),
    ("AlphaGenome IG CAGE", ("A3-alphagenome-single-tss", False), ("A9-alphagenome-single-tss-logscalar", False)),
    ("AlphaGenome IG RNA-seq", ("A6-alphagenome-gene-midpoint-rnaseq", False),
     ("A11-alphagenome-gene-midpoint-rnaseq-logscalar", False)),
    ("AlphaGenome IG PRO-cap", ("A8-alphagenome-single-tss-procap", False),
     ("A12-alphagenome-single-tss-procap-logscalar", False)),
]
#: Fig. 4's bar names for values()
FIG4_BAR = {"Enformer IG CAGE": "Enformer · Input Gradients — single-TSS · CAGE",
            "Borzoi IG CAGE": "Borzoi · Input Gradients — single-TSS · CAGE",
            "Borzoi IG RNA-seq": "Borzoi · Input Gradients — gene-mid · RNA-seq@exons",
            "AlphaGenome IG CAGE": "AlphaGenome · Input Gradients — single-TSS · CAGE",
            "AlphaGenome IG RNA-seq": "AlphaGenome · Input Gradients — gene-mid · RNA-seq@exons",
            "AlphaGenome IG PRO-cap": "AlphaGenome · Input Gradients — single-TSS · PRO-cap@TSS"}
STEPS = ("none", "gene readout", "map-mean", "both")            # Fig. 4's bar order
#: the three Fig. 4 cells Section 4.4 quotes
TEXT_CELLS = {"Enformer IG CAGE: gene-readout divisor vs none": "Enformer IG CAGE: gene-readout divisor vs none",
              "Enformer IG CAGE: map-mean divisor vs none": "Enformer IG CAGE: gradient-map divisor vs none",
              "Borzoi IG CAGE: gene-readout divisor vs none": "Borzoi IG CAGE: gene-readout divisor vs none"}
#: ISM arms and their statistic-table prefixes
ISM = [("Enformer", "M1ESTAT10A-"), ("Borzoi", "M1BSTAT10A-"), ("AlphaGenome", "M1STAT10A-"),
       ("AlphaGenome RNA-seq", "G3FSTAT-A-"), ("Borzoi RNA-seq", "G2FSTAT-A-"), ("AlphaGenome PRO-cap", "G3PSTAT-A-")]
#: Supplementary Fig. 13's cells: row = calibration (none, / wild type, log ratio), column = P inside / outside |.|
ISM_CELLS = {(0, 0): "abs_mean_diff", (0, 1): "abs_diff", (1, 0): "abs_mean_rel_diff", (1, 1): "rel_abs_diff",
             (2, 0): "log_mean_diff", (2, 1): "log_abs_diff"}
#: our reproduction of AlphaGenome's published protocol (Supplementary Fig. on Avsec et al. 2026)
REPRODUCTION = "scores/avsec2026/avsec2026-alphagenome-gradient-gschwind.scores.tsv.gz"
_RUN: dict = {}


def _run() -> dict:
    if _RUN:
        return _RUN
    coh = L.cohort("a1w")
    plan, cols = {}, {}
    # ---- A: the input-gradient series at sigma 300, each on its own support
    for nm, (a0, b0), (a1, b1) in SERIES:
        n0, g0, w0 = L.s300_columns(coh, a0, b0)
        n1, g1, w1 = L.s300_columns(coh, a1, b1)
        # Enformer's series is scored on A1w's window; N4's window is identical. The others intersect the two arms'
        # windows.
        m = w0.copy() if nm.startswith("Enformer") else (w0 & w1)
        plan[nm] = (m, {"none": n0, "gene readout": n1, "map-mean": g0, "both": g1})
        cols[nm] = (n0, g0, n1, g1)
    # ---- B: the ISM statistic tables, each arm on its own in-window pairs
    for tr, px in ISM:
        S, own = {}, np.ones(coh.n, bool)
        for cell, st in ISM_CELLS.items():
            S[cell], w = coh.arm(px + st)
            own &= w
        plan["ISM " + tr] = (own, S)
    # ---- C: Section 4.4's tuples on the receptive-field supports (the rf-* masks)
    abc, _ = coh.column(L.ABC)
    a1w_none = cols["Enformer IG CAGE"][0]
    a2wr_none = cols["Borzoi IG CAGE"][0]
    n4_none = cols["Enformer IG CAGE"][2]
    a3_none, a3_glob = cols["AlphaGenome IG CAGE"][:2]
    _, w_a1w = coh.arm("A1w-enformer-single-tss")
    _, w_a3 = coh.arm("A3-alphagenome-single-tss")
    plan["rf-enformer"] = (w_a1w, {"Borzoi CAGE unnormalized": a2wr_none, "Enformer CAGE unnormalized": a1w_none,
                                   "Enformer gene-readout normalized": n4_none, "ABC": abc})
    plan["rf-alphagenome"] = (w_a3, {"AlphaGenome CAGE unnormalized": a3_none,
                                     "AlphaGenome CAGE normalized": a3_glob, "ABC": abc})
    # ---- Section 4.4's two all-pairs auPRCs (values only): AlphaGenome CAGE / RNA-seq with gene-readout normalization
    # ---- E: the reproduced published protocol, all pairs
    plan["all pairs"] = (np.ones(coh.n, bool), {"A9 gene readout": cols["AlphaGenome IG CAGE"][2],
                                                "A11 gene readout": cols["AlphaGenome IG RNA-seq"][2],
                                                "reproduction": _table(coh, REPRODUCTION)})
    B = L.bootstrap(coh, plan)
    # ---- D: the denominators, in the cohort file's row order (its own components and draws)
    cc = L.cohort("cohort")
    D = {"ABC": cc.column(L.ABC)[0], "ABC*": cc.column(L.ABCSTAR)[0],
         "ABC numerator": _numerator_abc(cc), "ABC* numerator": _numerator_abcstar(cc)}
    BD = L.bootstrap(cc, {"all pairs": (np.ones(cc.n, bool), D)})
    _RUN.update(coh=coh, plan=plan, B=B, BD=BD)
    return _RUN


def _table(cc, rel: str) -> np.ndarray:
    """A table's score column (gene, enhancer, score) in the cohort's pair order; every pair must be present."""
    keys, c = L.read_columns(rel, ("gene", "enhancer", "score"), key=())
    v = np.full(cc.n, np.nan)
    for g, e, x in zip(c["gene"], c["enhancer"], c["score"]):
        v[cc.pos[(g, e)]] = float(x)
    assert np.isfinite(v).all()
    return v


def _numerator_abc(cc) -> np.ndarray:
    """ABC without its competing-element denominator (scripts/baselines/score_baselines.py)."""
    return _table(cc, "scores/baselines/abc_numerator.scores.tsv.gz")


def _numerator_abcstar(cc) -> np.ndarray:
    """ABC* without its competing-element denominator (scripts/baselines/score_baselines.py)."""
    return _table(cc, "scores/baselines/abcstar_numerator.scores.tsv.gz")


def compute() -> list:
    R = _run()
    B, BD = R["B"], R["BD"]
    out = []
    # A: 18 cells + 6 gene-readout vs gradient-map contrasts
    for nm, _a, _b in SERIES:
        for lab, a, b in ((f"{nm}: gene-readout divisor vs none", "gene readout", "none"),
                          (f"{nm}: map-mean divisor vs none", "map-mean", "none"),
                          (f"{nm}: both divisors vs none", "both", "none"),
                          (f"{nm}: gene-readout vs map-mean", "gene readout", "map-mean")):
            t = L.contrast(B, nm, a, b)
            tid = ("text:" + TEXT_CELLS[lab]) if lab in TEXT_CELLS else ("fig4:" + lab.replace("map-mean", "gradient-map"))
            out.append(L.row(tid, t))
    # B: 24 ISM cells -- per arm: none -> / wild type (the row move from row 0) and none -> log ratio (the span),
    # with the perturbation mean inside (column 0) and outside (column 1) the absolute value
    for tr, _px in ISM:
        for c, pos in ((0, "perturbation inside"), (1, "perturbation outside")):
            out.append(L.row(f"fig4:{tr} ISM: relative change (/ wild type) vs none ({pos})",
                             L.contrast(B, "ISM " + tr, (1, c), (0, c))))
            out.append(L.row(f"fig4:{tr} ISM: log ratio vs none ({pos})", L.contrast(B, "ISM " + tr, (2, c), (0, c))))
    # C: Section 4.4's tuples on the receptive-field supports
    for tid, sup, a, b in (
            ("Borzoi CAGE vs Enformer CAGE, unnormalized (rf-Enformer pairs)", "rf-enformer",
             "Borzoi CAGE unnormalized", "Enformer CAGE unnormalized"),
            ("Enformer vs ABC, gene-readout normalized (rf-Enformer pairs)", "rf-enformer",
             "Enformer gene-readout normalized", "ABC"),
            ("AlphaGenome CAGE vs ABC, unnormalized (rf-AlphaGenome pairs)", "rf-alphagenome",
             "AlphaGenome CAGE unnormalized", "ABC"),
            ("AlphaGenome CAGE vs ABC, normalized (rf-AlphaGenome pairs)", "rf-alphagenome",
             "AlphaGenome CAGE normalized", "ABC")):
        out.append(L.row("text:" + tid, L.contrast(B, sup, a, b)))
    # D: the competing-element denominators
    out.append(L.row("text:ABC without its competing-enhancer denominator vs ABC",
                     L.contrast(BD, "all pairs", "ABC numerator", "ABC")))
    out.append(L.row("text:ABC* without its competing-enhancer denominator vs ABC*",
                     L.contrast(BD, "all pairs", "ABC* numerator", "ABC*")))
    # E: gene-readout normalization against the reproduced published protocol, all pairs
    for rd, col in (("CAGE", "A9 gene readout"), ("RNA-seq", "A11 gene readout")):
        out.append(L.row(f"text:AlphaGenome {rd} vs reproduced published protocol, gene-readout normalized (all pairs)",
                         L.contrast(B, "all pairs", col, "reproduction")))
    return out


#: values() names: Fig. 4's bars and the ISM figure's cells. Stable: paper/expected/values.json is keyed by them.
STEP_LABEL = {"none": "no normalization", "gene readout": "gene readout", "map-mean": "gradient map",
              "both": "both normalizations"}
ISM_CAL = ("none", "divided by wild type", "log ratio")
ISM_POS = ("perturbation inside", "perturbation outside")


def ism_label(cell) -> str:
    return f"{ISM_CAL[cell[0]]}, {ISM_POS[cell[1]]} auPRC"


def values() -> dict:
    """Fig. 4's bars (the auPRC printed above each bar and its drawn 95% interval) per series, with pairs and
    positives; the ISM divisor figure's cells; Section 4.4's two all-pairs auPRCs with intervals; the four all-pairs
    auPRCs behind the denominator sentence; the q of every test; Section 4.4's two tallies.
    ("Both ... delta auPRC around 0.0" is read off the both / gradient-map bars.)"""
    R = _run()
    B, BD = R["B"], R["BD"]
    v = {}
    for nm, _a, _b in SERIES:
        base = f"fig4/{nm}"
        v[f"{base}/pairs"], v[f"{base}/positives"] = B[nm]["n"], B[nm]["pos"]
        for st in STEPS:
            iv = L.interval(B, nm, st)
            for f, lab in (("v", "auPRC"), ("lo", "lo"), ("hi", "hi")):
                v[f"{base}/{STEP_LABEL[st]}/{lab}"] = iv[f]
    for tr, _px in ISM:
        base = f"ism-divisor/{tr}"
        v[f"{base}/pairs"], v[f"{base}/positives"] = B["ISM " + tr]["n"], B["ISM " + tr]["pos"]
        for cell in ISM_CELLS:
            iv = L.interval(B, "ISM " + tr, cell)
            for f in ("v", "lo", "hi"):
                v[f"{base}/{ism_label(cell)}" + ("" if f == "v" else f"/{f}")] = iv[f]
    for lab, col in (("AlphaGenome CAGE@TSS with gene-readout normalization, all pairs", "A9 gene readout"),
                     ("AlphaGenome RNA-seq@exons with gene-readout normalization, all pairs", "A11 gene readout")):
        iv = L.interval(B, "all pairs", col)
        for f, nm in (("v", "auPRC"), ("lo", "lo"), ("hi", "hi")):
            v[f"sec4.4/{lab}/{nm}"] = iv[f]
    for arm in ("ABC", "ABC numerator", "ABC*", "ABC* numerator"):
        v[f"sec4.4/{arm} auPRC, all pairs"] = BD["all pairs"]["pt"][arm]
    # the q of every test, and Section 4.4's two tallies: "17/18 paired comparisons having q < 0.001" (the 18
    # input-gradient divisor-vs-none cells) and "24/24 ... (all q < 0.001)" (the ISM cells)
    rows = compute()
    q = L.bh_q(rows)
    v.update(L.q_values(rows))
    ig = [r["id"] for r in rows if " IG " in r["id"] and r["id"].endswith("vs none")]
    ism = [r["id"] for r in rows if " ISM:" in r["id"]]
    v["sec4.4/input-gradient divisor-vs-none cells"] = len(ig)
    v["sec4.4/input-gradient divisor-vs-none cells with q < 0.001"] = sum(q[i][0] < 0.001 for i in ig)
    v["sec4.4/input-gradient divisor-vs-none cells with delta > 0"] = sum(r["delta"] > 0 for r in rows if r["id"] in ig)
    v["sec4.4/ISM divisor-vs-none cells"] = len(ism)
    v["sec4.4/ISM divisor-vs-none cells with q < 0.001"] = sum(q[i][0] < 0.001 for i in ism)
    v["sec4.4/ISM divisor-vs-none cells with delta > 0"] = sum(r["delta"] > 0 for r in rows if r["id"] in ism)
    return v
