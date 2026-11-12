#!/usr/bin/env python3
"""Recompute every statistic the paper reports from the shipped scores, and check each one against the paper.

    python3 scripts/reproduce_paper.py                 # all families, all figure values, the reproductions
    python3 scripts/reproduce_paper.py fig1 fig4       # selected families (and nothing else)
    python3 scripts/reproduce_paper.py --list          # the families and value modules
    python3 scripts/reproduce_paper.py --out recomputed.tsv

WHAT IS CHECKED.
  * Supplementary Table 2: every comparison the paper reports (paper/expected/supp_table2.tsv), grouped into the
    paper's families (one figure = one family). For each: the paired difference, its 95% interval and its bootstrap
    p-value, recomputed by the family's module in scripts/paper/; then Benjamini-Hochberg within each family, giving
    the q-value and the verdict at 0.05.
  * The values each figure prints (auPRC bars, within-gene auROC, shares, counts) and the three reproductions of
    published results (paper/expected/values.json).
A recomputed number agrees when it is within half a unit of the last decimal the paper prints.

CONVENTIONS (Section 3.4 and Supplementary Table 3 of the paper). Pooled auPRC is tie-aware step-wise average
precision. Intervals and p-values come from 10,000 paired bootstrap replicates over the 663 connected components of
the gene-enhancer graph (seed 20260816) unless a figure states otherwise. A two-sided bootstrap p below the
resolution of 10,000 replicates is known only as p < 2e-4: it enters Benjamini-Hochberg at 2e-4 and its q is an
upper bound, printed "<" and rounded up. Ties in p are ranked by the comparison's name.

Needs numpy only. Every input is read from this repository (scores/, inputs/, data/, paper/data/).
"""
from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
os.environ.setdefault("E2G_RELEASE", ROOT)
sys.path.insert(0, os.path.join(HERE, "paper"))

ALPHA = 0.05
RESOLUTION = 2e-4
EXPECTED_TESTS = os.path.join(ROOT, "paper", "expected", "supp_table2.tsv")
EXPECTED_VALUES = os.path.join(ROOT, "paper", "expected", "values.json")
#: the paper's families, in the order Supplementary Table 2 lists them
FAMILIES = ["fig1", "fig2", "fig3", "fig4", "fig5", "fig6-anchor", "fig6-width", "fig6-gc", "fig6-pool",
            "fig6-tta", "sfig-support", "sfig-wg-support", "sfig-ism", "sfig-allpairs-anchor",
            "sfig-allpairs-celltype", "sfig-cpg", "sfig-ismdraws", "screen-Gasperini2019", "screen-Nasser2021",
            "screen-Schraivogel2020"]
#: modules whose values() hold figure values or reproductions (no Benjamini-Hochberg family)
VALUE_MODULES = ["fam_fig1", "fam_fig2", "fam_fig3", "fam_fig4", "fam_fig5", "fam_fig6_anchor", "fam_fig6_width", "fam_fig6_gc", "fam_fig6_pool", "fam_fig6_tta", "fam_sfig_support", "fam_sfig_wg_support", "fam_sfig_ism", "fam_sfig_allpairs_anchor", "fam_sfig_allpairs_celltype", "fam_sfig_cpg", "fam_sfig_ismdraws", "fam_screen_Gasperini2019", "fam_screen_Nasser2021", "fam_screen_Schraivogel2020", "rep_avsec2021", "rep_linder2025", "rep_avsec2026", "sup_attr", "sup_bootstrap", "sup_settings_counts"]


def decimals(s: str) -> int:
    """Decimals the paper printed for a number given as text (at least 3, as the comparison table prints)."""
    s = s.strip()
    if "e" in s.lower():
        return 6
    return max(len(s.split(".")[1]) if "." in s else 0, 3)


def agree(got: float, printed: str) -> bool:
    if got is None or (isinstance(got, float) and math.isnan(got)):
        return False
    return abs(float(got) - float(printed)) <= 0.5 * 10 ** (-decimals(printed)) + 1e-12


def load_expected():
    with open(EXPECTED_TESTS, encoding="utf-8") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def below(p) -> bool:
    return p is None or p < RESOLUTION - 1e-12


def bh(tests: list) -> None:
    """Benjamini-Hochberg within each family, in place: q, rank, rejected, q_bound."""
    P = lambda p: RESOLUTION if (p is None or p < RESOLUTION) else float(p)   # noqa: E731
    fams = {}
    for t in tests:
        fams.setdefault(t["family"], []).append(t)
    for ts in fams.values():
        m = len(ts)
        order = sorted(range(m), key=lambda i: (P(ts[i]["p_bh"]), ts[i]["name"]))
        k = 0
        for r, i in enumerate(order, 1):
            if P(ts[i]["p_bh"]) <= r / m * ALPHA:
                k = r
        run = 1.0
        for r in range(m, 0, -1):
            i = order[r - 1]
            run = min(run, P(ts[i]["p_bh"]) * m / r)
            ts[i]["q"] = run
        for r, i in enumerate(order, 1):
            ts[i]["rank"] = r
            ts[i]["rejected"] = r <= k
            ts[i]["q_bound"] = below(ts[i]["p_bh"])


def fmt_q(q: float, bound: bool) -> str:
    """Four decimals, as the paper prints; an upper bound is rounded up so the inequality stays true."""
    if bound:
        return f"{math.ceil(q * 1e4 - 1e-9) / 1e4:.4f}"
    return f"{q:.4f}"


def run_families(families, expected):
    got = []
    for fam in families:
        t0 = time.time()
        mod = importlib.import_module("fam_" + fam.replace("-", "_"))
        res = {r["id"]: r for r in mod.compute()}
        exp = [e for e in expected if e["family"] == fam]
        n_ok, problems = 0, []
        for e in exp:
            r = res.get(e["id"])
            if r is None:
                problems.append(f"   MISSING  {e['id']}")
                continue
            bad = [k for k in ("delta", "lo", "hi", "p") if not agree(r[k], e[k])]
            if float(e["p"]) == 0.0 and not below(r["p"]) and "p" not in bad:
                bad.append("p")                # the paper's p = 0 means p < 2e-4
            if bad:
                problems.append(f"   DIFFERS  {e['id']}: " + ", ".join(f"{k} {r[k]:.6f} (paper {e[k]})" for k in bad))
            else:
                n_ok += 1
            # Benjamini-Hochberg takes each p at the precision the paper's table stores it (exact for the
            # comparisons the text quotes, four decimals for most figure cells)
            got.append(dict(id=e["id"], family=fam, name=e["name"], delta=r["delta"], lo=r["lo"], hi=r["hi"],
                            p=r["p"], p_bh=round(float(r["p"]), decimals(e["p"]))))
        print(f"{fam:24s} {n_ok:3d}/{len(exp):3d} comparisons agree   ({time.time() - t0:5.0f} s)", flush=True)
        for line in problems:
            print(line)
    return got


def check_q(got, expected, families):
    """BH within each family needs the whole family; run it only for complete families."""
    complete = [f for f in families if all(e["id"] in {g["id"] for g in got} for e in expected if e["family"] == f)]
    sub = [g for g in got if g["family"] in complete]
    bh(sub)
    exp = {e["id"]: e for e in expected}
    n_ok, problems = 0, []
    for g in sub:
        e = exp[g["id"]]
        q_txt = fmt_q(g["q"], g["q_bound"])
        ok = (q_txt == e["q"] and str(g["q_bound"]) == e["q_bound"] and str(g["rejected"]) == e["rejected"])
        n_ok += ok
        if not ok:
            problems.append(f"   q DIFFERS {g['id']}: {'<' if g['q_bound'] else '='} {q_txt}, "
                            f"{'rejected' if g['rejected'] else 'not rejected'} (paper {e['q']}, bound "
                            f"{e['q_bound']}, rejected {e['rejected']})")
    print(f"{'Benjamini-Hochberg':24s} {n_ok:3d}/{len(sub):3d} q-values and verdicts agree "
          f"({len(complete)} families)")
    for line in problems:
        print(line)
    return n_ok == len(sub)


def value_agrees(got, exp) -> bool:
    """A figure value agrees with the paper's when it matches at the precision the paper holds it:
    a printed string at its printed decimals; a count exactly; a stored real to half a unit in its last stored
    decimal, or to 1e-9 relative when it is stored at full precision."""
    if got is None:
        return exp is None
    if isinstance(exp, bool) or isinstance(got, bool):
        return bool(got) == bool(exp)
    if exp is None:
        return isinstance(got, float) and math.isnan(got)
    if isinstance(exp, str):
        try:
            e = float(exp.replace(",", ""))
        except ValueError:
            return str(got) == exp
        d = len(exp.split(".")[1]) if "." in exp else 0
        return abs(float(got) - e) <= 0.5 * 10 ** (-d) + 1e-12
    if isinstance(exp, int):
        return float(got) == exp
    e, g = float(exp), float(got)
    if math.isnan(e) or math.isnan(g):
        return math.isnan(e) and math.isnan(g)
    r = repr(e)
    d = len(r.split(".")[1]) if ("." in r and "e" not in r.lower()) else 99
    tol = max(0.5 * 10 ** (-d) if d <= 12 else 0.0, 1e-9 * max(1.0, abs(e)))
    return abs(g - e) <= tol


def run_values(modules, expected_values):
    n_ok = n_all = 0
    for name in modules:
        t0 = time.time()
        mod = importlib.import_module(name)
        vals = mod.values()
        exp = expected_values.get(name, {})
        ok, problems = 0, []
        for label, ev in exp.items():
            gv = vals.get(label)
            if label not in vals and ev is not None:
                problems.append(f"   MISSING  {label}")
            elif value_agrees(gv, ev):
                ok += 1
            else:
                problems.append(f"   DIFFERS  {label}: {gv} (paper {ev})")
        print(f"{name:24s} {ok:3d}/{len(exp):3d} values agree   ({time.time() - t0:5.0f} s)", flush=True)
        for line in problems:
            print(line)
        n_ok += ok
        n_all += len(exp)
    return n_ok, n_all


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("only", nargs="*", help="families or value modules to run (default: all)")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--out", help="write every recomputed comparison (with q) to this TSV")
    a = ap.parse_args(argv)
    if a.list:
        print("families:", " ".join(FAMILIES))
        print("value modules:", " ".join(VALUE_MODULES))
        return 0
    fams = [f for f in FAMILIES if not a.only or f in a.only]
    picked = {f.replace("-", "_") for f in a.only}
    vmods = [m for m in VALUE_MODULES if not a.only or m in a.only or (m.startswith("fam_") and m[4:] in picked)]
    unknown = [x for x in a.only if x not in FAMILIES and x not in VALUE_MODULES]
    if unknown:
        ap.error(f"unknown: {unknown}; see --list")
    expected = load_expected()
    t0 = time.time()
    got = run_families(fams, expected) if fams else []
    ok_tests = sum(1 for g in got for e in expected if e["id"] == g["id"]
                   and all(agree(g[k], e[k]) for k in ("delta", "lo", "hi", "p")))
    ok_q = check_q(got, expected, fams) if fams else True
    vals_ok = vals_all = 0
    if vmods:
        expected_values = json.load(open(EXPECTED_VALUES, encoding="utf-8"))
        vals_ok, vals_all = run_values(vmods, expected_values)
    if a.out and got:
        with open(a.out, "w", newline="") as fh:
            w = csv.writer(fh, delimiter="\t", lineterminator="\n")
            w.writerow(["id", "family", "name", "delta", "lo", "hi", "p", "q", "q_bound", "rejected"])
            for g in got:
                w.writerow([g["id"], g["family"], g["name"], g["delta"], g["lo"], g["hi"], g["p"],
                            g.get("q", ""), g.get("q_bound", ""), g.get("rejected", "")])
    n_exp = sum(1 for e in expected if e["family"] in fams)
    print(f"\n{ok_tests}/{n_exp} comparisons, {'all' if ok_q else 'NOT all'} q-values, "
          f"{vals_ok}/{vals_all} values agree with the paper ({(time.time() - t0) / 60:.1f} min)")
    return 0 if (ok_tests == n_exp and ok_q and vals_ok == vals_all) else 1


if __name__ == "__main__":
    raise SystemExit(main())
