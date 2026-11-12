#!/usr/bin/env python3
"""
merge_shards.py — rebuild one score table from the partial tables of a `--shard i/n` run.

    python3 scripts/merge_shards.py --out <arm>.scores.tsv \
        <arm>.shard0.scores.tsv <arm>.shard1.scores.tsv

WHY A MERGE AND NOT `cat`. The paper's bootstrap numbers the connected components of the
gene-enhancer graph in the order pairs appear in the table, so the components each bootstrap draw
picks, and with them every interval and p-value, depend on row order. Concatenating shard 0 then
shard 1 gives genes in the order 0,2,4,...,1,3,5,..., which is a permutation of the unsharded
order and would therefore move intervals while every row in the file is individually correct. That is
the failure mode this script exists to prevent: it stable-sorts on the `_gene_rank` each shard
recorded, which reproduces the unsharded order exactly, and drops the column so the merged file
is indistinguishable from a single-process run.

Checks, all fatal:
  * every input carries `_gene_rank` (a file without it is an unsharded table, not a shard)
  * the shards' gene ranks are DISJOINT (no gene computed twice)
  * the union of ranks is contiguous from 0 (no gene missing -- a died-early shard is the thing
    most likely to go unnoticed, because the merged file still looks like a valid score table)
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("shards", nargs="+", help="partial score TSVs from --shard runs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--allow-gaps", action="store_true",
                    help="permit a non-contiguous rank set (only for a deliberate subset run)")
    ap.add_argument("--allow-missing-manifests", action="store_true",
                    help="merge without checking that every shard resolved the same theta. The "
                         "merged table then carries NO provenance; only for tables built by hand")
    args = ap.parse_args(argv)

    rows, header, seen = [], None, {}
    for path in args.shards:
        with open(path, newline="") as fh:
            rd = csv.DictReader(fh, delimiter="\t")
            if rd.fieldnames is None or "_gene_rank" not in rd.fieldnames:
                ap.error(f"{path} has no `_gene_rank` column -- that is an unsharded table, "
                         "not a shard. Merging it would be a silent duplication.")
            if header is None:
                header = [f for f in rd.fieldnames if f != "_gene_rank"]
            elif [f for f in rd.fieldnames if f != "_gene_rank"] != header:
                ap.error(f"{path} has different columns from {args.shards[0]}")
            n = 0
            for r in rd:
                k = int(r["_gene_rank"])
                if k in seen and seen[k] != path:
                    ap.error(f"gene rank {k} appears in both {seen[k]} and {path}; "
                             "the shards overlap, so some gene was computed twice")
                seen[k] = path
                rows.append((k, len(rows), r))
                n += 1
            print(f"  {os.path.basename(path):50s} {n:>8,} rows")

    ranks = sorted(seen)
    if not args.allow_gaps:
        missing = [k for k in range(max(ranks) + 1) if k not in seen] if ranks else []
        if missing:
            ap.error(f"{len(missing)} gene rank(s) missing from the union, first 10: "
                     f"{missing[:10]}. A shard died or was never run -- the merged table would "
                     "silently be a subset. Re-run the missing shard, or pass --allow-gaps if "
                     "the subset is deliberate.")

    # --- the merged manifest, and the check that makes merging legitimate at all ----------
    # Sharding is only sound if every shard computed the SAME experiment. Nothing else in the
    # pipeline verifies that: the score tables carry no theta, so two shards launched with
    # different configs, different weights or different offload modes would merge into a table
    # that looks perfectly ordinary and is a chimera.
    #
    # It also writes the merged table's manifest. `--shard` writes one manifest per shard, so
    # without this the merged table would carry NO provenance at all -- no theta, no weights
    # hash, no precision, no gate readings.
    mans, theta0, src = [], None, None
    for path in args.shards:
        # `.scores.tsv` -> `.manifest.json`, and a bare `.tsv` -> `.manifest.json` too. A plain
        # `str.replace(".scores.tsv", ...)` silently returns the INPUT PATH for a file named
        # `shard0.tsv`, which then exists and gets handed to json.load -- so the failure would be
        # a JSONDecodeError on a TSV rather than a clear message.
        mp = re.sub(r"(\.scores)?\.tsv$", ".manifest.json", path)
        if mp == path:
            ap.error(f"cannot derive a manifest path from {path!r} (expected it to end .tsv)")
        if not os.path.exists(mp):
            if args.allow_missing_manifests:
                print(f"  WARNING: no manifest for {os.path.basename(path)}; "
                      "theta agreement across shards is NOT checked")
                continue
            ap.error(f"{path} has no sibling manifest at {mp}. Refusing to merge a table whose "
                     "provenance cannot be checked. Pass --allow-missing-manifests to override, "
                     "but then the merged table carries no provenance -- see the header.")
        m = json.load(open(mp))
        mans.append((mp, m))
        # `config` is the RESOLVED theta -- score_eg.manifest() writes every non-underscore key
        # of theta into it. Do NOT fall back to the whole manifest
        # if it is absent: that would compare per-shard COUNTERS (gpu_calls, n_forward), which
        # differ by construction, so every merge would fail as a false chimera and the obvious
        # fix would be to weaken the check. Fail loudly on the unexpected shape instead.
        th = m.get("theta") or m.get("config")
        if th is None:
            ap.error(f"{mp} has neither a `theta` nor a `config` block, so the shards cannot be "
                     "checked for agreement. That is not a manifest score_eg.py wrote.")
        if theta0 is None:
            theta0, src = th, mp
        elif th != theta0:
            diff = sorted(k for k in set(th) | set(theta0)
                          if th.get(k) != theta0.get(k))
            ap.error(f"{mp} resolved a DIFFERENT experiment from {src}; merging them would "
                     f"produce a chimera. Differing keys: {diff}")

    merged_manifest = None
    if not mans:
        print("  WARNING: no manifests found; the merged table carries NO provenance")
    else:
        merged_manifest = dict(mans[0][1])
        merged_manifest["_merged_from"] = [os.path.basename(p) for p, _ in mans]
        merged_manifest["_n_shards"] = len(mans)
        merged_manifest["_shard"] = None      # the merged table is not a shard
        # Counters are per-shard, so they SUM. Everything else is identical by the theta check
        # above, which is what makes taking shard 0's copy of it legitimate.
        for key in ("gpu_calls", "cache_hits", "pairs_scored", "pairs_dropped"):
            vals = [m.get("stats", {}).get(key) for _, m in mans]
            if vals and all(isinstance(v, int) for v in vals):
                merged_manifest.setdefault("stats", {})[key] = sum(vals)
        for key in ("n_forward", "n_gradient"):
            vals = [m.get("runtime", {}).get(key) for _, m in mans]
            if vals and all(isinstance(v, int) for v in vals):
                merged_manifest.setdefault("runtime", {})[key] = sum(vals)
        # the scalar-mirror gate is a WORST case across the arm, so it maxes rather than sums --
        # summing it would report a bound no single call ever hit
        worst = [m.get("runtime", {}).get("scalar_gate_worst_rel") for _, m in mans]
        if worst and all(isinstance(v, (int, float)) for v in worst):
            merged_manifest.setdefault("runtime", {})["scalar_gate_worst_rel"] = max(worst)
            print(f"  provenance: {len(mans)} shard manifests agree on theta; "
                  f"worst scalar gate {max(worst):.2e}")
        else:
            print(f"  provenance: {len(mans)} shard manifests agree on theta")

    # stable: (gene rank, then original within-shard order) reproduces the unsharded emission
    rows.sort(key=lambda t: (t[0], t[1]))
    with open(args.out, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=header, delimiter="\t", lineterminator="\n")
        wr.writeheader()
        for _, _, r in rows:
            r.pop("_gene_rank", None)
            wr.writerow(r)
    print(f"merged {len(args.shards)} shards / {len(ranks):,} genes / {len(rows):,} rows "
          f"-> {args.out}")
    if merged_manifest is not None:
        dest = re.sub(r"(\.scores)?\.tsv$", ".manifest.json", args.out)
        if dest == args.out:
            dest = args.out + ".manifest.json"
        with open(dest, "w") as fh:
            json.dump(merged_manifest, fh, indent=1, sort_keys=True, default=str)
        print(f"  manifest -> {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
