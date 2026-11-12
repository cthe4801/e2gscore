# AGENTS.md

Compact instructions for coding agents. Humans: start with [README.md](README.md).

Run every command from the repository root. All commands are non-interactive and exit non-zero on failure.
They write only where `--out`, `--manifest`, `--json`, `--prefix` or `--ism-draws` point; scoring also downloads
model weights to the Hugging Face cache and writes the genome's index beside the FASTA.

## Commands

| task | command | environment | time |
|---|---|---|---|
| check the algorithm | `python3 score_eg.py --self-test` (last line: `self-test: 455 passed, 0 failed`) | base | 30 s |
| check the strand and bin frame logic | `python3 tests/test_frame_guard.py` (last line: `OK: 12 tests`) | base | 3 min |
| recompute the paper | `python3 scripts/reproduce_paper.py [family ...] [--out recomputed.tsv]`; `--list` prints the families | base | 15 min (`fig1`: under 1 min) |
| resolve a protocol | `python3 score_eg.py --config <cfg> [--set k=v ...] --print-config` (JSON on stdout; key `config` holds the 42 settings) | base | 1 s |
| evaluate score tables | `python3 scripts/evaluate.py <table> [<table> ...] [--column <col>] [--boot N] [--json out.json]` | base | 30 s |
| score pairs | `python3 score_eg.py --config <cfg> [--set k=v ...] --fasta hg38.fa --out scores.tsv --manifest run.json` | one model | minutes to hours |
| score a few genes | add `--only-genes ENSG...,ENSG...`; add `--device cpu` without a GPU | one model | 1 min per gene on a CPU (Enformer) |
| regenerate the baselines | `python3 scripts/baselines/score_baselines.py --check` | base | 5 s |

Environments, one virtual environment each, never combined:

| name | install | Python | notes |
|---|---|---|---|
| base | `pip install -r requirements-base.txt` | 3.8 or newer | numpy and PyYAML only |
| Enformer | `pip install -r requirements-enformer.txt` | tested on 3.9 | weights download automatically |
| Borzoi | `pip install -r requirements-borzoi.txt` | tested on 3.9 | weights download automatically |
| AlphaGenome | `pip install -r requirements-alphagenome.txt` | 3.12 or newer | set `ALPHAGENOME_PT_WEIGHTS=/path/to/alphagenome.pth` (see `requirements-alphagenome.txt`) |

Scoring also needs the hg38 genome as an uncompressed FASTA (`--fasta`); its index is created on first use.

## How the pieces fit

- A **protocol** is 42 settings. A file in `configs/` sets them; `--set key=value` overrides one. The schema
  (`PARAMS` in `score_eg.py`) is the source of truth, and `docs/parameters.md` lists every setting and value.
  To change a protocol, use a config or `--set`; do not edit `score_eg.py`.
- `--set key=null` unsets a setting. An invalid combination raises `ConfigError` with the reason.
- **`reproducing.tsv`** (tab-separated; columns `result`, `config`, `overrides`, `anchors`, `figures`, `note`)
  maps each table in `scores/` to its command:
  `python3 score_eg.py --config <config> --set <override> ... [--anchors <anchors>] --fasta hg38.fa --out <result>`.
  Rows whose `config` is `(derived)` come from `scripts/ism_statistic_tables.py`, and `(baseline)` rows from
  `scripts/baselines/score_baselines.py`; the `note` gives the command.
- A **score table** is a TSV, one row per enhancer–gene pair, with the columns `screen`, `gene`, `enhancer`,
  `chrom`, `start`, `end`, `distance_to_tss`, `in_window`, `score`, and `score__*` columns when the run used
  `--free-grid` (eight shipped reproduction tables have no `in_window`). Each shipped table from a run has a
  `<table>.manifest.json` beside it with its resolved protocol.
- `scripts/evaluate.py` evaluates tables of the K562 benchmark (`screen: gschwind`, 10,353 pairs): it takes the
  labels from `inputs/gschwind.hg38.tsv.gz` and scores the pairs a table holds, plain or gzipped.
- `scripts/reproduce_paper.py` reads only files in the repository (`scores/`, `inputs/`, `data/`, `paper/`) and
  compares what it computes with `paper/expected/`.

## Rules

- **Row order is data.** Never sort or concatenate score tables: the bootstrap numbers its resampling units in
  table order. Merge the parts of a `--shard I/N` run with `scripts/merge_shards.py`.
- Keep the files in `scores/`, `inputs/`, `data/` and `paper/` unchanged; `reproduce_paper.py` checks against them.
- Never install two model requirement files into one environment: their pinned dependencies conflict.
- CPU runs are deterministic. A re-run agrees with the shipped tables to about 1e-3 relative, not bit for bit:
  the tables were computed on GPUs, whose backward passes are not deterministic.
- AlphaGenome input gradients peak at about 42 GiB of GPU memory.
- The genome and model weights are never part of the repository; do not add them.
- Files listed in `NOTICE-AlphaGenome.md` are under the AlphaGenome Output Terms of Use: non-commercial use
  only, and no training of machine-learning models on them.

## More

[docs/parameters.md](docs/parameters.md) · [docs/scoring.md](docs/scoring.md) ·
[docs/reproduce.md](docs/reproduce.md) · [docs/baselines.md](docs/baselines.md) · [docs/data.md](docs/data.md)
