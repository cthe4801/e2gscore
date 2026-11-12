# E2GScore

Score enhancer–gene pairs with Enformer, Borzoi or AlphaGenome, and recompute every number in
*Benchmarking the Protocol, Not Just the Model: Failure Modes in Zero-Shot Enhancer–Gene Linking*
(ICBINB-BIO workshop, NeurIPS 2026; [OpenReview](https://openreview.net/forum?id=xcyw2PR0ZJ)).

A sequence model does not output an enhancer–gene score. The score comes from a procedure wrapped around the
model, with 42 settings: which track to read, where to centre the input, gradients or mutagenesis, how to
normalise, and so on. `score_eg.py` is that procedure, the same code for all three models, and one YAML file in
`configs/` fixes the settings.

## Install

```bash
git clone https://github.com/cthe4801/e2gscore.git && cd e2gscore
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements-base.txt        # numpy and PyYAML: all that steps 1 and 2 need
```

## 1. Recompute the paper's numbers (no GPU)

```bash
python3 scripts/reproduce_paper.py fig1     # one figure, under a minute
python3 scripts/reproduce_paper.py          # every figure and table, about 15 minutes
```

Each line of output is one figure checked against the paper. The full run ends with:

```
419/419 comparisons, all q-values, 6456/6456 values agree with the paper
```

## 2. Compare score tables (no GPU)

The per-pair scores behind every result are in `scores/`. Compare any of them on the K562 benchmark:

```bash
python3 scripts/evaluate.py \
    scores/unified-t1/unified-t1-A1w-enformer-single-tss.scores.tsv.gz \
    scores/unified-t1/unified-t1-A2wR-borzoi-single-tss.scores.tsv.gz
```

It prints auPRC with a 95% interval for each table and then their paired difference, on all pairs and on the
pairs each model's input window reaches, overall and by distance (30 seconds):

```
support            bin          arm                                            auPRC            95% CI       n   pos   cov+
all                All          unified-t1-A1w-enformer-single-tss            0.5142 [0.4487, 0.5808]  10,353   471 100.0%
all                All          unified-t1-A2wR-borzoi-single-tss             0.5863 [0.5273, 0.6460]  10,353   471 100.0%
...
support            bin          A vs B                                                delta              95% CI       p
all                All          1w-enformer-single-tss vs 2wR-borzoi-single-tss     -0.0722 [-0.1051, -0.0409]  0.000 *
```

## 3. Score pairs with a model

This needs the model's packages and the hg38 genome. A GPU is strongly recommended for full runs.

```bash
python3 -m venv .venv-enformer && . .venv-enformer/bin/activate
pip install -r requirements-enformer.txt
wget https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz && gunzip hg38.fa.gz

# try one gene on the CPU: about a minute, 9 GB of memory
python3 score_eg.py --config configs/unified-t1-backbone-enformer.yaml \
    --only-genes ENSG00000023516 --device cpu --fasta hg38.fa --out one_gene.tsv

# the whole K562 benchmark (10,353 pairs): about 10 minutes on one recent GPU
python3 score_eg.py --config configs/unified-t1-backbone-enformer.yaml --fasta hg38.fa --out scores.tsv
```

The output has one row per pair with its `score`. `python3 scripts/evaluate.py scores.tsv` evaluates a
full-benchmark table.

| model | install | weights |
|---|---|---|
| Enformer | `requirements-enformer.txt` | downloaded automatically |
| Borzoi | `requirements-borzoi.txt` | downloaded automatically |
| AlphaGenome | `requirements-alphagenome.txt` (Python 3.12 or newer) | manual: see [docs/scoring.md](docs/scoring.md) |

Use a separate virtual environment for each model: their dependencies conflict.

## 4. Change the protocol

```bash
# show all 42 settings of a config
python3 score_eg.py --config configs/unified-t1-backbone-enformer.yaml --print-config

# change settings without editing the file, e.g. read every cell type's tracks instead of K562's
python3 score_eg.py --config configs/unified-t1-backbone-enformer.yaml --set cell_type=all \
    --fasta hg38.fa --out scores.tsv
```

Every setting and its allowed values: [docs/parameters.md](docs/parameters.md). A combination that makes no
sense is refused with a message that says why.

## 5. Re-run an experiment from the paper

[`reproducing.tsv`](reproducing.tsv) has one row per table in `scores/`: the config and `--set` overrides that
produce it, and the figures that use it. For example, the row

```
scores/unified-t1/unified-t1-N1-enformer-single-tss-agnostic.scores.tsv.gz   configs/unified-t1-backbone-enformer.yaml   anchor=single-tss cell_type=all
```

is the command

```bash
python3 score_eg.py --config configs/unified-t1-backbone-enformer.yaml \
    --set anchor=single-tss --set cell_type=all --fasta hg38.fa --out scores.tsv
```

## What is where

| path | what it is |
|---|---|
| `score_eg.py`, `backend_*.py` | the scoring procedure, and one adapter per model |
| `configs/` | one protocol file per experiment |
| `reproducing.tsv` | which config and overrides produce each score table |
| `scores/` | the per-pair scores behind the paper (gzipped TSV) |
| `inputs/`, `data/` | the CRISPRi benchmark tables and the gene annotations |
| `scripts/` | `reproduce_paper.py`, `evaluate.py`, the baselines and helpers |
| `paper/` | the values the paper prints, which `reproduce_paper.py` checks against |
| `docs/` | reference: [parameters](docs/parameters.md), [running the models](docs/scoring.md), [reproducing the paper](docs/reproduce.md), [baselines](docs/baselines.md), [data](docs/data.md) |
| `AGENTS.md` | the same instructions in a compact form for coding agents |

## Licence

The code is under the MIT licence (`LICENSE`). The score tables computed with AlphaGenome, and the files
derived from them (listed in `NOTICE-AlphaGenome.md`), are instead provided under the
[AlphaGenome Output Terms of Use](http://deepmind.google.com/science/alphagenome/output-terms): non-commercial
use only, and no training of machine-learning models on them (see `LEGALLY_BINDING_TERMS_OF_USE.txt`). The
benchmark tables keep their sources' terms ([docs/data.md](docs/data.md)); please cite the original studies.

## Citation

```bibtex
@inproceedings{thetjoean2026protocol,
  title     = {Benchmarking the Protocol, Not Just the Model: Failure Modes in
               Zero-Shot Enhancer--Gene Linking},
  author    = {The-Tjoean, Caleb and Alinejad-Rokny, Hamid and Voineagu, Irina},
  booktitle = {ICBINB-BIO Workshop at NeurIPS},
  year      = {2026}
}
```
