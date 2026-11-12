# Running the models

Score from sequence with any config:

```bash
python3 score_eg.py --config configs/<name>.yaml [--set key=value ...] \
    --fasta /path/to/hg38.fa --out scores.tsv --manifest manifest.json
```

The manifest records the resolved protocol, the backend's provenance (weights,
precision, device) and the frame guard's certificate. `--free-grid` additionally
writes every pooling kernel and normalisation of the free grid as extra
`score__*` columns (the matched-protocol tables were run with it; Fig. 6's
pooling and gradient-correction ranges read those columns). `--shard I/N` spreads one run over N processes;
rebuild the table with `scripts/merge_shards.py` (never `cat`: the paper's
bootstrap numbers its resampling units in table order, so row order changes
intervals and p-values).

Install the environment for the model you want and **keep them separate**:
the Enformer file pins `transformers==4.56.2` while `borzoi-pytorch` needs
`>=4.57.6`, and installing both leaves Borzoi with the older `transformers` and raises a torch
"Cannot copy out of meta tensor" error that looks like a hardware fault.

| model | requirements file | weights |
|---|---|---|
| Enformer | `requirements-enformer.txt` | automatic, Hugging Face Hub (`EleutherAI/enformer-official-rough`) |
| Borzoi | `requirements-borzoi.txt` | automatic, Hugging Face Hub (`johahi/borzoi-replicate-0..3`) |
| AlphaGenome | `requirements-alphagenome.txt` (Python 3.12 or newer) | **manual**, see below |

The reference genome is never bundled. Download GRCh38 (for example UCSC
`hg38.fa.gz`), uncompress it and pass it with `--fasta`; its index is created on
first use.

**AlphaGenome weights.** The reported numbers came from a PyTorch checkpoint
converted from DeepMind's official JAX/Orbax release (`all_folds`) with the port's
own converter, pointed at via `ALPHAGENOME_PT_WEIGHTS`. A third-party mirror of
AlphaGenome-PyTorch weights exists and was deliberately not used: the numerical
agreement of this backend was established only against that conversion.
Obtaining the weights requires accepting the AlphaGenome Model Parameters Terms
of Use.

```bash
agt convert --input <alphagenome-all-folds> --output alphagenome.pth   # the port's converter
export ALPHAGENOME_PT_WEIGHTS=/path/to/alphagenome.pth
```

AlphaGenome input gradients peak at about 42 GiB of GPU memory.

## Try one gene first

`--only-genes` takes Ensembl gene ids and scores only those genes' pairs, and `--device cpu` runs without a GPU:

```bash
python3 score_eg.py --config configs/unified-t1-backbone-enformer.yaml \
    --only-genes ENSG00000023516 --device cpu --fasta /path/to/hg38.fa --out one_gene.tsv
```

## Output

One row per enhancer–gene pair, in the screen's order:

| column | meaning |
|---|---|
| `screen`, `gene`, `enhancer`, `chrom`, `start`, `end` | the pair: screen name, Ensembl gene id (gene symbol for the screens keyed by symbol), element id and coordinates (hg38) |
| `distance_to_tss` | enhancer midpoint minus the gene's TSS, in bp |
| `in_window` | whether the model's input window reaches the enhancer; pairs it cannot reach score 0 (`out_of_window: zero`). Eight shipped tables in `scores/avsec2021/` and `scores/avsec2026/` do not have this column |
| `score` | the enhancer–gene score under the config's protocol |
| `score__<pooling>-<normalise>-ms<0/1>` | only with `--free-grid`: the same run reduced with other pooling kernels and normalisations |

## In-silico mutagenesis runs

An ISM run reduces its perturbation draws to one statistic. `--ism-draws run.npz` keeps the draws, and
`scripts/ism_statistic_tables.py` then writes one score table per statistic from them, with no further model call:

```bash
python3 score_eg.py --config configs/unified-t1-backbone-enformer-pertrack-ism.yaml \
    --set anchor=single-tss --set ism_mode=eulerian-dinuc-shuffle --set ism_shuffles=5 \
    --fasta /path/to/hg38.fa --out run.scores.tsv --ism-draws run.ism_draws.npz
python3 scripts/ism_statistic_tables.py --config configs/unified-t1-backbone-enformer-pertrack-ism.yaml \
    --set anchor=single-tss --set ism_mode=eulerian-dinuc-shuffle --set ism_shuffles=5 \
    --scores run.scores.tsv --draws run.ism_draws.npz --strand-mask auto --prefix out/run
```

The paper's matched ISM protocol reads `log_abs_diff` over ten draws: five at `seed=0` and five at `seed=1`,
passed as two `--draws` arguments.

## Speed

`score_eg.py` caches everything the pair reduction derives from a map but not
from the enhancer, memoises the Gaussian weights (both bit-identical
to the uncached computation; the self-test compares them), encodes one-hot input
from a byte table, runs AlphaGenome's 1 bp decoder only over the readout span plus
a 2,048 bp halo (its receptive field is 762 bp per side;
`SCORE_EG_AG_DECODER_CROP=0` restores the full decoder) with host offload decided
per gene (`SCORE_EG_AG_OFFLOAD=auto`), and keeps Borzoi's four replicates resident
with `EG_GPU_REPLICATE_CACHE=1`.

## Determinism

Re-scoring is deterministic on the CPU, but not bit-exact across devices:
  cuDNN's backward pass is not deterministic, so a GPU reproduces itself only
  to about 1e-4 relative, and the CPU differs from the GPUs the tables were
  computed on by up to about 1e-3 relative (more where a major-TSS anchor is
  re-picked between near-tied candidates). These differences are far below the
  bootstrap intervals the paper reports.

## Correctness guards

**The reverse-complement frame.** A reverse-complement augmentation feeds the
model the reverse-complemented input, so on that pass output bin `b` is genomic
bin `n_bins - 1 - b` and the `+` tracks carry the `-` strand; the gene's readout
must be read in that frame. Four layers make any other reading impossible, on
every model:

1. **One mapping.** `frame_for` is the only place an augmentation becomes a
   readout frame; the gene scalar and the attention query both take their bins
   from it.
2. **No silent default.** Every model pass runs inside `augmented_frame`, and a
   readout built there for any other augmentation, a forgotten `aug=` included,
   raises `FrameError`.
3. **An exact probe on the real geometry.** Before the first model call of every
   run, `frame_selftest` drives the production readout, gradient-map, ISM and
   attention code through a stand-in that is exactly reverse-complement and
   whole-bin-shift equivariant on the model's own geometry, and requires every
   augmented pass to reproduce the forward pass. It must also be able to fail:
   reading the forward frame on the rc pass has to change the readout, or the
   probe reports itself blind.
4. **The real model agrees with itself.** `FrameAgreementGate` compares each
   gene's reverse-complement readout with its forward readout from the passes the
   run makes anyway, and stops the run if the median |log ratio| over 16 genes
   exceeds 0.5. `SCORE_EG_FRAME_GATE_MAX` changes the threshold and
   `SCORE_EG_FRAME_GATE=off` disables it.

Shifts are either under half an output bin, where the readout stays put, or a
whole number of bins, where it follows; anything in between is refused. The
manifest records the probe's certificate and the gate's verdict under
`runtime.frame_guard`.

**Exact readout derivatives.** Every backend differentiates the gene scalar with
respect to the model output in float64: Enformer in closed form where the scalar
is linear-then-log (every shipped config), gated on every call against the
scalar itself (`runtime.dudy`; `SCORE_EG_ENF_DUDY=numeric` forces float64 central
differences); Borzoi and AlphaGenome through a gated float64 torch mirror of the
scalar.

**TF32 off.** On Ampere and newer GPUs torch would otherwise run convolutions with
a 10-bit mantissa. All three backends pin TF32 off at construction.
