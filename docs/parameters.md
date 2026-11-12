# The 42 parameters

A protocol is one value for each of these. A config file in `configs/` sets them, `--set key=value`
overrides one, and `python3 score_eg.py --config <file> --print-config` prints the full resolved set.
Combinations that make no sense are refused with a message that says why.

**Matched protocol** is the value in `configs/unified-t1-backbone-<model>.yaml`, the protocol the paper
holds fixed to compare the three models. **Default** is what an unset key takes.

## Model

| parameter | values | default | matched protocol | what it does |
|---|---|---|---|---|
| `model` | enformer · borzoi · alphagenome | enformer | one per config | Which model scores the pairs |
| `replicates` | 1 · 4 | 1 | 1 (Enformer) · 4 (Borzoi) · 1 (AlphaGenome) | How many checkpoint folds are scored and averaged (Borzoi has 4) |

## Pairs and annotation

| parameter | values | default | matched protocol | what it does |
|---|---|---|---|---|
| `screen` | fulco2019 · flowfish-plus · gasperini2019 · gasperini2019-linder · gschwind · flowfish-plus-authors · gasperini2019-linder-authors | gschwind | gschwind | Which CRISPRi screen supplies the enhancer–gene pairs (a table in `inputs/`) |
| `gencode` | v32 · v41 · v46 · v50 · linder2025 | v50 | v50 | GENCODE release for TSS and exon annotation (`linder2025` is the annotation Linder et al. supplied with their tables) |
| `exon_source` | all-transcripts · basic-nort | all-transcripts | all-transcripts | Which transcripts contribute exons to the merged exon union |
| `pair_filters` | mapping | {} | require_gene_in_annotation: true | Which pairs are kept: a mapping with any of `max_tss_distance`, `min_tss_distance`, `min_elements_per_gene`, `require_gene_in_annotation` |
| `out_of_window` | zero · drop | zero | zero | Pairs beyond the model's input window: score zero, or drop them |

## Input window

| parameter | values | default | matched protocol | what it does |
|---|---|---|---|---|
| `anchor` | single-tss · each-tss · gene-midpoint · enhancer · enhancer-tss-midpoint | single-tss | single-tss | Where the input window is centred |
| `tss_major` | true · false | false | false | Re-pick the gene's TSS as the one with the highest predicted signal |
| `tss_major_assay` | cage · procap | unset | unset | Which predicted assay picks it (required when `tss_major` is true) |
| `tss_major_cell_type` | k562 · all | unset | unset | Which cell type's prediction picks it (unset follows `cell_type`) |
| `anchor_agg` | mean · max · sum | unset | unset | How the per-TSS scores combine (required when `anchor` is `each-tss`) |

## Gene readout

| parameter | values | default | matched protocol | what it does |
|---|---|---|---|---|
| `assay` | cage · rnaseq · procap · multi | cage | cage | Which predicted assay is read out |
| `cell_type` | k562 · all · not-k562 · k562-cnhs11250 · k562-encode · rand1-s0…s9 · rand2-s0…s9 | k562 | k562 | Which track set is read out: `k562` the benchmark's own cell type, `all` every sample the model predicts, `rand*` fixed random non-K562 draws |
| `region` | tss · all-tss · exons | tss | tss | Read a window at one TSS, at every TSS, or the gene's exons |
| `tss_width` | 128 · 256 · 384 · 640 | unset | 384 | Width of the TSS readout window in bp (required unless `region` is `exons`) |
| `readout_anchor` | window-centre · gene-tss | window-centre | window-centre | Whether the TSS readout sits at the window's centre or at the gene's TSS (the same under `anchor: single-tss`) |
| `bin_overlap` | any · majority · fractional · fixed | any | any | How a bp interval snaps onto the model's output bins |
| `track_agg` | sum · mean · max | mean | mean | How tracks are combined within a bin |
| `strand_mask` | auto · on · off | auto | auto | Read only the tracks on the gene's strand (`auto`: yes where the model's tracks are stranded) |
| `inverse_transform` | true · false | false | false (Enformer) · true (Borzoi) · false (AlphaGenome) | Undo the model's squashed target scale before reading (needed for Borzoi) |
| `log_scalar` | true · false | true | false | Take the log of the gene readout, which makes the score a relative change per gene (gene-readout normalisation) |

## Attribution method

| parameter | values | default | matched protocol | what it does |
|---|---|---|---|---|
| `method` | gradient · ism · attention | gradient | gradient | Input gradients, in-silico mutagenesis (ISM) or attention |
| `mean_subtract` | true · false | true | true | Gradients: subtract the per-position mean over the four bases |
| `bin_abs_order` | after · before | after | after | Gradients: absolute value before or after the readout-bin sum |
| `track_abs_order` | after · before | after | after | Absolute value before or after the mean over tracks |
| `ism_window` | 500 · 1000 · 2000 · 2048 · 3000 · 4000 | 2000 | 2000 | ISM: width of the perturbed span in bp, centred on the enhancer |
| `ism_mode` | random-probs · zero-replace · random-replace · genomic-replace · mononuc-shuffle · dinuc-shuffle · eulerian-dinuc-shuffle | dinuc-shuffle | dinuc-shuffle | ISM: how the span's sequence is replaced |
| `ism_shuffles` | 1 · 2 · 4 · 5 · 8 · 10 · 16 · 32 · 64 | 10 | 10 | ISM: how many perturbation draws are averaged |
| `ism_statistic` | abs_mean_diff · abs_diff · abs_mean_rel_diff · rel_abs_diff | abs_mean_diff | abs_mean_diff | ISM: which difference between wild type and perturbed readout is the score (more statistics from stored draws: `scripts/ism_statistic_tables.py`) |
| `ism_inner_mean` | any subset of tss, strand, replicate | [] | [] | ISM: which axes (`tss`, `strand`, `replicate`) are averaged with sign before the absolute value |
| `seed` | any integer | 0 | 0 | Seeds the perturbation draws and augmentation shifts |

## From attribution map to pair score

| parameter | values | default | matched protocol | what it does |
|---|---|---|---|---|
| `abs_map` | true · false | true | true | Take the absolute value per nucleotide |
| `normalise` | global · none · median · trim99 · rms · std · p90 · p95 · p99 · p999 · max | global | global | Divide each gene's map by a per-gene statistic of it (gradient-map normalisation); `global` is its mean absolute value |
| `enh_agg` | sum · mean · max · gaussian · point | sum | gaussian | How the map is reduced over the enhancer |
| `enh_span` | 500 · 1000 · 2000 · 3000 | unset | unset | Width of the enhancer window in bp (required for `sum`, `mean`, `max`) |
| `gaussian_sigma` | 147 · 164 · 250 · 268 · 293 · 300 · 329 · 375 · 535 · 586 · 657 · 878 · 985 · 1070 · 1605 | unset | 300 | Sigma of the Gaussian enhancer kernel in bp (required for `gaussian`) |
| `gaussian_truncate` | 1 · 2 · 3 · 4 | unset | 4 | Where that kernel is cut off, in sigma (required for `gaussian`) |

## Augmentation and replicates

| parameter | values | default | matched protocol | what it does |
|---|---|---|---|---|
| `tta` | none · rc · shift · rc+shift · avsec8 | none | none | Test-time augmentation: reverse complement, shifts, both, or Avsec et al. (2021)'s eight random augmentations |
| `tta_shift_bp` | 1 · 2 · 3 · 43 | unset | unset | Shift size in bp (required by any shifting `tta`) |
| `tta_abs_order` | before · after | before | before | Absolute value before or after the mean over augmentations |
| `replicate_abs_order` | before · after | after | before (Enformer) · after (Borzoi) · before (AlphaGenome) | Absolute value before or after the mean over replicates |

## The matched ISM protocol

The paper's ISM runs change these settings of the matched protocol: `method: ism`,
`ism_mode: eulerian-dinuc-shuffle`, `ism_shuffles: 5` (run at `seed` 0 and 1, ten draws in all),
`enh_agg: point` and `normalise: none`. They record the draws per track (`strand_mask: off`,
`track_abs_order: before`, `--ism-draws`), and `scripts/ism_statistic_tables.py --strand-mask auto` then
computes the statistic the paper reads, `log_abs_diff`, from the stored draws. That script also offers
`log_mean_diff`, `signed_mean`, `signed_rel_mean` and `signed_log_mean`, which are not settings.

## Fixed, not parameters

10 further quantities are fixed in `score_eg.py` (the `FROZEN_*` constants near its top) at the value
every run in the paper used, for example summing over readout bins and averaging replicates as scores.
