# Baselines: ABC, ABC* and distance

The paper compares every model with three expert baselines that read no DNA sequence, all scored on the 10,353
pairs of the K562 screen (Gschwind et al.), and reports ABC and ABC* also without their competing-element
denominators. The
five per-pair tables ship in `scores/baselines/`. They regenerate from three small files in
`paper/data/baselines/`, with numpy alone, in about five seconds:

```bash
python3 scripts/baselines/score_baselines.py --check   # recompute and compare with scores/baselines/
python3 scripts/baselines/score_baselines.py           # rewrite scores/baselines/
```

| table | score of a tested element E and gene G |
|---|---|
| `abc` | ABC(E, G) = A_E C_EG / Σ A_E′ C_E′G, the sum over every candidate element E′ whose midpoint lies within 5 Mb of G's TSS. A = √(DNase × H3K27ac), each the candidate's reads per million, quantile-normalised to ABC's K562 reference (ENCODE ENCFF860XAE, ENCFF790GFL). C = K562 Hi-C contact (ENCFF621AIY, 5 kb, SCALE), rescaled to ABC's reference power law (γ = 0.87) and floored by ABC's power-law pseudocount |
| `abcstar` | the same loop with A = ABC's own H3K27ac term (the quantile-normalised H3K27ac factor inside ABC's activity, from the same data, ENCFF790GFL; no DNase) and C = ABC's distance-based contact at γ = 1, that is 1 / (max(d, 5 kb) + 1), over the same candidates |
| `abc_numerator` | A_E C_EG: ABC without its competing-element denominator |
| `abcstar_numerator` | the same for ABC* |
| `distance` | 1 / max(\|d\|, 1), with d = enhancer midpoint − the screen's TSS |

d is the distance from the candidate's midpoint to the TSS. In `abc` and `abcstar` a promoter candidate whose
interval ±500 bp contains the gene's TSS scores 1, ABC's self-promoter rule; the two numerator tables do not apply
it. A tested element takes the maximum over the candidates it overlaps, and 0 if it overlaps none (the same
1,000 pairs for ABC and ABC*). The tables have the
format and row order of `scores/unified-t1/`, so `scripts/evaluate.py` reads them directly; on all pairs it
prints auPRC 0.6396 (ABC), 0.5432 (ABC*), 0.4377 (distance), 0.5219 and 0.4707 (the two numerators), the paper's
values.

## The derived files

| file | rows | columns | what it is |
|---|---|---|---|
| `candidates.tsv.gz` | 143,676 | `chrom start end class activity h3k27ac` | ABC's candidate elements on these inputs, with ABC's element class (`promoter`, `genic`, `intergenic`), ABC's activity, and the H3K27ac term of that activity, which is ABC*'s activity (activity = √(DNase term × `h3k27ac`)) |
| `genes.tsv.gz` | 2,111 | `gene_id gene_symbol chrom tss` | the TSS ABC and ABC* use for each screen gene |
| `hic_contact.tsv.gz` | 370,927 | `gene_id bin contact` | Hi-C contact of the 5 kb bin holding the gene's TSS with bin `bin` (position // 5,000), divided by the chromosome's mean row sum, the TSS bin's own value replaced by the larger of its two neighbours. Only nonzero values at bins that hold a candidate's midpoint within 5 Mb of the TSS; every other bin is 0 |

Floats are written with Python's `repr`, so they parse back to the exact doubles the paper used.

**Where they come from.** The candidates, their classes, their activity and its H3K27ac term are those of the ABC pipeline
([broadinstitute/ABC-Enhancer-Gene-Prediction](https://github.com/broadinstitute/ABC-Enhancer-Gene-Prediction),
commit `ba9a790`) run on ENCFF860XAE and ENCFF790GFL with its default configuration (MACS2 p-value 0.1; the
150,000 strongest peaks by read count, ±250 bp around each summit; the repository's hg38 blocklist, gene bounds and
TSS includelist; K562 quantile normalisation, `use_qnorm: True`). The shipped values come from an independent
re-implementation of those steps: its candidate list is line-identical to the pipeline's, its classes are identical,
and its activity equals the pipeline's `activity_base` to that file's 6-decimal precision (|Δ| ≤ 5 × 10⁻⁷). Scoring
the pipeline's printed activity instead moves ABC by at most 1.9 × 10⁻⁶ relative and changes no auPRC. The TSS is
the `tss` column of the same run's `GeneList.txt` (the repository's `CollapsedGeneBounds.hg38.bed`: start on +, end
on −), matched by gene symbol; it lies 1 bp from the screen's own TSS for every gene. The Hi-C contact and the
scoring follow the pipeline's ENCODE parameterisation (`--scale_hic_using_powerlaw`, `hic_gamma` 1.024238616787792,
`hic_scale` 5.9594510043736655, `hic_pseudocount_distance` 5,000); on the 9,328 pairs the pipeline itself scores,
this ABC agrees with the pipeline's `ABC.Score` at Spearman 0.9996, and their auPRCs differ by at most 0.0001 in
every distance bin.

## Recomputing the derived files from ENCODE

An optional script rebuilds the Hi-C file from public data and exits non-zero unless every value is bit-identical:

```bash
pip install hicstraw
python3 scripts/baselines/abc_contact_from_hic.py ENCFF621AIY.hic    # hic_contact; about 15 min, 1 GB
```

| accession | assay (K562, GRCh38) | feeds | bytes | md5 |
|---|---|---|---|---|
| [ENCFF621AIY](https://www.encodeproject.org/files/ENCFF621AIY/@@download/ENCFF621AIY.hic) | in situ Hi-C (.hic) | `hic_contact` | 33,783,625,697 | `71e5dc072beddbdcda11c9016696e142` |
| [ENCFF860XAE](https://www.encodeproject.org/files/ENCFF860XAE/@@download/ENCFF860XAE.bam) | DNase-seq alignments (BAM) | candidates, `activity` (via the ABC pipeline) | 1,628,064,081 | `2f00a4d60183c1f0e5d6601ef86c6c7a` |
| [ENCFF790GFL](https://www.encodeproject.org/files/ENCFF790GFL/@@download/ENCFF790GFL.bam) | H3K27ac ChIP-seq alignments (BAM) | `activity`, `h3k27ac` (via the ABC pipeline) | 631,541,681 | `0e01bab632295c3c3f6c54d95b3c1438` |

The candidate list, classes, activity, H3K27ac term and TSS are not recomputed by this repository: run the ABC pipeline at the
commit above on the two BAMs, and compare its `Neighborhoods/EnhancerList.txt` and `GeneList.txt` with
`candidates.tsv.gz` and `genes.tsv.gz` (join on coordinates; the pipeline's row order differs).

## Exactness

`score_baselines.py` reproduces every table bit for bit on the platform the paper used (x86-64 with AVX-512, numpy
1.23.5 or 1.26.4). The only transcendental operations are `exp` and `log`, whose last bit depends on the CPU and the
numpy build; elsewhere ABC and ABC* differ from the shipped tables by at most 4 × 10⁻¹⁵ relative, rank all 10,353
pairs identically, ties included, and so leave every auPRC, interval and p-value unchanged. `--check` reports such
a table as "equal to rounding". Tested with numpy 1.17 to 2.5. The distance table is exact everywhere.
