# Data

## The CRISPRi screens

`inputs/` holds the pair table of each screen, with the binary outcome the
evaluator needs, and a build summary per reproduction recording what the
reconstruction dropped.

| screen | pairs | used for |
|---|---|---|
| `gschwind` | 10,353 | the K562 benchmark every main figure uses |
| `fulco2019` | 4,777 | reproducing Avsec et al. (2021) |
| `gasperini2019` | 38,582 | reproducing Avsec et al. (2021) |
| `flowfish-plus` | 4,793 | the reconstruction of Linder et al. (2025)'s FlowFISH+ table |
| `gasperini2019-linder` | 78,309 | the reconstruction of Linder et al. (2025)'s Gasperini table |

Linder et al. (2025) also provided, on request, the preprocessed pair tables and
the gene annotation behind their Fig. 4c,d and Supplementary Fig. 7a. They ship
as received in `inputs/linder2025-authors/` (its README gives each file's sha256
and the derivation rules) and as two further screens the scorer reads directly.
The paper's Linder et al. (2025) reproduction uses these:

| screen | pairs | annotation |
|---|---|---|
| `flowfish-plus-authors` | 2,194 | `gencode: linder2025` |
| `gasperini2019-linder-authors` | 19,104 | `gencode: linder2025` |

A screen table has one row per tested pair, with the columns `screen`, `chrom`, `enhancer_start`,
`enhancer_end`, `enhancer_mid`, `enhancer_id`, `gene_symbol`, `gene_id`, `tss`, `distance`, `significant` (the
binary outcome), `effect_size` and `adjusted_p`; some tables carry a few more. `score_eg.py` reads the seven
tables its `screen` parameter names (`SCREEN_TABLES` in `score_eg.py` maps each name to its file).

## Sources

| data | source |
|---|---|
| `inputs/gschwind.hg38.tsv.gz`, `paper/data/epcrispr_benchmark_K562.tsv.gz` | the K562 CRISPRi benchmark of Gschwind et al., as released in [EngreitzLab/CRISPR_comparison](https://github.com/EngreitzLab/CRISPR_comparison) v1.0.0 (screens of Gasperini et al. 2019, Nasser et al. 2021 and Schraivogel et al. 2020) |
| `inputs/fulco2019.hg38.tsv.gz` | Fulco et al. (2019), *Nature Genetics*, supplementary tables, lifted to hg38 |
| `inputs/gasperini2019.hg38.tsv.gz`, `inputs/gasperini2019-linder.hg38.tsv.gz` | Gasperini et al. (2019), *Cell*, GEO GSE120861 |
| `inputs/flowfish-plus.hg38.tsv.gz` | Nasser et al. (2021), *Nature*, CRISPRi-FlowFISH data |
| `inputs/linder2025-authors/` and the two `*-authors` tables | provided on request by the authors of Linder et al. (2025), *Nature Genetics* |
| `data/` | GENCODE releases 32, 41, 46 and 50 (TSS and exon annotations derived from them) |
| `paper/data/baselines/`, `scripts/baselines/` | ENCODE K562 DNase-seq ENCFF860XAE, H3K27ac ENCFF790GFL, Hi-C ENCFF621AIY; the ABC pipeline (Fulco et al. 2019; Gschwind et al.) |
| `paper/data/*_values.json`, `avsec2026_targets.json` | the published values of Avsec et al. (2021), Linder et al. (2025) and Avsec et al. (2026), read from their papers |

Each source keeps its own terms; please cite the original studies when you use
their data.

The Eulerian dinucleotide shuffle in `score_eg.py` is vendored verbatim from
`kundajelab/deeplift` and carries its upstream licence; it supplies
`ism_mode: eulerian-dinuc-shuffle`. Benchmark tables in `inputs/` are
reconstructed from the public releases of the screens they name (the paper's
appendix quantifies the agreement with the published pair counts); the two
`*-authors` tables and the `linder2025` annotation are the Linder et al. (2025)
authors' own files, provided on request.
