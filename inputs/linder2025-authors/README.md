# The Linder et al. (2025) authors' own tables

The four files in this folder are the preprocessed CRISPRi pair tables and the
gene annotation behind Fig. 4c,d and Supplementary Fig. 7a of Linder et al.
(2025), provided by the authors on request and kept here byte for byte as
received.

| file | what it is | sha256 |
|---|---|---|
| `flowfish_table.tsv.gz` | the FlowFISH+ pair table (2,194 rows, 117 positives, 53 genes) | `6643e1c7bc3073ae57f8dfa1347ab252c3de111d44bb7955f28b56dc39d32152` |
| `flowfish_genes.gtf.gz` | its `crispr_genes.gtf` | `b86bcf8820fb48a13105fd7e17589d5cac3959386065b29d8e705c1b9e7c10f3` |
| `gasperini_table.tsv.gz` | the Gasperini pair table (19,104 rows, 404 positives, 4,776 genes) | `43a1632dbda5a07d276361e9820cd8c52af20047162b7c074eb29d816d15b726` |
| `gasperini_genes.gtf.gz` | its `crispr_genes.gtf` | `22adc0c252713f37ae3fdd3558e2df51429c29077486bb98995ee7fc1ef5aec5` |

The `crispr_genes.gtf` files are what the authors' gradient script and scoring
notebooks read: they define the gene whose gradient is taken, the window centre
(the midpoint of the merged exons of every transcript in the file) and the exon
bins the RNA-seq readout sums. They are not a GENCODE v41 subset, so the two are
registered together as their own annotation, `gencode: linder2025`
(`data/gencode_linder2025_tssdefs.json.gz`, `data/gencode_linder2025_exon_spans.json.gz`),
read with `exon_source: all-transcripts`.

The scorer reads the tables as `inputs/flowfish-plus-authors.hg38.tsv.gz` and
`inputs/gasperini2019-linder-authors.hg38.tsv.gz`, which are these tables with
the framework's column names and nothing else changed: row order and every
label are the files' own; `gene_id` is resolved from the annotation by symbol
(the last id in file order when a symbol carries two, which is what the
authors' notebooks do); `distance` is the table's `tss_distance` verbatim, and
the TSS column is the point that distance implies. No cap and no
minimum-elements filter is applied, because the supplied tables already carry
both. `inputs/linder2025_authors_build_summary.json` records these rules, the
sha256 of every input and output, and the per-distance-bin pair counts beside
the published ones, which agree exactly in the two shortest bins and within
three pairs in the two longest.
