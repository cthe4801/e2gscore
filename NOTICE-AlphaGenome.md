# AlphaGenome outputs in this repository

The files below hold, or are computed from, predictions of Google DeepMind's AlphaGenome (run here through the
`alphagenome-pytorch` port with locally converted weights). They are **AlphaGenome Output Derivatives**, provided
under and subject to the [AlphaGenome Output Terms of Use](http://deepmind.google.com/science/alphagenome/output-terms),
not under this repository's MIT licence. By using them you agree to those terms; see
`LEGALLY_BINDING_TERMS_OF_USE.txt`, and `licenses/AlphaGenome-Output-Terms-of-Use.pdf` for a copy of the terms.
Among other things, the terms permit non-commercial use only and do not permit training machine-learning models
on these files.

**Modifications made to the Output.** None of the files holds raw AlphaGenome predictions. Each score is an
enhancer–gene attribution computed from AlphaGenome's predicted K562 tracks by `score_eg.py` under the protocol in
the file's manifest: an input-gradient or in-silico-mutagenesis (ISM) response of a gene's predicted readout,
pooled over the enhancer and normalised as configured; the ISM tables marked `(derived)` in `reproducing.tsv`
reduce the stored per-shuffle responses to another statistic, and `paper/data/ism_matched_perdraw.npz` holds those
responses per perturbation draw.

## Files

- `scores/avsec2026/avsec2026-alphagenome-gradient-gschwind.scores.tsv.gz`
- `scores/avsec2026/avsec2026-alphagenome-ism-gschwind.scores.tsv.gz`
- `scores/unified-t1/unified-t1-A10g-alphagenome-gene-midpoint-procap-genetssreadout.scores.tsv.gz`
- `scores/unified-t1/unified-t1-A11-alphagenome-gene-midpoint-rnaseq-logscalar.scores.tsv.gz`
- `scores/unified-t1/unified-t1-A12-alphagenome-single-tss-procap-logscalar.scores.tsv.gz`
- `scores/unified-t1/unified-t1-A3-alphagenome-single-tss.scores.tsv.gz`
- `scores/unified-t1/unified-t1-A4g-alphagenome-gene-midpoint-genetssreadout.scores.tsv.gz`
- `scores/unified-t1/unified-t1-A5-alphagenome-single-tss-rnaseq.scores.tsv.gz`
- `scores/unified-t1/unified-t1-A6-alphagenome-gene-midpoint-rnaseq.scores.tsv.gz`
- `scores/unified-t1/unified-t1-A8-alphagenome-single-tss-procap.scores.tsv.gz`
- `scores/unified-t1/unified-t1-A9-alphagenome-single-tss-logscalar.scores.tsv.gz`
- `scores/unified-t1/unified-t1-G3-alphagenome-single-tss-ism.scores.tsv.gz`
- `scores/unified-t1/unified-t1-G3F-alphagenome-gene-midpoint-rnaseq-tssnull-ism-eulerian.scores.tsv.gz`
- `scores/unified-t1/unified-t1-G3FSTAT-A-abs_diff.scores.tsv.gz`
- `scores/unified-t1/unified-t1-G3FSTAT-A-abs_mean_diff.scores.tsv.gz`
- `scores/unified-t1/unified-t1-G3FSTAT-A-abs_mean_rel_diff.scores.tsv.gz`
- `scores/unified-t1/unified-t1-G3FSTAT-A-log_abs_diff.scores.tsv.gz`
- `scores/unified-t1/unified-t1-G3FSTAT-A-log_mean_diff.scores.tsv.gz`
- `scores/unified-t1/unified-t1-G3FSTAT-A-rel_abs_diff.scores.tsv.gz`
- `scores/unified-t1/unified-t1-G3P-alphagenome-single-tss-procap-ism-eulerian.scores.tsv.gz`
- `scores/unified-t1/unified-t1-G3PSTAT-A-abs_diff.scores.tsv.gz`
- `scores/unified-t1/unified-t1-G3PSTAT-A-abs_mean_diff.scores.tsv.gz`
- `scores/unified-t1/unified-t1-G3PSTAT-A-abs_mean_rel_diff.scores.tsv.gz`
- `scores/unified-t1/unified-t1-G3PSTAT-A-log_abs_diff.scores.tsv.gz`
- `scores/unified-t1/unified-t1-G3PSTAT-A-log_mean_diff.scores.tsv.gz`
- `scores/unified-t1/unified-t1-G3PSTAT-A-rel_abs_diff.scores.tsv.gz`
- `scores/unified-t1/unified-t1-M1-alphagenome-ism-eulerian-dinuc.scores.tsv.gz`
- `scores/unified-t1/unified-t1-M1A-seed1-alphagenome-ism-eulerian-dinuc.scores.tsv.gz`
- `scores/unified-t1/unified-t1-M1STAT10A-abs_diff.scores.tsv.gz`
- `scores/unified-t1/unified-t1-M1STAT10A-abs_mean_diff.scores.tsv.gz`
- `scores/unified-t1/unified-t1-M1STAT10A-abs_mean_rel_diff.scores.tsv.gz`
- `scores/unified-t1/unified-t1-M1STAT10A-log_abs_diff.scores.tsv.gz`
- `scores/unified-t1/unified-t1-M1STAT10A-log_mean_diff.scores.tsv.gz`
- `scores/unified-t1/unified-t1-M1STAT10A-rel_abs_diff.scores.tsv.gz`
- `scores/unified-t1/unified-t1-M2-alphagenome-ism-mononuc.scores.tsv.gz`
- `scores/unified-t1/unified-t1-M3-alphagenome-ism-random-replace.scores.tsv.gz`
- `scores/unified-t1/unified-t1-M4-alphagenome-ism-zero-replace.scores.tsv.gz`
- `scores/unified-t1/unified-t1-M5-alphagenome-ism-genomic-replace.scores.tsv.gz`
- `scores/unified-t1/unified-t1-NA1-alphagenome-single-tss-agnostic.scores.tsv.gz`
- `scores/unified-t1/unified-t1-UA0-alphagenome-single-tss-rand1-s0.scores.tsv.gz`
- `scores/unified-t1/unified-t1-UA1-alphagenome-single-tss-rand1-s1.scores.tsv.gz`
- `scores/unified-t1/unified-t1-UA2-alphagenome-single-tss-rand1-s2.scores.tsv.gz`
- `scores/unified-t1/unified-t1-UA3-alphagenome-single-tss-rand1-s3.scores.tsv.gz`
- `scores/unified-t1/unified-t1-UA4-alphagenome-single-tss-rand1-s4.scores.tsv.gz`
- `scores/unified-t1/unified-t1-UA5-alphagenome-single-tss-rand1-s5.scores.tsv.gz`
- `scores/unified-t1/unified-t1-UA6-alphagenome-single-tss-rand1-s6.scores.tsv.gz`
- `scores/unified-t1/unified-t1-UA7-alphagenome-single-tss-rand1-s7.scores.tsv.gz`
- `scores/unified-t1/unified-t1-UA8-alphagenome-single-tss-rand1-s8.scores.tsv.gz`
- `scores/unified-t1/unified-t1-UA9-alphagenome-single-tss-rand1-s9.scores.tsv.gz`
- `paper/data/ism_matched_perdraw.npz` (its AlphaGenome arms)
- the manifests beside the tables above (`*.manifest.json`)
