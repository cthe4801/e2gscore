#!/usr/bin/env python3
"""
Run one enhancer-gene scoring experiment.

An enhancer-gene score is not a number a trunk emits: it is an algorithm
with 42 parameters wrapped around a sequence-to-function model, and this
file is that algorithm. Every procedure carries the name it has in the
paper's pseudocode, and a `# A.<n>` comment marks the pseudocode line a
statement implements.

This module emits per-pair scores. It does not evaluate them;
`scripts/evaluate.py` does.

    python3 score_eg.py --self-test
    python3 score_eg.py --config <cfg> --print-config
    python3 score_eg.py --config <cfg> --out scores.tsv

One process scores exactly one (screen, model, method) combination: a
trunk occupies the whole accelerator, so experiments are queued rather
than interleaved.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field, fields as _dc_fields
from types import SimpleNamespace
from functools import lru_cache
from typing import (Any, Callable, Dict, FrozenSet, Iterable, List, Optional,
                    Sequence, Tuple)

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))

ABSENT = None      # an enhancer that falls outside the input window
LOG_EPS = 1e-6     # guards log(0) in log(total + pseudocount + eps)

#: (shard_index, n_shards), set by `--shard i/n`. Deliberately not a
#: Param and not part of the cache key: which device computed a row is a
#: runtime fact rather than a scientific choice, and not a property of
#: the score. `n = 1` is the unsharded path, and is byte-identical to
#: having no sharding code at all.
SHARD: Tuple[int, int] = (0, 1)

#: Gene IDs to score, set by `--only-genes A,B`; `None` scores the whole
#: cohort. It exists so that scoring part of a cohort costs only those
#: genes rather than a whole arm: a score is a pure function of (gene,
#: enhancer, theta), because `normalise = global` divides by the mean of
#: the gene's OWN map and never by a cohort-wide statistic, and
#: `distance_to_tss` reads the screen's own TSS column. Scoring one gene
#: on its own and splicing its rows into an existing table is therefore
#: exact rather than an approximation. `i` stays the GLOBAL gene rank, so
#: a restricted run's rows sort into the unsharded order.
ONLY_GENES: Optional[FrozenSet[str]] = None

#: With `--ism-draws PATH`, the raw per-shuffle signed differences
#: accumulate here and are written at the end of the run; `None` records
#: nothing. Keeping them leaves every `ism_statistic` and every
#: `ism_inner_mean` subset recoverable after the fact, where recomputing
#: any one of them would cost another full ISM wave.
ISM_DRAWS: Optional[List[Dict[str, Any]]] = None

#: THE READOUT UNDER A REVERSE-COMPLEMENT AUGMENTATION. `attribution_map` and `ism_map` reverse-complement
#: the INPUT, so on that pass the model's output bin b is genomic bin (n_bins - 1 - b) and its '+' tracks
#: carry the '-' strand. The gene's readout must be read in that frame: the mirrored bins, on the opposite
#: strand's tracks (calico's `gene_slice = target_length - gene_slice - 1` plus the swapped strand mask).
#: Read in the forward frame instead, an rc pass would read the gene's mirror image on the antisense
#: tracks: severe for a stranded readout (Borzoi RNA-seq@exons), mild for an unstranded one (Enformer
#: CAGE@TSS: the 3-bin readout would sit one 128 bp bin off). `frame_for` is the only place an
#: augmentation becomes a readout frame, and the frame guard below makes any other reading impossible.


# ============================================================================
# 1. The parameter schema
# ============================================================================

@dataclass(frozen=True)
class Param:
    """One parameter: its legal values, default and how it is read."""
    choices: Optional[Tuple[Any, ...]]  # None => free-form; see kind
    default: Any
    kind: str = "choice"                # choice int float bool set dict
    members: Tuple[str, ...] = ()       # for kind == "set"
    note: str = ""                      # prose for the reader


#: Quantities the algorithm admits but no reported run varied, pinned
#: here rather than exposed as parameters. Each carries the value those
#: runs used, which is not always the nominal default: no config sets
#: `precision_policy`, whose nominal default is `native`, and pinning
#: it there would hard-error against the backend guards.
FROZEN_BIN_AGG = "sum"                       # every paper sums them
FROZEN_ATTENTION_AGGREGATION = "mean-heads-layers"  # Avsec 2021's mean
FROZEN_ATTENTION_QUERY_WIDTH = None          # follows tss_width
FROZEN_ISM_DRAW_SCOPE = "per-pair"           # every pair draws its own
FROZEN_PRECISION_POLICY = "port"             # never native weights
FROZEN_REPLICATE_AGGREGATION = "score-mean"  # scores, not predictions
FROZEN_USE_TF_GAMMA = True                   # the port's own default
FROZEN_PSEUDOCOUNT = 0.0                     # the Borzoi default
FROZEN_MAP_PRECISION = "float32"             # fp16 is a null
FROZEN_ISM_WINDOW_MODE = "center"            # Avsec 2021's reading


PARAMS: Dict[str, Param] = {
    # ---- model -------------------------------------------------------------
    "model": Param(("enformer", "borzoi", "alphagenome"), "enformer"),
    "replicates": Param((1, 4), 1, kind="int"),
    "screen": Param(("fulco2019", "flowfish-plus", "gasperini2019",
                     "gasperini2019-linder", "gschwind",
                     # the Linder et al. (2025) authors' own tables, loaded
                     # as shipped (inputs/linder2025-authors/README.md)
                     "flowfish-plus-authors", "gasperini2019-linder-authors"),
                    "gschwind"),
    # `linder2025` is not a GENCODE release: it is the gene annotation the
    # Linder et al. (2025) authors supplied with their tables, the file their
    # gradient script reads. It is registered as its own release because it
    # is not a subset of v41. Only the *-authors screens are meant to run
    # under it.
    "gencode": Param(("v32", "v41", "v46", "v50", "linder2025"), "v50"),
    "anchor": Param(("single-tss", "each-tss", "gene-midpoint",
                     "enhancer", "enhancer-tss-midpoint"), "single-tss"),
    "tss_major": Param((True, False), False, kind="bool"),
    "tss_major_assay": Param(("cage", "procap"), None),
    "tss_major_cell_type": Param(
        ("k562", "all"), None,
        note="which cell type's predicted assay DEFINES the major TSS, "
             "as distinct from which one is READ OUT. Null follows "
             "`cell_type`, which is right whenever the two coincide. "
             "They do not coincide for the cell-type-agnostic arm: "
             "there the readout widens to all 638 CAGE tracks while "
             "the gene's TSS stays the argmax of predicted K562 CAGE. "
             "Conflating the two would move the anchor as well as the "
             "readout, and that arm measures the cost of widening the "
             "readout only if everything else is held fixed."),
    "exon_source": Param(
        ("all-transcripts", "basic-nort"), "all-transcripts",
        note="WHICH transcripts contribute exons to the merged union "
             "that `region = exons` reads and `EXON-CENTROID` anchors "
             "on. `all-transcripts` takes every `exon` line of the "
             "release; `basic-nort` takes only the basic / "
             "no-read-through subset that the Borzoi release fetches "
             "for gene scoring, which is about half the exon records "
             "of the release."),
    "pair_filters": Param(None, {}, kind="dict"),
    "out_of_window": Param(("zero", "drop"), "zero"),

    # ---- gene readout ------------------------------------------------------
    "inverse_transform": Param((True, False), False, kind="bool"),
    # `multi` is the four-assay K562 promoter readout: CAGE, DNase,
    # H3K4me3 and RNAPII. H3K27ac, the canonical active-promoter mark,
    # has no K562 track in Enformer, so the readout is four assays and
    # not five. The per-assay grouping is ENFORMER_MULTI_GROUPS.
    "assay": Param(("cage", "rnaseq", "procap", "multi"), "cage"),
    "cell_type": Param(
        ("k562", "all", "not-k562",
         "k562-cnhs11250", "k562-encode",
         # The ten random-2-sample draws, for Enformer and Borzoi. The
         # names resolve per model, exactly as `k562` does, and the
         # Borzoi draws are the same FANTOM samples as the Enformer
         # ones by design; see BORZOI_CAGE_RAND2_ORDINALS.
         *(f"rand2-s{i}" for i in range(10)),
         # The ten random-1-sample draws, AlphaGenome only. Size 1 and
         # not 2 because AlphaGenome's K562 CAGE is one biological
         # sample where Enformer's and Borzoi's is two: the draw is
         # pinned to the K562 anchor it is contrasted against, so a
         # size-2 draw would compare 2-sample noise against a 1-sample
         # anchor. See ALPHAGENOME_CAGE_RAND1_ORDINALS.
         *(f"rand1-s{i}" for i in range(10))), "k562",
        note="names a TRACK SET, not a biological sample: the resolver "
             "looks up (assay, cell_type) in ModelMeta.tracks. The two "
             "singleton K562 keys isolate one FANTOM library each, so "
             "that a single-track readout is a configuration rather "
             "than a separate script."),
    "strand_mask": Param(("auto", "on", "off"), "auto"),
    "region": Param(
        ("tss", "all-tss", "exons"), "tss",
        note="`tss` reads tss_width bp at ONE anchor bin. `all-tss` "
             "reads the same window at EVERY annotated TSS that the "
             "model predicts, de-duplicated: the readout half of what "
             "`anchor = each-tss` does, without re-anchoring the input "
             "window, so it costs one pass instead of one per TSS. "
             "`exons` reads the mature transcript body."),
    "tss_width": Param(
        (128, 256, 384, 640), None, kind="int",
        note="Avsec 2021 reads THREE adjacent bins, so 384 is the "
             "paper's value for Enformer. 128, a single bin, is "
             "included because the attention arm's query width is "
             "genuinely ambiguous in the text: 'CAGE at TSS' names a "
             "3-bin READOUT, and whether the attention query is "
             "correspondingly 3 tokens wide or 1 is never stated."),
    "readout_anchor": Param(
        ("window-centre", "gene-tss"), "window-centre",
        note="WHERE the `region = tss` readout is placed. READOUT-BINS "
             "defines it positionally, at BIN-OF(centre of w), so it "
             "follows the anchor. That is invisible while the anchor "
             "is gene-derived -- single-tss, each-tss and gene-midpoint "
             "all centre on a gene feature -- and it breaks for the "
             "two pair-derived anchors: at `anchor = enhancer` the "
             "readout lands on the enhancer and the score stops "
             "depending on the gene at all, and at "
             "`anchor = enhancer-tss-midpoint` it lands on neither "
             "feature. `gene-tss` pins the readout to the gene's own "
             "TSS, so the anchor moves the window alone. A no-op for "
             "`single-tss` without `tss_major`, where the window "
             "centre IS the gene's given TSS."),
    "bin_overlap": Param(
        ("any", "majority", "fractional", "fixed"), "any",
        note="how a bp interval maps onto a coarse grid. The first "
             "three follow the element and so vary with grid phase: a "
             "2 kb window on Enformer's 128 bp grid covers 16 or 17 "
             "tokens under `any` and 15 or 16 under `majority`, while "
             "`fractional` is phase-independent by interpolating "
             "15.625 tokens rather than by fixing a count. `fixed` is "
             "the fourth reading -- whole token values AND a constant "
             "count -- taking round(width / grid) tokens centred on "
             "the element."),
    "track_agg": Param(("sum", "mean", "max"), "mean"),
    "log_scalar": Param((True, False), True, kind="bool"),
    "method": Param(("gradient", "ism", "attention"), "gradient"),
    "mean_subtract": Param((True, False), True, kind="bool"),
    "bin_abs_order": Param(
        ("after", "before"), "after",
        note="where |.| falls relative to the readout-bin sum, for "
             "gradients only. Avsec 2021: \"'CAGE at TSS' always means "
             "summing the absolute gradient values from three adjacent "
             "bins\" -- word order says abs-then-sum (`before`), "
             "while \"as is also done in gene-focused model "
             "evaluation\", where the 3-bin sum is over PREDICTIONS, "
             "says sum-then-abs (`after`). `before` costs one backward "
             "pass per bin."),
    "track_abs_order": Param(
        ("after", "before"), "after",
        note="where |.| falls relative to the CROSS-TRACK reduction, "
             "the exact analogue of `bin_abs_order` one axis over. "
             "`after` scores |mean_P mean_T d_T|: the tracks are "
             "averaged into one gene scalar and the difference is "
             "taken once. `before` scores mean_T |mean_P d_T|, so each "
             "track gets its own signed difference and absolute value "
             "before the average, and tracks that respond in OPPOSITE "
             "directions reinforce instead of cancelling. The two "
             "coincide for a single track. Implemented for "
             "`method = ism` only: the per-track readout returns a "
             "vector, which the gradient arm cannot differentiate."),
    "tta_abs_order": Param(
        ("before", "after"), "before",
        note="where |.| falls relative to the TEST-TIME-AUGMENTATION "
             "mean, the third instance of the `bin_abs_order` question "
             "and the one the Linder 2025 Borzoi arm forces. `before` "
             "absolutises each augmentation's map inside REDUCE-PAIR "
             "and averages the resulting SCALARS in COMBINE, "
             "mean_a |M_a|; `after` averages the signed MAPS first and "
             "absolutises once, |mean_a M_a|. Linder 2025 ran the "
             "second: the released gradient runner accumulates the "
             "forward and reverse-complement gradients into one array, "
             "and the absolute value only arrives in the benchmark "
             "script. The two differ wherever an orientation flips the "
             "sign of a nucleotide's gradient, which is exactly where "
             "the signal is weak. Free: the maps are unchanged, so "
             "this is not in GPU-SUBSET."),
    "replicate_abs_order": Param(
        ("before", "after"), "after",
        note="where |.| falls relative to the REPLICATE mean, the "
             "fourth instance of the `bin_abs_order` question and the "
             "one Linder 2025's Fig 4c,d turns on. `before` scores "
             "each replicate and averages the scalars, which is "
             "`replicate_aggregation = score-mean`; `after` averages "
             "the signed MAPS across replicates and absolutises once. "
             "Linder ran `after`: the aggregation notebook sums the "
             "per-replicate gradient maps and divides by the replicate "
             "count, and |.| only arrives later in the benchmark "
             "script. Unlike the other three this one has a MECHANISM "
             "for its direction -- averaging signed maps cancels "
             "replicate-specific noise and leaves the shared signal, "
             "so the enhancer numerator sharpens against the whole-map "
             "normaliser. Inert at `replicates = 1`, where it warns "
             "rather than refusing. Free: the per-replicate maps are "
             "cached either way."),
    "abs_map": Param((True, False), True, kind="bool"),
    # `global`, the map mean, is the only divisor any published protocol
    # uses. The rest are the DIVISOR_KINDS ladder: legal, so that an arm
    # can be run at one directly, though the `nu_*` output columns
    # already make every one of them recoverable after the fact at every
    # pooling width.
    "normalise": Param(("global", "none", "median", "trim99", "rms", "std",
                        "p90", "p95", "p99", "p999", "max"), "global"),
    "enh_agg": Param(("sum", "mean", "max", "gaussian", "point"), "sum"),
    # The gaussian and the flat-window families are matched on width, so
    # that `enh_agg` tests kernel SHAPE rather than confounding shape
    # with width. A flat window of full width W has standard deviation
    # W/sqrt(12); a gaussian truncated at t*sigma has sigma*c(t), with
    # c(2) = 0.880. Equating the two at t = 2 gives sigma = W/3.047,
    # which is where the sigmas paired with W = 500, 1000, 2000 and 3000
    # come from: four widths spanning 6x.
    "enh_span": Param((500, 1000, 2000, 3000), None, kind="int"),
    "gaussian_sigma": Param((147, 164, 250, 268, 293, 300, 329, 375, 535, 586,
                             657, 878, 985, 1070, 1605), None, kind="int"),
    "gaussian_truncate": Param((1, 2, 3, 4), None, kind="int"),

    # ---- in silico mutagenesis --------------------------------------------
    # 2048, not 2000, is what Linder 2025 ran: the released ISM runners
    # pass `--window_size 2048`. The paper says "a 2 kb window"; the
    # code uses the neighbouring power of two.
    "ism_window": Param((500, 1000, 2000, 2048, 3000, 4000), 2000, kind="int"),
    "ism_mode": Param(
        ("random-probs", "zero-replace", "random-replace",
         "genomic-replace", "mononuc-shuffle", "dinuc-shuffle",
         "eulerian-dinuc-shuffle"), "dinuc-shuffle",
        note="Avsec 2021 says `random-replace` -- \"the 2-kb enhancer "
             "sequence was replaced by a random sequence\" -- and that "
             "literal reading stands. `random-probs` replaces the "
             "channel VALUES rather than the sequence, which is "
             "off-manifold. The modes differ mainly in draw noise "
             "rather than in which one is right. See PERTURB."),
    # The draws NEST: `stable_seed` keys on (gene, enhancer) and not on
    # the draw index, so the first four draws of P = 10 are
    # byte-identical to P = 4. A ladder therefore measures the lift-vs-P
    # curve without recomputing anything, which is why it is enumerated
    # here rather than left free. The ladder is the powers of two plus
    # the published 10 and its half, so that topping a five-draw run up
    # by five more lands on 10 rather than on 8 or 12.
    "ism_shuffles": Param((1, 2, 4, 5, 8, 10, 16, 32, 64), 10, kind="int"),
    "ism_statistic": Param(("abs_mean_diff", "abs_diff", "abs_mean_rel_diff",
                            "rel_abs_diff"), "abs_mean_diff"),
    "ism_inner_mean": Param(
        None, (), kind="set", members=("tss", "strand", "replicate"),
        note="signed averaging BEFORE the absolute value"),

    # ---- combining ---------------------------------------------------------
    "anchor_agg": Param(("mean", "max", "sum"), None),
    "tta": Param(
        ("none", "rc", "shift", "rc+shift", "avsec8"), "none",
        note="the first four are DETERMINISTIC enumerations, identical "
             "for every input sequence: {rc?} x {-k, 0, +k}. `avsec8` "
             "is Avsec 2021's own test-time augmentation and the only "
             "value that is RANDOM: eight i.i.d. draws of "
             "(shift ~ U{-k..k}, rc ~ Bern(1/2)) per input sequence, "
             "seeded from (`seed`, gene) so that the draw is a pure "
             "function of the gene. See AVSEC8_N for the quotation and "
             "`avsec8_seed` for why the seed is derived rather than "
             "streamed."),
    "seed": Param(None, 0, kind="int"),

    # 43 is NOT a published value, and its provenance matters because
    # the sources disagree. Avsec 2021 states eight randomly augmented
    # sequences with shifts of at most 3 bp. Karollus 2023 uses a
    # deterministic six instead, "for both strands and with small
    # offsets respectively", and never states the magnitude. The
    # (-43, 0, +43) reading is a downstream reconstruction: Enformer's
    # bin is 128 bp, and 128/3 ~ 43 samples that bin at even thirds. It
    # is a legal value because that rationale holds on its own terms --
    # 3 bp is 2.3% of a 128 bp bin, too little to average over bin
    # phase, which is the stated purpose -- but an arm using it tests a
    # proposal rather than reproducing a protocol.
    "tta_shift_bp": Param(
        (1, 2, 3, 43), None, kind="int",
        note="1-3 are Avsec 2021's stated TTA range; 43 = 128/3 is the "
             "bin-thirds offset, unstated in Karollus 2023 and larger "
             "than Avsec allows."),
}

RELATIVE_STATISTICS = ("abs_mean_rel_diff", "rel_abs_diff")

#: Readout scheme: assay and region are ONE axis, not two independent
#: ones. Their cross product spans four cells and two are biologically
#: meaningless: CAGE@exons measures 5'-cap initiation over a body that
#: has no cap, and RNA-seq@TSS measures steady-state abundance in a
#: 384 bp window that is mostly first exon by accident. Sweeping the two
#: as independent axes would spend compute on those cells and, worse,
#: report them as though they were interpretable protocol variants.
#:
#: All three reproduced papers obey the pairing independently:
#:     Avsec 2021  (Enformer)     cage   @ tss
#:     Linder 2025 (Borzoi)       rnaseq @ exons
#:     Avsec 2026  (AlphaGenome)  rnaseq @ exons
#: so enforcing it removes only combinations nobody has argued for.
#:
#: Enforced only under `_protocol: unified-t1`. The reproduction
#: protocols must stay free to be wrong: their job is to mirror a paper,
#: and a future paper is entitled to a readout this table calls
#: incoherent.
READOUT_SCHEMES: Dict[str, str] = {
    "cage": "tss",       # 5' cap initiation -> the TSS window
    "procap": "tss",     # nascent 5' initiation -> the TSS window
    "rnaseq": "exons",   # steady-state abundance -> the transcript body
    "multi": "tss",      # four promoter-proximal assays -> TSS window
}

#: Regions that are a TSS readout, so `READOUT_SCHEMES` treats them
#: interchangeably. `all-tss` is `tss` evaluated at every annotated TSS
#: instead of one, which changes HOW MUCH of the promoter region is
#: summed, not WHICH biology is read: a CAGE arm may use either, and the
#: assay/region pairing must not reject it as if it were CAGE@exons.
TSS_REGIONS = ("tss", "all-tss")


# ============================================================================
# 2. Model metadata
# ============================================================================

@dataclass(frozen=True)
class ModelMeta:
    """One trunk's fixed geometry and its track inventory.

    Everything here is a property of the released model rather than of
    a configuration: `resolve` reads it, nothing writes it.
    """
    name: str
    seq_len: int
    bin_size: int
    n_bins: int
    has_attention: bool
    scalar_is_squashed: bool          # => inverse_transform is meaningful
    transform_generation: str         # "old" | "new" | "none"
    log_base: str                     # "e" | "2" | "none"
    has_strand_column: bool
    # (assay, cell_type) -> the tracks that readout reads
    tracks: Dict[Tuple[str, str], Tuple[int, ...]]
    #: Strand per track, keyed by the SAME integers that appear in
    #: `tracks`. Safe only while those integers are globally unique,
    #: which holds for Enformer and Borzoi because theirs are absolute
    #: head indices: BORZOI_META carries ("cage", "k562") and
    #: ("rnaseq", "k562") in one dict with no collision. It does NOT
    #: hold for a backend that returns a positional SUBSET of columns,
    #: where every track set is numbered 0..n-1 and position 1 names a
    #: different track in each. Use `track_strand_by_set` there.
    track_strand: Dict[int, str] = field(default_factory=dict)
    #: (assay, cell_type) -> {track -> strand}, taking precedence over
    #: `track_strand` for that set. AlphaGenome needs it:
    #: `backend_alphagenome._check_track_contract` requires
    #: `tracks[key] == range(len(absolute_tracks))`, so its CAGE set is
    #: forced to positions (0, 1) -- the same positions its RNA-seq set
    #: uses, where 1 means "+". Under the flat dict a '-' gene would
    #: keep CAGE position 1 and silently read the '+' track.
    track_strand_by_set: Dict[Tuple[str, str], Dict[int, str]] = field(
        default_factory=dict)
    # (assay, cell_type) -> (scale, exponent, clip_soft)
    transform_constants: Dict[
        Tuple[str, str], Tuple[float, float, Optional[float]]] = field(
            default_factory=dict)

    @property
    def crop_bins(self) -> int:
        """Bins trimmed from each side of the prediction.

        This is the offset between the input-bin grid and the
        output-bin grid, because these models predict only the CENTRE
        of their input. Enformer takes 196,608 bp = 1,536 input bins
        and emits 896 bins = 114,688 bp, so it discards
        (1536-896)/2 = 320 bins = 40,960 bp per side; Borzoi's crop
        depends on which parameter file a run used, as BORZOI_META
        records; AlphaGenome discards nothing. Derived rather than
        stored, so it cannot drift out of step with seq_len, bin_size
        and n_bins.

        Confusing the two grids reads the target signal a crop width
        away from where it belongs, silently, because the wrong index
        is still in range. See `bin_of`.
        """
        return (self.seq_len // self.bin_size - self.n_bins) // 2


# Enformer: 196,608 bp in, 896 x 128 bp bins out, 5,313 human tracks.
# CAGE:K562 is two unstranded outputs: 4828 is the library CNhs11250,
# and 5111 is the ENCODE replicate group CNhs12334 + 12335 + 12336
# collapsed into one output. Basenji2 targets carry no strand column, so
# the strand-mask question is void here.
ENFORMER_META = ModelMeta(
    name="enformer", seq_len=196_608, bin_size=128, n_bins=896,
    has_attention=True, scalar_is_squashed=True, transform_generation="none",
    log_base="e", has_strand_column=False,
    tracks={("cage", "k562"): (4828, 5111),
            ("cage", "all"): tuple(range(4675, 5313)),
            # The CAGE block minus K562's own two: every other cell
            # type, 636 tracks. `cell_type: all` averages K562 in with
            # the rest and costs auPRC; this set contrasts against the
            # same tracks instead, which is what separates cell-type
            # specificity from track count. Same exclusion the rand2-s*
            # draws use, so the three sets are comparable.
            ("cage", "not-k562"): tuple(
                i for i in range(4675, 5313) if i not in (4828, 5111)),
            # ---- the multi-assay K562 promoter readout -----------------
            # Four assays, 26 tracks, declared as ONE set so `resolve`
            # hands the backend every index it must compute. The
            # per-assay grouping lives in ENFORMER_MULTI_GROUPS, because
            # the readout logs each assay separately before summing and
            # so needs the split.
            #
            # K562 H3K27ac does not exist: Enformer carries 229 H3K27ac
            # tracks across other cell types and none for K562, under
            # any spelling of the name. The canonical active-promoter
            # mark is unavailable for the cell line this benchmark is
            # built on, so the arm runs four assays rather than five.
            # That is a gap in the model's coverage, not a choice.
            ("multi", "k562"): (
                4828, 5111,                                    # CAGE     x2
                33, 34, 35, 121, 122, 123, 625,                # DNase    x7
                701, 1210, 1387, 3507,                         # H3K4me3  x4
                952, 979, 1341, 1364, 1367, 1427, 1428, 1429,  # RNAPII  x13
                1542, 2331, 2442, 2631, 2664),
            # The two K562 libraries read separately. Summing both is
            # the correct reading of Avsec 2021; these keys exist to
            # measure that choice, not to replace it -- 4828 alone gives
            # Gasperini bin 1 = 0.378 against the paper's 0.377, where
            # the two-track sum gives 0.392. They are named for what
            # they are, because "the two K562 CAGE tracks" reads as two
            # replicates of one thing and they are not: one is a
            # library, the other is three collapsed into a single
            # output.
            ("cage", "k562-cnhs11250"): (4828,),
            ("cage", "k562-encode"): (5111,),
            # The random-2-track cell-type-agnostic control. Moving
            # `cell_type` from k562 to all is the largest effect in the
            # free-parameter sweep, but it moves two things at once: 2
            # K562-specific tracks become 638 tracks of every cell type.
            # These draws hold the COUNT at 2 and vary only specificity,
            # so the pair of arms separates the two. Ten draws give a
            # null BAND rather than a point, which matters because one
            # random pair could land on a K562-like cell by luck.
            #
            # Drawn once with numpy default_rng(20260816) over
            # range(4675, 5313), excluding K562's own 4828 and 5111,
            # without replacement, all ten distinct. Literals rather
            # than a draw at import, because a future numpy may change
            # the generator's stream.
            ("cage", "rand2-s0"): (4711, 4762),
            ("cage", "rand2-s1"): (4745, 5185),
            ("cage", "rand2-s2"): (5144, 5251),
            ("cage", "rand2-s3"): (5018, 5112),
            ("cage", "rand2-s4"): (4927, 5073),
            ("cage", "rand2-s5"): (4800, 4948),
            ("cage", "rand2-s6"): (4875, 5198),
            ("cage", "rand2-s7"): (4820, 5245),
            ("cage", "rand2-s8"): (4709, 5129),
            ("cage", "rand2-s9"): (4975, 5032)},
)

# Borzoi: 524,288 bp in, 32 bp bins, 16,352 bins out, 7,611 human
# tracks. Manuscript weights, hence the OLD transform: y^0.75, residual
# sqrt above 384, then a scale factor.
#
# n_bins is 16,352, not the 6,144 usually quoted, and the difference is
# 5,104 bins of crop. Borzoi TRAINS with `Cropping1D(5120)`, giving
# 6,144 output bins = 196,608 bp, which is the number in the paper and
# the default of every PyTorch port. But both CRISPRi runners pass
# `params_pred.json`, whose trunk crops 16 bins, giving 16,384 - 32 =
# 16,352 bins = 523,264 bp; the runners then derive the output start and
# length from that. So the released benchmark read a gene's exon bins
# across nearly the whole input, and `crop_bins` here is 16 (512 bp),
# not 5,120 (163,840 bp).
#
# ("rnaseq", "k562") is the 20 STRANDED RNA:K562 tracks, 10 +/- pairs,
# and not the 23 tracks whose description is `RNA:K562`. The target file
# the runners pass is pinned by `--aggregate_tracks 10`: baskerville's
# `targets_prep_strand()` writes strand '.' for a self-paired row and
# collapses on `strand != '-'`, so the 23-row reading leaves
# 3 unstranded + 10 sense = 13 targets and the reshape to
# (batch, bins, 13 // 10, 10) cannot run. Dropping the 3 unstranded rows
# collapses to exactly 10, the flag's value. With the strand mask on a
# `+` gene therefore reads the 10 sense tracks, and `track_agg = mean`
# over them is exactly `--aggregate_tracks 10`.

# ---- Borzoi's CAGE block: 1,276 tracks = 638 FANTOM samples x {+, -} ---
#
# These sets are the Borzoi twins of Enformer's three cell-type arms:
# two K562 CAGE tracks, two random non-K562 tracks, all 638 samples.
# Without them, splitting the cell-type effect into specificity and
# track count would rest on one trunk, and could not be called a
# property of the protocol rather than of Enformer.
#
# Read off the released targets table:
#   * CAGE occupies indices 0..1275 contiguously, 1,276 of the head's
#     7,611 human tracks.
#   * The block is 638 stranded PAIRS laid out (2k, 2k+1) = (+, -).
#     Checked exhaustively: every even index's identifier ends '+',
#     every odd one '-', and `strand_pair` is involutive with
#     |partner - i| == 1 across all 1,276. That is why the strand map
#     below can be a rule rather than 1,276 literals, and the self-test
#     asserts the invariant rather than trusting it.
#   * All 1,276 share ONE transform: clip 768, clip_soft 384, scale 1.0,
#     sum_stat sum. So ("cage", "k562")'s (1.0, 1.0, 384.0) is correct
#     for EVERY CAGE set, and `inverse_transform: true` -- which the
#     Borzoi backbone sets and Enformer's does not -- un-squashes all
#     638 samples identically. No per-sample constants are needed.
#
# Declaring the strands is load-bearing, not bookkeeping. GENE-SCALAR
# keeps a track when its strand is `None` or matches the gene, so an
# undeclared track is kept for BOTH strands. Declaring 1,276 tracks
# without their strands would not raise: it would silently make
# `strand_mask=auto` inert and average 1,276 tracks where the arm is
# specified to read 638, and nothing in the score table would show it.
#
# Borzoi has `has_strand_column=True`, so `strand_mask: auto` resolves
# on and the declared sets read:
#       ("cage", "k562")       4 tracks (2 samples)   -> 2
#       ("cage", "rand2-s*")   4 tracks (2 samples)   -> 2
#       ("cage", "all")    1,276 tracks (638 samples) -> 638
# i.e. 2 / 2 / 638 EFFECTIVE tracks, exactly Enformer's 2 / 2 / 638. The
# three-point comparison is therefore matched on effective track count
# across the two models, and the only thing that differs is the model.
BORZOI_CAGE_BLOCK: Tuple[int, ...] = tuple(range(1276))

#: Absolute track -> strand for the whole CAGE block, from the
#: (even, odd) = (+, -) invariant above. Merged UNDER the hand-written
#: literals in `track_strand` so those stay the authority; they agree,
#: and the self-test asserts the agreement rather than trusting it.
BORZOI_CAGE_STRAND: Dict[int, str] = {
    i: ("+" if i % 2 == 0 else "-") for i in BORZOI_CAGE_BLOCK}

#: K562's own two CAGE SAMPLES, as ordinals into the 638:
#: 153 = CNhs11250, 436 = CNhs12336.
BORZOI_CAGE_K562_ORDINALS: Tuple[int, int] = (153, 436)

#: The ten random-2-sample draws: the SAME ten draws Enformer uses, not
#: an independent redraw, and that is the whole design.
#:
#: The two target files are index-aligned by biological sample: their
#: `description` strings are byte-identical and in the same order for
#: all 638, so Enformer index 4675 + k and Borzoi tracks (2k, 2k+1) are
#: the same FANTOM sample, and both K562 anchors confirm it. The
#: alignment is by description and NOT by identifier -- 112 of the 638
#: `identifier` strings differ, because the two files picked different
#: FANTOM libraries of the same described sample.
#:
#: Drawn by the same procedure and seed as ENFORMER_META's rand2-s*:
#: `default_rng(20260816)`, then ten times
#: `sorted(rng.choice(pool, 2, replace=False))` over the 636 non-K562
#: samples. The two pools are order-isomorphic -- Enformer's is Borzoi's
#: ordinals shifted by 4675 -- so the generator selects the same
#: positions and therefore the same samples, and mapping these ordinals
#: through 4675 + k reproduces ENFORMER_META's ten literals exactly, as
#: the self-test asserts.
#:
#: Ten independent draws would instead give Borzoi a band whose position
#: confounds "which cell types were drawn" with "which model read them".
#: Reusing them pairs the two bands at the level of the cell type, so a
#: difference between bands is a model difference; that pairing cannot
#: be recovered after the fact. Ordinals rather than tracks, because an
#: ordinal is what is shared with Enformer; literals rather than a live
#: `rng.choice`, for ENFORMER_META's stated reason.
BORZOI_CAGE_RAND2_ORDINALS: Dict[str, Tuple[int, int]] = {
    "rand2-s0": (36, 87), "rand2-s1": (70, 510), "rand2-s2": (469, 576),
    "rand2-s3": (343, 437), "rand2-s4": (252, 398), "rand2-s5": (125, 273),
    "rand2-s6": (200, 523), "rand2-s7": (145, 570), "rand2-s8": (34, 454),
    "rand2-s9": (300, 357),
}


def _borzoi_cage_tracks(ordinals: Sequence[int]) -> Tuple[int, ...]:
    """CAGE sample ordinals -> absolute tracks, both strands, ascending.

    One sample k is the pair (2k, 2k+1) = (+, -). BOTH strands are
    declared even though `strand_mask=auto` reads only one: the mask
    is a readout choice that has to stay live, and declaring only the
    sense strand would hard-code the '+' answer and then silently
    mis-serve every '-' gene, whose own strand's track would not be in
    the set at all.
    """
    out: List[int] = []
    for k in ordinals:
        if not 0 <= k < 638:
            raise ValueError(f"CAGE sample ordinal {k} is outside 0..637")
        out += [2 * k, 2 * k + 1]
    return tuple(sorted(out))


BORZOI_META = ModelMeta(
    name="borzoi", seq_len=524_288, bin_size=32, n_bins=16_352,
    has_attention=True, scalar_is_squashed=True, transform_generation="old",
    log_base="e", has_strand_column=True,
    tracks={("cage", "k562"): (306, 307, 872, 873),
            # 638 samples x 2 strands. `strand_mask=auto` reads 638 of
            # them: the same 638 CAGE readouts that Enformer's
            # ("cage", "all") averages, one strand each.
            ("cage", "all"): BORZOI_CAGE_BLOCK,
            # The CAGE block minus K562's own two samples: 636 samples,
            # 1,272 tracks. This is the pool the rand2-s* draws come
            # from, so naming it keeps the three sets provably
            # consistent rather than consistent by inspection.
            ("cage", "not-k562"): _borzoi_cage_tracks(
                [k for k in range(638) if k not in BORZOI_CAGE_K562_ORDINALS]),
            # The ten random-2-sample draws: 4 tracks each, 2 read under
            # the strand mask. Same ten FANTOM samples as
            # ENFORMER_META's rand2-s*; see BORZOI_CAGE_RAND2_ORDINALS.
            **{("cage", name): _borzoi_cage_tracks(ords)
               for name, ords in BORZOI_CAGE_RAND2_ORDINALS.items()},
            ("rnaseq", "k562"): (
                6122, 6123, 6124, 6125, 6126, 6127, 6128, 6129,
                6230, 6231, 6464, 6465, 7044, 7045, 7077, 7078,
                7263, 7264, 7390, 7391)},
    track_strand={
        **BORZOI_CAGE_STRAND,                 # by the (even, odd) rule
        306: "+", 307: "-", 872: "+", 873: "-",
        # RNA-seq, from the identifier suffix in the released targets
        # table
        6122: "+", 6123: "-", 6124: "+", 6125: "-", 6126: "+",
        6127: "-", 6128: "+", 6129: "-", 6230: "+", 6231: "-",
        6464: "+", 6465: "-", 7044: "+", 7045: "-", 7077: "+",
        7078: "-", 7263: "+", 7264: "-", 7390: "+", 7391: "-"},
    transform_constants={("rnaseq", "k562"): (0.3, 0.75, 384.0),
                         # Every CAGE set gets K562's constants, because
                         # all 1,276 CAGE tracks carry the same
                         # clip_soft, scale and sum_stat. Checked over
                         # the whole block, not assumed.
                         **{("cage", name): (1.0, 1.0, 384.0)
                            for name in ("k562", "all", "not-k562",
                                         *BORZOI_CAGE_RAND2_ORDINALS)}},
)

#: The per-assay split of ("multi", "k562"). The readout is
#: u = sum_a log(E_a + eps) -- each assay logged SEPARATELY, then summed
#: -- so the grouping is load-bearing and cannot be recovered from the
#: flat track tuple. The order is fixed, and is the order the manifest
#: records.
#:
#: The sum is over logs because the four assays sit on genuinely
#: different scales: CAGE is clip 384, scale 1, sum_stat sum, while
#: DNase, H3K4me3 and RNAPII are all clip 32, scale 2, sum_stat mean. A
#: raw sum or mean over the 26 tracks is dominated by whichever assay
#: happens to be numerically largest, the same failure that pins a raw
#: linear mixture at its dominant endpoint. Summing logs makes the
#: readout scale-free:
#:     du/dx = sum_a (dE_a/dx) / E_a
#: so each assay contributes its gradient normalised by its own level,
#: and the score measures whether the modalities AGREE about an enhancer
#: rather than which one carries the biggest numbers. In linear space
#: this is the geometric mean of the four readouts, and it is the direct
#: generalisation of `log_scalar`. Being one scalar it costs one
#: backward pass, where giving each assay its own map channels, and so
#: leaving the weighting free afterwards, would cost one pass per assay.
ENFORMER_MULTI_GROUPS: Tuple[Tuple[str, Tuple[int, ...]], ...] = (
    ("cage", (4828, 5111)),
    ("dnase", (33, 34, 35, 121, 122, 123, 625)),
    ("h3k4me3", (701, 1210, 1387, 3507)),
    ("rnapii", (952, 979, 1341, 1364, 1367, 1427, 1428, 1429,
                1542, 2331, 2442, 2631, 2664)),
)

#: (model, assay, cell_type) -> ((assay_name, n_tracks), ...), as
#: CONTIGUOUS SLICES of the declared track tuple, which is the
#: concatenation of the groups above in that order. Keyed on the model
#: as well, because the indices above are Enformer head positions and
#: would silently address unrelated tracks on another trunk.
#:
#: Groups are sizes rather than absolute indices, because a backend is
#: free to hand back a renumbered subset of columns: `SyntheticBackend`
#: remaps every index as `i % n_tracks`, and a lookup keyed on Enformer
#: head indices then matches nothing and the readout fails with
#: "resolved no assay group" while looking correct on paper. Slicing is
#: valid only because nothing removes tracks between declaration and
#: readout -- Enformer has `has_strand_column=False`, so
#: `strand_mask_on` is always False, and `assay=multi` is refused on
#: other trunks. The size assertion in the readout enforces that rather
#: than trusting it.
MULTI_GROUPS: Dict[Tuple[str, str, str], Tuple[Tuple[str, int], ...]] = {
    ("enformer", "multi", "k562"): tuple(
        (n, len(v)) for n, v in ENFORMER_MULTI_GROUPS),
}

# AlphaGenome: 1,048,576 bp in, RNA-seq at base resolution, targets not
# scaled.
#
# The indices here are POSITIONAL, not absolute, unlike Enformer's and
# Borzoi's. Enformer's backend returns all 5,313 output tracks, so
# (4828, 5111) can be absolute head indices. AlphaGenome cannot afford
# that: its RNA_SEQ head is 768 tracks at 1 bp over 1 Mb, which is
# 3.2 GB of output for a single forward pass, and ISM needs one forward
# pass per shuffled draw. So `backend_alphagenome.forward` returns ONLY
# the columns named here, in this order, and these indices are positions
# in that array. The absolute head indices live in the backend's
# ABSOLUTE_TRACKS, and the two are asserted to agree in length at
# construction.
#
# The five K562 RNA-seq tracks are the rows of AlphaGenome's own
# output metadata whose biosample_name is K562, absolute
# [119, 120, 390, 391, 571]:
#
#   positional  absolute  assay               strand
#   0           119       polyA plus RNA-seq  +
#   1           120       total RNA-seq       +
#   2           390       polyA plus RNA-seq  -
#   3           391       total RNA-seq       -
#   4           571       polyA plus RNA-seq  .
#
# Position 4 is unstranded, and is deliberately absent from the strand
# map so that `.get(t, None)` returns None and GENE-SCALAR keeps it for
# genes of either strand. That is what AlphaGenome's own recommended
# workflow does: `tidy_anndata(match_gene_strand=True)` drops only
# (+ gene, - track) and (- gene, + track), so a '.' track survives both.
# The strand mask therefore leaves THREE tracks per gene, not one, and
# `track_agg: mean` is a mean over three, which is what "the average
# predicted gene expression ... across K562 tracks" asks for. A
# two-track set would leave one.
#
# The absolute head indices for the other two K562 assays, confirmed
# against AlphaGenome's released metadata (`metadata_lib` is not part of
# the public `alphagenome` package), are CAGE 150 (+) / 423 (-) and
# PROCAP 2 (+) / 8 (-). Each pair is corroborated three ways: the
# metadata strand column; being the only K562 rows in its table, of 640
# CAGE and 128 PROCAP; and AlphaGenome's own `strand_reindexing` map,
# which sends 150 <-> 423 and 2 <-> 8, with the RNA-seq block reproduced
# as a control. The +/- offset is NOT constant across heads -- 273 for
# CAGE, 271 for RNA_SEQ -- because each head has a different track count
# and a different number of interleaved '.' rows, which is why the
# strands had to be read rather than inferred.

#: AlphaGenome's CAGE head is 640 channels of which 546 are real:
#: 273 (+) at 0-272 and their 273 (-) mirrors at 273-545. The remaining
#: 94 carry strand '.' and no `data_source` -- organism padding shared
#: with the mouse head. They are emitted and they are meaningless, so
#: every set below is built from ordinals and never from a raw slice.
#:
#: The expansion rule is a CONSTANT OFFSET, not Borzoi's interleave.
#: Sample k is the pair (k, k+273); Borzoi's is (2k, 2k+1). Copying
#: Borzoi's rule here would silently read a different sample's plus
#: strand as this sample's minus strand. Checked against AlphaGenome's
#: released metadata: the set of (minus - plus) offsets over all 273
#: pairs is exactly {273}, where an interleave would give {1}.
ALPHAGENOME_CAGE_N_SAMPLES: int = 273

#: K562's single CAGE sample as an ordinal: 150 (+), mirror 423 (-). One
#: biological sample -- `biosample_name == 'K562'` matches exactly these
#: two rows. Enformer and Borzoi each have two K562 CAGE samples;
#: AlphaGenome has one, which is what fixes the draw size at 1.
ALPHAGENOME_CAGE_K562_ORDINAL: int = 150

#: The ten random-1-sample draws, as ordinals into the 273.
#:
#: Drawn by the same procedure and seed as ENFORMER_META's and
#: BORZOI_META's rand2-s* -- `default_rng(20260816)`, then ten times
#: `sorted(rng.choice(pool, 1, replace=False))` over the 272 non-K562
#: sample ordinals -- but they are NOT, and cannot be, the same cell
#: types.
#:
#: Enformer and Borzoi share one 638-sample FANTOM panel, which is why
#: BORZOI_CAGE_RAND2_ORDINALS reuses Enformer's draws and their two
#: bands are paired sample for sample. AlphaGenome's CAGE panel is a
#: different 273-sample set: same consortium (`data_source == 'fantom'`)
#: but with the FANTOM library IDs stripped, no `CNhs` string anywhere
#: in the table, so the panels can be joined only by name. Of the 273,
#: 76 join exactly and 120 at token-Jaccard >= 0.90; of the twenty
#: samples Enformer and Borzoi drew, only 4 are present at all, the
#: misses being specific cancer lines an ontology-canonical panel does
#: not carry. So a figure putting the three bands side by side has to
#: say that these draws are independent: a gap between bands confounds
#: "which cells were drawn" with "which trunk read them".
#:
#: Literals rather than a live `rng.choice`, for ENFORMER_META's stated
#: reason. Distinctness is checked in the self-test rather than
#: guaranteed: ten size-1 draws from 272 collide with about 15%
#: probability, so a re-roll must re-check rather than assume.
ALPHAGENOME_CAGE_RAND1_ORDINALS: Dict[str, int] = {
    "rand1-s0": 15, "rand1-s1": 37, "rand1-s2": 41, "rand1-s3": 218,
    "rand1-s4": 30, "rand1-s5": 163, "rand1-s6": 247, "rand1-s7": 201,
    "rand1-s8": 136, "rand1-s9": 146,
}


def _alphagenome_cage_tracks(ordinals: Sequence[int]) -> Tuple[int, ...]:
    """CAGE ordinals -> absolute tracks, all '+' then all '-'.

    Sample k is the pair (k, k+273). BOTH strands are declared even
    though `strand_mask=auto` reads only one: the mask is a readout
    choice that must stay live, and declaring only the sense strand
    would hard-code the '+' answer and then silently mis-serve every
    '-' gene.

    The order is every '+' ascending, then every '-' ascending, which is
    also plain ascending order because every '+' index is < 273 and
    every '-' index is >= 273. `_alphagenome_cage_strands` depends on
    exactly that split, so the two must not drift apart.
    """
    ks = sorted(ordinals)
    for k in ks:
        if not 0 <= k < ALPHAGENOME_CAGE_N_SAMPLES:
            raise ValueError(
                f"CAGE sample ordinal {k} is outside "
                f"0..{ALPHAGENOME_CAGE_N_SAMPLES-1}")
    if len(set(ks)) != len(ks):
        raise ValueError(f"duplicate CAGE sample ordinals: {ks}")
    return tuple(ks) + tuple(k + ALPHAGENOME_CAGE_N_SAMPLES for k in ks)


def _alphagenome_cage_strands(n_samples: int) -> Dict[int, str]:
    """Position -> strand, matching `_alphagenome_cage_tracks`.

    Positions, not absolute indices: AlphaGenome is addressed
    positionally, and backend_alphagenome forces the metadata block to
    be `range(len(absolute_tracks))`, so several sets share the same
    block and their strands cannot live in the flat `track_strand`.
    """
    return {**{i: "+" for i in range(n_samples)},
            **{n_samples + i: "-" for i in range(n_samples)}}


#: Every non-K562 CAGE sample: the pool the rand1-s* draws come from.
#: Named rather than left implicit, so the three sets are provably
#: consistent instead of consistent by inspection.
ALPHAGENOME_CAGE_NOT_K562_ORDINALS: Tuple[int, ...] = tuple(
    k for k in range(ALPHAGENOME_CAGE_N_SAMPLES)
    if k != ALPHAGENOME_CAGE_K562_ORDINAL)

ALPHAGENOME_META = ModelMeta(
    name="alphagenome", seq_len=1_048_576, bin_size=1, n_bins=1_048_576,
    has_attention=True, scalar_is_squashed=False, transform_generation="none",
    log_base="2", has_strand_column=True,
    # POSITIONAL blocks. backend_alphagenome refuses anything but
    # range(len(absolute_tracks)), so these say only HOW MANY columns
    # the backend returns; the absolute head indices live in
    # backend_alphagenome.ABSOLUTE_TRACKS.
    tracks={("rnaseq", "k562"): (0, 1, 2, 3, 4),
            ("cage", "k562"): (0, 1),
            ("procap", "k562"): (0, 1),
            # 273 samples x 2 strands = 546. `strand_mask=auto`
            # reads 273 of them.
            ("cage", "all"): tuple(range(2 * ALPHAGENOME_CAGE_N_SAMPLES)),
            # 272 samples x 2 strands = 544: the CAGE block minus
            # K562's own sample.
            ("cage", "not-k562"): tuple(
                range(2 * len(ALPHAGENOME_CAGE_NOT_K562_ORDINALS))),
            # The ten random-1-SAMPLE draws: 2 tracks each, 1 read
            # under the strand mask.
            **{("cage", name): (0, 1)
               for name in ALPHAGENOME_CAGE_RAND1_ORDINALS}},
    # Empty: several AlphaGenome sets share the same positional block,
    # so one flat index -> strand map cannot serve them all.
    track_strand={},
    track_strand_by_set={
        # Positions -> strands. The rnaseq absolutes are 119 + / 120 + /
        # 390 - / 391 - / 571 '.'; position 4 is deliberately absent so
        # `.get` returns None and the '.' track survives both strand
        # filters.
        ("rnaseq", "k562"): {0: "+", 1: "+", 2: "-", 3: "-"},
        # cage: 150 + / 423 -
        ("cage", "k562"): {0: "+", 1: "-"},
        # procap: 2 + / 8 - (ENCSR740IPL). Same positional block (0, 1)
        # as cage, which is precisely why these strands cannot live in
        # the flat `track_strand`: three sets share (0, 1), and a flat
        # map has room for only one of them.
        ("procap", "k562"): {0: "+", 1: "-"},
        # The CAGE cell-type sets, built by the same helper that orders
        # the absolute indices, so the two cannot drift.
        ("cage", "all"): _alphagenome_cage_strands(ALPHAGENOME_CAGE_N_SAMPLES),
        ("cage", "not-k562"): _alphagenome_cage_strands(
            len(ALPHAGENOME_CAGE_NOT_K562_ORDINALS)),
        **{("cage", name): _alphagenome_cage_strands(1)
           for name in ALPHAGENOME_CAGE_RAND1_ORDINALS},
    },
)

MODEL_META = {m.name: m
              for m in (ENFORMER_META, BORZOI_META, ALPHAGENOME_META)}


# ============================================================================
# 3. Validation and resolution
# ============================================================================

class ConfigError(ValueError):
    """The configuration is not legal. Raised before any GPU work."""


@dataclass
class Resolved:
    """What resolution derives from metadata, not from the user."""
    meta: ModelMeta
    tracks: Tuple[int, ...]
    #: Strand of each entry of `tracks`, same order, None where a track
    #: is unstranded. Resolved once here rather than looked up at score
    #: time: resolution is the only place that knows which track SET is
    #: in play, and a lookup made anywhere else can read a wrong strand.
    track_strands: Tuple[Optional[str], ...]
    scale: float
    exponent: float
    clip_soft: Optional[float]
    strand_mask_on: bool
    warnings: List[str] = field(default_factory=list)


def load_config(path: str) -> Dict[str, Any]:
    """Read a JSON or YAML config file into a raw mapping."""
    with open(path) as fh:
        text = fh.read()
    if path.endswith((".yaml", ".yml")):
        try:
            import yaml
        except ImportError as exc:
            raise ConfigError("PyYAML is needed for .yaml configs") from exc
        raw = yaml.safe_load(text)
    else:
        raw = json.loads(text)
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: top level must be a mapping")
    return raw


#: THE PROTOCOL TABLE. These parameters are a pure function of which
#: paper a config reproduces: one value per protocol, no exceptions.
#: Hoisting them out of the individual configs keeps the study's own
#: independent variable -- the exact ways in which the protocols
#: differ -- readable as one table, with Linder 2025 the outlier on
#: three of the four. A config that states one of them still wins.
#:
#: `ism_window` and `ism_inner_mean` are deliberately absent. Linder
#: 2025 sets 2048 / {strand} on its ISM arm and leaves the schema
#: defaults on its GRADIENT arm, so they are a function of
#: (protocol, method) and belong in the configs that use them; hoisted
#: here they make the linder gradient configs fail validation.
#:
#: PRECEDENCE, lowest to highest: Param.default ->
#: PROTOCOL_DEFAULTS[_protocol] -> the config.
PROTOCOL_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "avsec2021": dict(gencode="v32", exon_source="all-transcripts",
                      out_of_window="zero", mean_subtract=False),
    "linder2025": dict(gencode="v41", exon_source="basic-nort",
                       out_of_window="drop", mean_subtract=True),
    "avsec2026": dict(gencode="v46", exon_source="all-transcripts",
                      out_of_window="zero", mean_subtract=True),
    "unified-t1": dict(gencode="v50", exon_source="all-transcripts",
                       out_of_window="zero", mean_subtract=True),
}


def fill_defaults(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Reject unknown keys; fill unset parameters from the defaults."""
    unknown = sorted(set(raw) - set(PARAMS)
                     - {"_comment", "_name", "_protocol", "_overrides"})
    if unknown:
        raise ConfigError(f"unknown parameter(s): {', '.join(unknown)}")
    # A protocol name with no entry is not an error: an ad-hoc config
    # simply gets the plain schema defaults.
    proto = PROTOCOL_DEFAULTS.get(raw.get("_protocol"), {})
    theta = {name: raw.get(name, proto.get(name, p.default))
             for name, p in PARAMS.items()}
    for key in ("_name", "_protocol", "_comment", "_overrides"):
        if key in raw:
            theta[key] = raw[key]
    if isinstance(theta["ism_inner_mean"], list):
        theta["ism_inner_mean"] = tuple(theta["ism_inner_mean"])
    return theta


def coerce_value(key: str, val: str) -> Any:
    """Turn a `--set` string into the type a YAML config would produce.

    The coercion has to be exact, not lenient. `validate` tests
    membership in the declared domain, so `ism_shuffles=4` arriving as
    the string "4" is not in `(1, 2, 4, ...)`, and a legal override
    would be rejected with a domain error pointing at the value rather
    than at the parsing.
    """
    p = PARAMS[key]
    if p.kind == "dict":
        return json.loads(val) if val else {}
    if p.kind == "set":
        return tuple(x for x in (s.strip() for s in val.split(",")) if x)
    if val == "none" and p.choices and "none" in p.choices:
        # `none` is a value of its own for `normalise` and `tta`
        return "none"
    if val in ("", "null", "none", "None"):
        return None
    if p.kind == "int":
        return int(val)
    if p.kind == "float":
        return float(val)
    if p.kind == "bool":
        low = val.lower()
        if low in ("true", "1", "yes"):
            return True
        if low in ("false", "0", "no"):
            return False
        raise ConfigError(f"--set {key}: expected a boolean, got {val!r}")
    return val


def apply_overrides(raw: Dict[str, Any],
                    settings: Sequence[str]) -> Dict[str, Any]:
    """Apply `--set KEY=VALUE` on top of a loaded config, in order.

    Overrides go in BEFORE `fill_defaults`, so an override is
    indistinguishable from having written the value into the YAML: it
    passes through the same unknown-key rejection and the same
    validator. There is deliberately no way to bypass validation from
    the command line.
    """
    out = dict(raw)
    for item in settings:
        if "=" not in item:
            raise ConfigError(f"--set expects KEY=VALUE, got {item!r}")
        key, _, val = item.partition("=")
        key = key.strip()
        if key not in PARAMS:
            raise ConfigError(f"--set: unknown parameter {key!r}")
        out[key] = coerce_value(key, val.strip())
    if settings:
        # Recorded so a manifest says how the run was produced, not
        # merely what it resolved to.
        out["_overrides"] = list(settings)
    return out


def major_tss_cell_type(theta: Dict[str, Any]) -> Any:
    """Which cell type defines the MAJOR TSS.

    Not always the cell type being read out. Null follows `cell_type`,
    so a config that leaves it unset picks the major TSS in the cell
    type it reads out. Only the cell-type-agnostic arms set it, and they
    set it to k562.
    """
    return theta.get("tss_major_cell_type") or theta["cell_type"]


def refuse(cond: bool, msg: str) -> None:
    """Raise `ConfigError(msg)` when `cond` holds."""
    if cond:
        raise ConfigError(msg)


def check_readout_scheme(theta: Dict[str, Any]) -> None:
    """Refuse an incoherent assay/region pairing under `unified-t1`.

    Every other check asks whether the run is REPRESENTABLE: does the
    model own these tracks, is the width a whole number of bins. None
    of them can catch CAGE@exons, because that combination is perfectly
    representable -- Borzoi has CAGE tracks, exon intervals exist, and
    the map computes. It is meaningless rather than illegal, which is a
    claim about biology and so has to be asserted rather than derived.
    Hence a separate rule, scoped to the sweep.
    """
    if theta.get("_protocol") != "unified-t1":
        return
    want = READOUT_SCHEMES.get(theta["assay"])
    # `all-tss` satisfies a `tss` pairing: it is the same biology read
    # over more of the promoter region, not a different readout scheme.
    # Without this the rule would reject cage@all-tss as if it were the
    # meaningless cage@exons cell.
    paired = (theta["region"] == want
              or (want == "tss" and theta["region"] in TSS_REGIONS))
    refuse(want is not None and not paired,
           f"_protocol=unified-t1 forbids assay={theta['assay']} with "
           f"region={theta['region']}: {theta['assay']} pairs with "
           f"region={want}. Under this protocol assay and region are "
           "one axis -- the readout scheme -- with levels "
           + ", ".join(f"{a}@{r}" for a, r in READOUT_SCHEMES.items())
           + f". Set region={want} (and tss_width accordingly: a value "
           "for tss, null for exons), or drop _protocol if you "
           "deliberately want an unpaired readout.")


def validate(theta: Dict[str, Any]) -> List[str]:
    """Refuse illegal parameter combinations; return the warnings."""
    meta = MODEL_META[theta["model"]]
    warnings: List[str] = []

    # ---- every argument is in its declared domain -----------------
    for name, p in PARAMS.items():
        v = theta[name]
        if v is None or p.choices is None:
            continue
        if p.kind == "set":
            continue
        refuse(v not in p.choices,
               f"{name}={v!r} is not one of {p.choices}")
    for member in theta["ism_inner_mean"]:
        refuse(member not in PARAMS["ism_inner_mean"].members,
               f"ism_inner_mean member {member!r} unknown; "
               f"allowed: {PARAMS['ism_inner_mean'].members}")

    refuse(theta["method"] == "ism"
           and theta["ism_statistic"] in RELATIVE_STATISTICS
           and theta["log_scalar"],
           f"ism_statistic={theta['ism_statistic']} requires "
           "log_scalar=false (dividing by a logarithm is meaningless, "
           "and with pseudocount 0 it can be negative)")
    refuse(theta["assay"] == "multi" and theta["log_scalar"],
           "assay=multi is already a sum of per-assay logs "
           "(u = sum_a log E_a); log_scalar=true would take a log of a "
           "log. Set log_scalar=false.")
    refuse(theta["assay"] == "multi" and theta["model"] != "enformer",
           "assay=multi is declared only for enformer: MULTI_GROUPS "
           "carries Enformer head indices, which would address "
           "unrelated tracks on another trunk.")

    # `all-tss` needs a TSS width for the same reason `tss` does: it is
    # the width of the window placed at each TSS. So both TSS_REGIONS
    # require it, and only `exons`, whose extent comes from the
    # annotation, refuses it.
    refuse(theta["region"] in TSS_REGIONS and theta["tss_width"] is None,
           f"region={theta['region']} requires tss_width")
    refuse(theta["region"] not in TSS_REGIONS
           and theta["tss_width"] is not None,
           f"tss_width is only meaningful when region is one of {TSS_REGIONS}")
    if theta["region"] in TSS_REGIONS:
        refuse(theta["tss_width"] % meta.bin_size != 0,
               f"tss_width={theta['tss_width']} is not a multiple of "
               f"{meta.name}'s {meta.bin_size} bp bin")
    refuse(theta["enh_agg"] in ("sum", "mean", "max")
           and theta["enh_span"] is None,
           f"enh_agg={theta['enh_agg']} requires enh_span")
    refuse(theta["enh_agg"] not in ("sum", "mean", "max")
           and theta["enh_span"] is not None,
           "enh_span is only meaningful when enh_agg is one of sum, mean, max")
    refuse(theta["enh_agg"] == "gaussian"
           and (theta["gaussian_sigma"] is None
                or theta["gaussian_truncate"] is None),
           "enh_agg=gaussian requires gaussian_sigma and gaussian_truncate")
    refuse(theta["enh_agg"] != "gaussian"
           and (theta["gaussian_sigma"] is not None
                or theta["gaussian_truncate"] is not None),
           "gaussian_sigma / gaussian_truncate are only meaningful when "
           "enh_agg=gaussian")
    refuse(theta["anchor_agg"] is not None and theta["anchor"] != "each-tss",
           "anchor_agg is only meaningful when anchor=each-tss")
    refuse(theta["anchor_agg"] is None and theta["anchor"] == "each-tss",
           "anchor=each-tss requires anchor_agg")
    refuse(theta["tss_major_assay"] is not None and not theta["tss_major"],
           "tss_major_assay is only meaningful when tss_major=true")
    refuse(theta["tss_major_assay"] is None and theta["tss_major"],
           "tss_major=true requires tss_major_assay")
    refuse(theta["tss_major_cell_type"] is not None and not theta["tss_major"],
           "tss_major_cell_type is only meaningful when tss_major=true")
    refuse(theta["method"] == "attention" and not meta.has_attention,
           f"method=attention is not available for {meta.name}; "
           f"{meta.name}'s attention is not readable on a genomic token grid")
    refuse(bool(theta["ism_inner_mean"]) and theta["method"] != "ism",
           "ism_inner_mean is only meaningful when method=ism")
    refuse(theta["track_abs_order"] == "before" and theta["method"] != "ism",
           "track_abs_order=before is implemented for method=ism only. "
           "The per-track readout returns a vector, which the gradient "
           "arm cannot differentiate to a scalar and the attention arm "
           "never reads; both would need their own per-track path")
    # tta_abs_order places |.| before or after the augmentation mean.
    refuse(theta["tta_abs_order"] == "after" and theta["method"] == "ism",
           "tta_abs_order is the gradient/attention form of the "
           "question ism_inner_mean already answers on the ISM path: "
           "put 'strand' in ism_inner_mean to average signed over the "
           "orientations. Two mechanisms for one ordering would be two "
           "places to disagree")
    refuse(theta["tta_abs_order"] == "after" and theta["tta"] == "none",
           "tta_abs_order=after with tta=none is inert -- there is one "
           "augmentation to average. Set tta=rc (Linder 2025's --rc) or "
           "leave tta_abs_order at before")
    # A warning, not a refusal: the ISM branch returns at `ism_combine`,
    # before `merge_maps`, which is this parameter's only reader, so on
    # that path the setting is UNREAD rather than wrong.
    if theta["replicate_abs_order"] == "after" and theta["method"] == "ism":
        warnings.append(
            "replicate_abs_order is not read on the ISM path -- it is "
            "decided in merge_maps, which ism_combine returns before "
            "reaching. To average the signed replicate maps under ISM, "
            "put 'replicate' in ism_inner_mean instead.")
    # A warning, not a refusal: at replicates=1 the replicate mean is
    # the identity, so `before` and `after` reduce to |M| exactly, not
    # approximately. The setting is inert rather than wrong, and a
    # default has to be legal everywhere it is reached.
    if theta["replicate_abs_order"] == "after" and theta["replicates"] == 1:
        warnings.append(
            "replicate_abs_order=after is inert at replicates=1 -- "
            "there is one map to average, so it reduces to `before`. "
            "Live only on the Borzoi arm (replicates=4).")
    refuse("shuffles" in theta["ism_inner_mean"],
           "the shuffle axis P is placed by ism_statistic, not by "
           "ism_inner_mean")
    refuse("tss" in theta["ism_inner_mean"] and theta["anchor"] != "each-tss",
           "ism_inner_mean contains 'tss' but anchor is not each-tss")
    refuse("strand" in theta["ism_inner_mean"] and "rc" not in theta["tta"],
           "ism_inner_mean contains 'strand' but tta does not include rc")
    refuse(theta["enh_agg"] == "point" and theta["method"] != "ism",
           "enh_agg=point is the ISM reduction; it is meaningless for "
           "a positional map")
    refuse(theta["enh_agg"] != "point" and theta["method"] == "ism",
           "method=ism requires enh_agg=point (the perturbed window IS "
           "the enhancer window)")

    refuse(theta["bin_overlap"] == "fractional"
           and theta["method"] != "attention",
           "bin_overlap=fractional weights boundary bins by their "
           "overlap, which only a coarse-grid map supports; gradient "
           "and ISM maps are per-nucleotide and their readout bins are "
           "a set, so use any or majority")
    refuse(theta["method"] == "ism" and theta["normalise"] != "none",
           "method=ism requires normalise=none; ISM-COMBINE does not "
           "apply a global normaliser (the statistic is already "
           "cross-gene calibrated)")
    # `avsec8` needs tta_shift_bp too: it is the k of Avsec's "<=3 bp
    # shifts", the support of the uniform draw, so an arm's shift range
    # stays in the config and the manifest instead of becoming a
    # constant in the code.
    refuse((("shift" in theta["tta"]) or theta["tta"] == "avsec8")
           != (theta["tta_shift_bp"] is not None),
           "tta_shift_bp is required by, and only by, a tta that "
           "includes shift (avsec8 included -- it is the support of "
           "Avsec 2021's uniform shift draw)")
    # 43 bp is the bin-thirds proposal, larger than Avsec's stated
    # maximum. Averaging eight random 43 bp shifts is neither Avsec's
    # protocol nor Karollus's deterministic six, so refuse the
    # combination rather than let an arm claim Avsec's name for it; arms
    # Z1/Z2 are where 43 bp belongs.
    refuse(theta["tta"] == "avsec8" and theta["tta_shift_bp"] not in (1, 2, 3),
           "tta=avsec8 reproduces Avsec 2021's stated test-time "
           "augmentation, which is '<=3 bp shifts'. For the 43 bp "
           "bin-thirds proposal use tta=shift or rc+shift (arms Z1/Z2)")
    key = (theta["assay"], theta["cell_type"])
    refuse(key not in meta.tracks,
           f"{meta.name} has no track set for "
           f"assay={key[0]}, cell_type={key[1]}")
    refuse(theta["inverse_transform"] and not meta.scalar_is_squashed,
           f"inverse_transform=true but {meta.name} does not predict "
           "in a squashed space")

    if (theta["log_scalar"] and not theta["inverse_transform"]
            and meta.scalar_is_squashed):
        warnings.append(
            "log_scalar=true with inverse_transform=false is "
            "unreachable in calico (seqnn.py:726 applies the log only "
            "on the un-transformed branch)")
    # Linder 2025 is exempt from this warning: its own gradient
    # invocations pass `--rc`, and its saturation-gradient script loops
    # over both orientations on the GRADIENT path, accumulating them
    # into one map. Avsec 2021's gradient arms are undocumented on the
    # point, so the warning stands for them and is withdrawn only where
    # a paper's own invocation settles it.
    if (theta["method"] == "gradient" and theta["tta"] != "none"
            and theta.get("_protocol") != "linder2025"):
        warnings.append(
            "no paper states test-time augmentation for a gradient "
            "arm; this is a deviation")
    if (theta["method"] == "gradient" and theta["tta"] != "none"
            and theta["tta_abs_order"] == "before"
            and theta.get("_protocol") == "linder2025"):
        warnings.append(
            "linder2025 + gradient + tta=rc with tta_abs_order=before "
            "absolutises each orientation separately; "
            "borzoi_satg_gene.py sums the signed maps and "
            "borzoi_bench_crispr.py:356 takes |.| once, which is "
            "tta_abs_order=after")

    check_readout_scheme(theta)
    return warnings


def resolve(theta: Dict[str, Any],
            meta: Optional[ModelMeta] = None) -> Resolved:
    """Derive everything that comes from metadata, not from user choice.

    `meta` defaults to the real model's metadata; a backend passes its
    own so that a reduced-geometry stand-in resolves to its own track
    indices and bin size.
    """
    meta = meta if meta is not None else MODEL_META[theta["model"]]
    key = (theta["assay"], theta["cell_type"])
    if key not in meta.tracks:
        # A bare KeyError(('cage', 'k562')) three frames deep does not
        # say what to do about it. The path is reachable: a model
        # declares only the track sets someone has checked, so adding
        # an arm means stating its block rather than inheriting indices
        # that could resolve to the wrong strand.
        raise ConfigError(
            f"{meta.name} has no track set for assay={theta['assay']!r} "
            f"cell_type={theta['cell_type']!r}. Declared: "
            f"{sorted(meta.tracks)}. Add the entry to "
            f"{meta.name.upper()}_META.tracks (and its strands to "
            "track_strand, or to track_strand_by_set when the backend "
            "numbers each set positionally) before running this "
            "configuration.")
    scale, exponent, clip_soft = meta.transform_constants.get(
        key, (1.0, 1.0, None))
    if theta["strand_mask"] == "auto":
        strand_on = meta.has_strand_column
    else:
        strand_on = theta["strand_mask"] == "on"
    # The per-set map wins over the flat one, and only for the set it
    # names, so Enformer and Borzoi, which declare no per-set map, are
    # unaffected.
    by_set = meta.track_strand_by_set.get(key, {})
    tracks = meta.tracks[key]
    strands = tuple(by_set.get(t, meta.track_strand.get(t)) for t in tracks)
    return Resolved(meta=meta, tracks=tracks, track_strands=strands,
                    scale=scale, exponent=exponent, clip_soft=clip_soft,
                    strand_mask_on=strand_on)


# ============================================================================
# 4. Genomic inputs
# ============================================================================

@dataclass(frozen=True)
class Enhancer:
    """A candidate element, as the screen delivered it."""

    chrom: str
    start: int
    end: int
    eid: str

    def mid(self) -> int:
        """Midpoint of the interval, rounded down."""
        return (self.start + self.end) // 2


@dataclass(frozen=True)
class Gene:
    """A target gene and the anchors a window can be centred on."""

    gid: str
    chrom: str
    strand: str
    provided_tss: int
    gencode_tss: Tuple[int, ...]
    exons: Tuple[Tuple[int, int], ...]        # half-open [begin, end)
    # A precomputed MAJOR-TSS, filled by --anchors. It depends only on
    # (model, gencode, tss_major_assay, cell_type), so it is computed
    # once and cached rather than recomputed for every pair.
    #
    # It stays separate from `provided_tss`, which must keep meaning
    # "the TSS the screen shipped": provided_tss is also the distance
    # reference for `distance_to_tss` and for pair_filters, so
    # overwriting it would silently redefine the distance bins the pair
    # tables were built against.
    anchor_tss: Optional[int] = None
    # Which GENCODE release `gencode_tss` and `exons` came from.
    # Building the pair table refuses a mismatch against the configured
    # release: v32 and v41 disagree on the 5' TSS of 7.2% of shared
    # genes, and an annotation swap is invisible in every downstream
    # number. `None` means "this gene makes no claim", which is what
    # synthetic test fixtures are; the guard fires only on genes that
    # came from a loader, and every loader sets it.
    gencode: Optional[str] = None


@dataclass(frozen=True)
class Window:
    """The model's input window, in genomic coordinates."""

    #: Genomic, and may be negative near a chromosome end.
    start: int
    length: int

    @property
    def centre(self) -> int:
        """Centre of the window, rounded down."""
        return self.start + self.length // 2


def window_at(centre: int, length: int) -> Window:
    """A window of `length` bp whose centre is `centre`."""
    return Window(start=centre - length // 2, length=length)


def exon_centroid(gene: Gene) -> int:
    """The mean of every exonic base position, introns excluded.

    This reproduces the Borzoi reference gene midpoint: accumulate
    `range(begin, end)` for every exon, take the mean, truncate.
    """
    twice_total = 0          # 2 * sum of exonic positions, so this stays exact
    count = 0
    for begin, end in gene.exons:
        n = end - begin
        # sum_{i=begin}^{end-1} i = n * (begin + end - 1) / 2
        twice_total += n * (begin + end - 1)
        count += n
    if count == 0:
        raise ConfigError(
            f"gene {gene.gid} has no exons in this GENCODE release")
    return int((twice_total / 2) / count)          # truncate, not round


# ============================================================================
# 4b. Eulerian dinucleotide shuffle
#
# The Altschul-Erickson algorithm, as implemented in kundajelab/deeplift
# (deeplift/dinuc_shuffle.py, commit 0201a21). It is reproduced rather
# than reimplemented because a hand-rolled Eulerian shuffle that omits
# the last-edge arborescence still runs and still looks plausible, but
# does not sample a valid Eulerian path, so the null would be wrong in
# a way nothing downstream would catch.
#
# An all-zero one-hot column round-trips as token D, so an ambiguous
# base stays ambiguous rather than being silently called.
# ============================================================================

def one_hot_to_tokens(one_hot):
    """Turn an L x D one-hot array into L integers in [0, D].

    The token D marks a column that is all 0. The encoding is assumed
    well formed: at most one 1 per column, 0s elsewhere.
    """
    tokens = np.tile(one_hot.shape[1], one_hot.shape[0])  # Vector of all D
    seq_inds, dim_inds = np.where(one_hot)
    tokens[seq_inds] = dim_inds
    return tokens


def tokens_to_one_hot(tokens, one_hot_dim):
    """Turn L integers in [0, D] back into an L x D one-hot array.

    `one_hot_dim` is D; the token D decodes to a column of 0s.
    """
    identity = np.identity(one_hot_dim + 1)[:, :-1]  # Last row is all 0s
    return identity[tokens]


def dinuc_shuffle(seq, num_shufs=None, rng=None):
    """Shuffle a sequence, preserving its dinucleotide frequencies.

    `seq` is an L x D one-hot array, `num_shufs` the number of shuffles
    N to create and `rng` a NumPy RandomState. Returns an N x L x D
    one-hot array, or a single L x D array when `num_shufs` is not
    given.
    """
    if type(seq) is str:
        raise ValueError(
            "this vendored copy takes one-hot only; strings are unused here")
    elif type(seq) is np.ndarray and len(seq.shape) == 2:
        seq_len, one_hot_dim = seq.shape
        arr = one_hot_to_tokens(seq)
    else:
        raise ValueError("Expected string or one-hot encoded array")

    if not rng:
        rng = np.random.RandomState()

    # The set of characters present, and where each one occurs; work in
    # `tokens`, the integer representation of the characters.
    chars, tokens = np.unique(arr, return_inverse=True)

    # For each token, the indices of the tokens that come after it.
    shuf_next_inds = []
    for t in range(len(chars)):
        mask = tokens[:-1] == t  # Excluding last char
        inds = np.where(mask)[0]
        shuf_next_inds.append(inds + 1)  # Add 1 for next token

    all_results = np.empty(
        (num_shufs if num_shufs else 1, seq_len, one_hot_dim),
        dtype=seq.dtype
    )

    for i in range(num_shufs if num_shufs else 1):
        # Shuffle the next indices
        for t in range(len(chars)):
            inds = np.arange(len(shuf_next_inds[t]))
            inds[:-1] = rng.permutation(len(inds) - 1)  # Keep last index same
            shuf_next_inds[t] = shuf_next_inds[t][inds]

        counters = [0] * len(chars)

        # Build the resulting array
        ind = 0
        result = np.empty_like(tokens)
        result[0] = tokens[ind]
        for j in range(1, len(tokens)):
            t = tokens[ind]
            ind = shuf_next_inds[t][counters[t]]
            counters[t] += 1
            result[j] = tokens[ind]

        all_results[i] = tokens_to_one_hot(chars[result], one_hot_dim)
    return all_results if num_shufs else all_results[0]


# ============================================================================
# 5. Model backends
# ============================================================================

class Backend:
    """What the algorithm needs from a model: methods and geometry.

    ONLY `forward` IS MANDATORY. The rest are independently optional,
    and which ones a backend implements decides which `method` values
    it can serve:

        forward           -> ism, and any config with tss_major (the
                             major TSS is an argmax over predicted CAGE,
                             so it is forward-only)
        scalar_gradient   -> gradient
        attention         -> attention

    A new model can therefore be brought up in one step -- implement
    `forward`, run the ISM arms -- and grow from there. `check_backend`
    enforces this BEFORE the run starts: asking for `gradient` from a
    forward-only backend is a ConfigError naming the missing method and
    listing what the backend can do, not a NotImplementedError thrown
    after the weights are on the GPU and the pair table is loaded.

    Leave these bodies as `NotImplementedError`. `backend_capabilities`
    detects an implementation by identity against this class, so a
    subclass that does not override a method must inherit it unchanged
    rather than override it with its own raise. A backend that DOES
    override a method only to raise -- `UnavailableBackend` does, to
    keep a better message -- must say so by setting `IMPLEMENTS`, or
    identity-detection will call it fully capable and the failure will
    land mid-run, which is what this machinery exists to prevent.
    """

    meta: ModelMeta

    #: Authoritative capability declaration, for a subclass that cannot
    #: be read by identity. `None` means detect by identity, the usual
    #: case.
    IMPLEMENTS: Optional[FrozenSet[str]] = None

    #: Optional one-liner shown when `check_backend` refuses this
    #: backend.
    capability_note: str = ""

    #: The gene readout of the LAST `scalar_gradient` call: `_last_u` as
    #: differentiated (logged under log_scalar), `_last_T` un-logged
    #: (`scalar_fn.linear`). NaN until a call has been made. The frame
    #: agreement gate reads them, so it costs no model call.
    _last_T: float = float("nan")
    _last_u: float = float("nan")

    def forward(self, x: np.ndarray, replicate: Any) -> np.ndarray:
        """x: (L, 4) -> predictions (n_bins, n_tracks). REQUIRED."""
        raise NotImplementedError

    def scalar_gradient(
            self, x: np.ndarray, replicate: Any,
            scalar_fn: Callable[[np.ndarray], float]) -> np.ndarray:
        """d scalar_fn(forward(x))/dx, shape (L, 4). One backward pass.

        OPTIONAL: omit it and `method = gradient` is refused, while
        everything else still runs. `scalar_fn` is numpy, so the
        contract is a vector-Jacobian product: form dU/dy on the entries
        it advertises (`scalar_fn.bins`, `scalar_fn.tracks`) and let the
        framework supply dy/dx. backend_enformer.py is the reference
        implementation; torch, TF and JAX all expose the primitive.
        """
        raise NotImplementedError

    def attention(self, x: np.ndarray, replicate: Any,
                  how: str) -> np.ndarray:
        """(n_tokens, n_tokens) attention, already reduced per `how`.

        OPTIONAL: omit it and `method = attention` is refused. Whether
        this method exists is a claim about the CODE, separate from
        `ModelMeta.has_attention`, which is a claim about the MODEL.

        All three trunks attend on a 128 bp token grid -- Enformer 1,536
        tokens, Borzoi 4,096, AlphaGenome 8,192 -- and all three
        materialise the softmax explicitly, so an attention map is
        readable from each. An output bin maps to a token by integer
        division: `//1` for Enformer (crop offset only), `//4` for
        Borzoi, `//128` for AlphaGenome.

        The caveat that belongs on the score: Enformer has no skip
        connections around its transformer, so everything its output
        depends on passed through attention. Borzoi and AlphaGenome are
        U-Nets whose encoder skips re-join the decoder having never been
        attended over, so for them an attention map is a PARTIAL account
        of enhancer-to-gene information flow. That is an interpretation
        caveat, not an impossibility.
        """
        raise NotImplementedError


#: Which Backend method each `method` value needs. `tss_major` needs
#: `forward` too, which is mandatory anyway.
METHOD_REQUIRES = {"gradient": "scalar_gradient",
                   "attention": "attention",
                   "ism": "forward"}


def backend_capabilities(backend: Backend) -> Dict[str, bool]:
    """Which of Backend's methods this backend actually implements.

    Detected by identity against the base class rather than by calling
    anything: probing by invocation would need a real input, a device
    and possibly a forward pass, which is exactly the expensive setup
    this check exists to run before.
    """
    cls = type(backend)
    names = ("forward", "scalar_gradient", "attention")
    declared = getattr(cls, "IMPLEMENTS", None)
    if declared is not None:
        return {n: n in declared for n in names}
    return {n: getattr(cls, n, None) is not getattr(Backend, n) for n in names}


def check_backend(theta: Dict[str, Any], backend: Backend) -> None:
    """Refuse a (config, backend) pair the backend cannot serve."""
    caps = backend_capabilities(backend)
    have = sorted(k for k, v in caps.items() if v)
    can = sorted(m for m, req in METHOD_REQUIRES.items()
                 if caps[req]) or ["nothing"]
    name = type(backend).__name__

    if not caps["forward"]:
        missing = backend.capability_note
        note = f"\n  {missing}" if missing else ""
        raise ConfigError(
            f"{name} does not implement `forward`, which is mandatory.{note}")

    need = METHOD_REQUIRES[theta["method"]]
    if not caps[need]:
        raise ConfigError(
            f"method={theta['method']!r} needs {name}.{need}(), "
            "which is not implemented.\n"
            f"  {name} implements: {', '.join(have)}\n"
            f"  so it can serve method= {', '.join(can)}\n"
            f"  Implement {need}() on {name}, or run one of those "
            "methods instead.")


class SyntheticBackend(Backend):
    """A deterministic stand-in with an exact Jacobian, for testing.

    forward is linear in the one-hot input:

        y[b,t] = sum_{i in bin b} sum_k x[i,k] * K[k,t] * w[i]

    so dy[b,t]/dx[i,k] = K[k,t] * w[i] exactly. The gradient of the
    SCALAR is then obtained by differentiating the gene scalar
    numerically over the (readout bins x selected tracks) subtensor it
    actually reads, and chaining. The gene scalar is therefore exercised
    verbatim rather than reimplemented, which is the point of the test.

    A real backend replaces scalar_gradient with a single autograd call.
    """

    def __init__(self, meta: ModelMeta, n_replicates: int = 1,
                 n_tracks: int = 32, seed: int = 0):
        # A 1 bp bin size (AlphaGenome) would make S an (L x L) matrix.
        # The stand-in only has to exercise control flow, so this
        # geometry is coarsened rather than mirrored faithfully.
        bs = max(meta.bin_size, 32)
        # The stand-in is deliberately UNCROPPED (output bins == input
        # bins): a faithful crop perturbs the receptive-field decay
        # check, which exists only to confirm the stand-in has a
        # receptive field at all. The input-grid/output-grid invariant
        # this geometry cannot exercise is asserted directly against the
        # three real ModelMetas in the self-test instead, and it is the
        # invariant that catches a readout placed on the wrong grid.
        self.meta = ModelMeta(
            name=meta.name, seq_len=SYNTH_SEQ_LEN, bin_size=bs,
            n_bins=SYNTH_SEQ_LEN // bs, has_attention=meta.has_attention,
            scalar_is_squashed=meta.scalar_is_squashed,
            transform_generation=meta.transform_generation,
            log_base=meta.log_base,
            has_strand_column=meta.has_strand_column,
            tracks={k: tuple(i % n_tracks for i in v)
                    for k, v in meta.tracks.items()},
            track_strand={i % n_tracks: s
                          for i, s in meta.track_strand.items()},
            track_strand_by_set={k: {i % n_tracks: s for i, s in v.items()}
                                 for k, v in meta.track_strand_by_set.items()},
            transform_constants=meta.transform_constants,
        )
        self.n_tracks = n_tracks
        self.n_replicates = max(1, n_replicates)
        self.seed = seed
        self.rf = 1500.0                  # receptive-field decay length, bp
        self._S: Optional[np.ndarray] = None
        self._cache: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

    @property
    def S(self) -> np.ndarray:
        """Receptive field in bin space, shape (n_bins, n_bins).

        Entry (i, j) is how much output bin i draws on bin j, decaying
        with genomic distance. Without a receptive field the model would
        be strictly local, the gradient at a distal enhancer would be
        exactly zero, and the distal half of every test vacuous.
        Applying the decay across bins rather than across base pairs is
        far cheaper for the same qualitative property, which matters
        because the ISM tests run very many forward passes.
        """
        if self._S is None:
            nb, bs = self.meta.n_bins, self.meta.bin_size
            b = np.arange(nb)
            self._S = np.exp(-np.abs(b[:, None] - b[None, :]) * bs / self.rf)
        return self._S

    def _params(self, r: int) -> Tuple[np.ndarray, np.ndarray]:
        """Draw one replicate's parameters.

        Each replicate is an independently drawn model, as real
        replicates are. That matters: it lets the sign of an ISM effect
        disagree between replicates, the only situation in which an
        ism_inner_mean containing `replicate` differs from one that does
        not.
        """
        if r not in self._cache:
            rng = np.random.RandomState(1000 * self.seed + r)
            # The offsets keep every prediction strictly positive.
            K = np.abs(rng.randn(4, self.n_tracks)) + 0.2
            w = np.abs(rng.randn(self.meta.seq_len)) + 0.05
            self._cache[r] = (K, w)
        return self._cache[r]

    def _replicate_ids(self, replicate: Any) -> Tuple[int, ...]:
        """The replicates named by `replicate`; ENSEMBLE means all."""
        if replicate == "ENSEMBLE":
            return tuple(range(1, self.n_replicates + 1))
        return (int(replicate),)

    def _forward_one(self, x: np.ndarray, r: int) -> np.ndarray:
        K, w = self._params(r)
        contrib = (x * w[:, None]) @ K                            # (L, T)
        nb, bs = self.meta.n_bins, self.meta.bin_size
        # The predicted bins cover the CENTRE of the input, not its
        # start. Taking contrib[:nb*bs] would put output bin 0 at input
        # bp 0, confusing the input grid with the output grid and making
        # this stand-in disagree with the geometry every real backend
        # has.
        lo = self.meta.crop_bins * bs             # 0 for the stand-in
        per_bin = (contrib[lo: lo + nb * bs]
                   .reshape(nb, bs, self.n_tracks).sum(axis=1))
        return self.S @ per_bin + 0.05                            # (n_bins, T)

    def forward(self, x: np.ndarray, replicate: Any) -> np.ndarray:
        ids = self._replicate_ids(replicate)
        out = self._forward_one(x, ids[0])
        for r in ids[1:]:
            out = out + self._forward_one(x, r)
        return out / len(ids)

    @staticmethod
    def _dscalar_dy(y: np.ndarray,
                    scalar_fn: Callable[[np.ndarray], float]) -> np.ndarray:
        """dU/dy, the gene scalar differentiated over the predictions.

        By central differences, on exactly the entries the gene scalar
        reads: it advertises them, and every other entry has derivative
        exactly zero.
        """
        bins = getattr(scalar_fn, "bins", range(y.shape[0]))
        tracks = getattr(scalar_fn, "tracks", range(y.shape[1]))
        dUdy = np.zeros_like(y)
        h = 1e-4
        for b in bins:
            for t in tracks:
                yp = y.copy()
                yp[b, t] += h
                ym = y.copy()
                ym[b, t] -= h
                dUdy[b, t] = (scalar_fn(yp) - scalar_fn(ym)) / (2 * h)
        return dUdy

    def scalar_gradient(
            self, x: np.ndarray, replicate: Any,
            scalar_fn: Callable[[np.ndarray], float]) -> np.ndarray:
        y = self.forward(x, replicate)
        self._last_u = float(scalar_fn(y))
        self._last_T = float(getattr(scalar_fn, "linear", scalar_fn)(y))
        if not math.isfinite(scalar_fn(y)):
            raise ConfigError(
                "the gene scalar is not finite on the reference sequence")
        dUdy = self._dscalar_dy(y, scalar_fn)
        # Chain through the receptive field:
        #   dU/dx[i,k] = w[i] * sum_t K[k,t] * (S^T dU/dy)[bin(i), t]
        # averaged over whatever replicates `forward` averaged over.
        nb, bs = self.meta.n_bins, self.meta.bin_size
        per_pos = np.repeat(self.S.T @ dUdy, bs, axis=0)           # (nb*bs, T)
        ids = self._replicate_ids(replicate)
        grad = np.zeros_like(x)
        for r in ids:
            K, w = self._params(r)
            grad[: nb * bs] += (per_pos @ K.T) * w[: nb * bs, None]
        return grad / len(ids)

    def attention(self, x: np.ndarray, replicate: Any, how: str) -> np.ndarray:
        """A distance kernel on the bin grid, rows summing to one.

        The stand-in has one head and one layer, so `how` has nothing
        to reduce over and is ignored.
        """
        n = self.meta.n_bins
        idx = np.arange(n)
        A = np.exp(-np.abs(idx[:, None] - idx[None, :]) / 12.0)
        return A / A.sum(axis=1, keepdims=True)


class UnavailableBackend(Backend):
    """A model for which this repository ships no adapter.

    Every entry point raises, naming the weights and packages an
    adapter would need.
    """

    # Declared rather than detected. The three methods below are all
    # overridden, so inferring capabilities by comparing them against
    # the base class would report this backend as fully capable and
    # check_backend would wave it through.
    IMPLEMENTS: FrozenSet[str] = frozenset()

    def __init__(self, meta: ModelMeta, what: str):
        self.meta = meta
        self.what = what
        self.capability_note = (
            f"no backend is wired up for {meta.name}. Needed: {what}")

    def _die(self, verb: str):
        raise NotImplementedError(
            f"{self.meta.name}: no {verb} backend wired up.\n"
            f"  Needed: {self.what}\n"
            "  Run with --backend synthetic to exercise the algorithm "
            "without weights.")

    def forward(self, *a, **k): self._die("forward")
    def scalar_gradient(self, *a, **k): self._die("gradient")
    def attention(self, *a, **k): self._die("attention")


#: What an adapter would need, per model, quoted back when one is
#: missing.
BACKEND_REQUIREMENTS = {
    "enformer": "the enformer-pytorch port, with attention matrices "
                "retained per layer and head",
    "borzoi": "borzoi-pytorch with the four replicate checkpoints; "
              "the crop is 16 bins, not 5,120",
    "alphagenome": "the AlphaGenome client for forward passes; input "
                   "gradients need local weights, which the public API "
                   "does not expose",
}


def _borzoi_narrow_tracks(theta: Dict[str, Any], meta: ModelMeta,
                          rho: Resolved) -> List[int]:
    """Every track index a Borzoi run may be asked to read.

    Borzoi's `forward` computes only the tracks it is told to and
    leaves the rest NaN, so the readout set alone is not enough: a
    major-TSS assay adds a further track set. Both come from the same
    resolver the algorithm itself uses, so the
    backend and the score cannot disagree about which tracks matter.
    """
    narrow = set(rho.tracks)
    if theta["tss_major"]:
        key = (theta["tss_major_assay"], major_tss_cell_type(theta))
        narrow |= set(meta.tracks[key])
    return sorted(narrow)


def make_backend(theta: Dict[str, Any], which: str, device: str = "cuda",
                 fasta: Optional[str] = None) -> Backend:
    """Build the backend named by `theta["model"]`, or a stand-in."""
    meta = MODEL_META[theta["model"]]
    if which == "synthetic":
        return SyntheticBackend(meta, n_replicates=theta["replicates"],
                                seed=theta["seed"])
    if theta["model"] == "enformer":
        from backend_enformer import EnformerBackend
        # Passed explicitly rather than left to the adapter's own
        # defaults: the output geometry follows `meta`, and the gamma
        # convention is pinned module-wide.
        return EnformerBackend(meta, device=device, target_length=meta.n_bins,
                               use_tf_gamma=FROZEN_USE_TF_GAMMA)
    if theta["model"] == "alphagenome":
        from backend_alphagenome import AlphaGenomeBackend
        return AlphaGenomeBackend(meta, theta, device=device, fasta=fasta)
    if theta["model"] == "borzoi":
        from backend_borzoi import BorzoiBackend
        rho = resolve(theta, meta)
        narrow = _borzoi_narrow_tracks(theta, meta, rho)
        return BorzoiBackend(meta, theta, rho, device=device,
                             narrow_tracks=narrow)
    return UnavailableBackend(meta, BACKEND_REQUIREMENTS[theta["model"]])


# ============================================================================
# 6. THE ALGORITHM
# ============================================================================

@dataclass(frozen=True)
class Aug:
    """A shift in bp plus an optional reverse complement."""

    shift: int = 0
    rc: bool = False


# An `Aug` field the readout frame does not map would be applied to the input and ignored by the
# readout. Refuse at import; `frame_for`, `augment`, `unaugment`, `augment_position` and the frame
# probe must all learn a new augmentation kind before it can exist.
if tuple(f.name for f in _dc_fields(Aug)) != ("shift", "rc"):
    raise RuntimeError("Aug gained a field that the readout frame does not map")


#: How many augmentations `tta = avsec8` averages.
#:
#: Avsec 2021: "we averaged the predictions from 8 sequences randomly
#: augmented the same way as during training (<=3 bp shifts and
#: reverse-complementation)". That training augmentation draws the
#: shift and the orientation independently per sequence, so the eight
#: are i.i.d. with replacement over a 2(2k+1) = 14-cell space rather
#: than a subset of 8 distinct cells. At k = 3 a draw of eight repeats
#: at least one cell 92% of the time, and the repeat carries double
#: weight in the mean, which is what independent draws mean. The other
#: `tta` values cannot express that: they enumerate at most 3 shifts x
#: 2 orientations and have no RNG on the path.
#:
#: 8 is a constant rather than a parameter because it is the paper's
#: constant. An arm averaging some other number is not this protocol
#: and should say so in its own `tta` value.
AVSEC8_N = 8


def avsec8_seed(seed: int, key: str) -> int:
    """Per-sequence RNG seed for `tta = avsec8`, free of run order.

    Not `hash()`, which Python salts per process, so running the arm
    twice would draw different augmentations and no manifest could pin
    it down. A single global stream fails more quietly: the draw for
    gene i would depend on how many genes ran before it, so a shard
    would average a different eight than the unsharded run and the two
    would stop being bit-identical. It would also break the equality
    between the whole-run driver and `score_pair`, which scores one
    pair in isolation.

    Deriving the seed from (theta.seed, key) makes the draw a pure
    function of the gene, which keeps all three properties. `seed` is
    already part of the cache key, so two runs differing only in it
    stay separate.
    """
    h = hashlib.blake2b(f"avsec8|{seed}|{key}".encode(), digest_size=8)
    return int.from_bytes(h.digest(), "big")


def augmentations(theta: Dict[str, Any],
                  key: Optional[str] = None) -> Tuple[Aug, ...]:
    """The augmentations averaged for one input sequence.

    `key` names that sequence: the gene id, since one gene gives one
    input window under every gene-derived anchor. Only `tta = avsec8`
    reads it; the other four values are deterministic enumerations,
    identical for every sequence in the run, and leave `key` unused.
    """
    if theta["tta"] == "avsec8":
        if key is None:
            raise ConfigError(
                "tta=avsec8 draws its augmentations per input "
                "sequence, so it needs that sequence's key: call "
                "augmentations(theta, gene.gid)")
        k = theta["tta_shift_bp"]
        rng = np.random.default_rng(avsec8_seed(theta["seed"], key))
        # One stream, shifts drawn before orientations, so the pairing
        # is fixed by the seed alone.
        #
        # The three details below follow Basenji2's training layers
        # rather than the paper's prose, since "the same data
        # augmentation as Basenji2" is a pointer to that code:
        #   * the shift is drawn uniformly over range(-k, k+1), seven
        #     values at k = 3 and zero among them, not over the three
        #     values {-k, 0, +k} that `tta = shift` enumerates;
        #   * the reverse complement is a separate Bernoulli(1/2), so
        #     shift and orientation are independent, as they are here;
        #   * a shifted sequence is padded with zeros. `augment` below
        #     rolls and then zeroes the vacated end, which is the same
        #     array: np.roll(x, s) with y[:s] = 0 equals
        #     concat([zeros(s), x[:-s]]) for s > 0, and likewise on the
        #     left. So this arm stays a one-axis contrast against the
        #     enumerated shift arms.
        return tuple(Aug(shift=int(sh), rc=bool(rc))
                     for sh, rc in zip(rng.integers(-k, k + 1, size=AVSEC8_N),
                                       rng.random(AVSEC8_N) < 0.5))
    k = theta["tta_shift_bp"]
    shifts = (0,) if k is None else (-k, 0, k)
    rcs = (False, True) if "rc" in theta["tta"] else (False,)
    return tuple(Aug(shift=s, rc=r) for r in rcs for s in shifts)


def augment(x: np.ndarray, a: Aug) -> np.ndarray:
    """Apply one augmentation, zero-filling the end a shift vacates."""
    y = np.roll(x, a.shift, axis=0)
    if a.shift > 0:
        y[: a.shift] = 0
    elif a.shift < 0:
        y[a.shift:] = 0
    if a.rc:
        # Reversing the channel axis complements: A, C, G, T read
        # backwards is T, G, C, A. So one flip of both axes is the
        # whole reverse complement.
        y = y[::-1, ::-1].copy()
    return y


def unaugment(m: np.ndarray, a: Aug) -> np.ndarray:
    """Map a positional map back to unaugmented window coordinates."""
    y = m[::-1].copy() if a.rc else m
    y = np.roll(y, -a.shift, axis=0)
    return y


def augment_position(i: int, a: Aug, length: int) -> int:
    """Map one window-relative position through an augmentation."""
    j = i + a.shift
    return length - 1 - j if a.rc else j


def augment_interval(i: int, j: int, a: Aug, length: int) -> Tuple[int, int]:
    """Map a half-open interval [i, j) through an augmentation.

    Mapping the two endpoints independently is not enough: under `rc`
    the image of the start is the larger coordinate, so the pair comes
    back reversed and a slice on it would be empty and silently perturb
    nothing. Map both, then sort. Mapping `j - 1` rather than `j` keeps
    the interval half-open; `j` itself lands one base past the image
    and widens the span by one under `rc`.
    """
    u = augment_position(i, a, length)
    v = augment_position(j - 1, a, length)
    lo, hi = (v, u) if a.rc else (u, v)
    return lo, hi + 1


def reduce_over(values: Sequence[float], op: str) -> float:
    """Reduce a sequence to one float by sum, mean or max."""
    arr = np.asarray(values, dtype=np.float64)
    if op == "sum":
        return float(arr.sum())
    if op == "mean":
        return float(arr.mean())
    if op == "max":
        return float(arr.max())
    raise ConfigError(f"unknown reduction {op!r}")


def out_of_window(theta: Dict[str, Any]):
    """The score an unreachable pair gets: 0.0, or ABSENT to drop it."""
    return 0.0 if theta["out_of_window"] == "zero" else ABSENT


def pair_in_window(enh: Enhancer, windows: Sequence[Window]) -> bool:
    """Is the enhancer midpoint inside any anchor window?

    Reachability is a property of the pair and the model, not of the
    scoring policy, which is why it lives in its own function. Under
    `out_of_window=zero` an unreachable pair is written as `score=0.0`,
    indistinguishable in the output from a genuine attribution of zero;
    under `out_of_window=drop` the row vanishes. Neither output lets a
    reader tell "the model cannot see this enhancer" from "the model
    sees it and ranks it last", and across models whose receptive
    fields differ by 5x (Enformer 196 kb, Borzoi 524 kb, AlphaGenome
    1 Mb) that is the dominant confounder. On the Gschwind cohort the
    reachable counts are 1,798 / 3,989 / 6,694, so a full-cohort auPRC
    partly measures receptive field rather than ranking skill.

    Every row therefore carries `in_window` from this test, whatever
    `out_of_window` says, so one run supports all four quantities:
      * common-support auPRC, over pairs every model compared reaches
      * reachable-support auPRC, each model judged where it can see
      * full-cohort auPRC, unreachable scored 0, receptive field in
      * coverage, the fraction of positives and negatives reachable
    It also makes `out_of_window` a post-hoc filter rather than a
    run-time axis, since `drop` is `in_window == True`.

    The test is `0 <= mid - start < length`, matching the gene-level
    skip in `run_experiment`. `reduce_pair` writes the same predicate
    against `M.shape[0]` rather than `w.length`; the two agree for
    gradient and ISM maps, which hold one entry per base, and
    `--self-test` asserts rather than assumes it by checking that the
    rows surviving `out_of_window=drop` are exactly the reachable ones.
    """
    return any(0 <= enh.mid() - w.start < w.length for w in windows)


# ---- A.1 choosing the window ----------------------------------------------

def anchor_windows(enh: Optional[Enhancer], gene: Gene, theta: Dict[str, Any],
                   rho: Resolved, backend: Backend) -> Tuple[Window, ...]:
    """The input windows this pair is scored in, one per anchor."""
    L = backend.meta.seq_len
    mode = theta["anchor"]
    if mode == "single-tss":
        c = (major_tss(gene, theta, rho, backend) if theta["tss_major"]
             else gene.provided_tss)
        return (window_at(c, L),)
    if mode == "each-tss":
        return tuple(window_at(t, L) for t in gene.gencode_tss)
    if mode == "gene-midpoint":
        # No exon record means no centroid and so no window, the same
        # answer `each-tss` gives a gene with an empty TSS set: the
        # pairs fall through to the out-of-window policy. Raising
        # instead would kill the run on the first screen row whose
        # symbol GENCODE has renamed, and it is not a different
        # scientific decision, since the reference implementation's own
        # gene annotation cannot contain such a gene either.
        if not gene.exons:
            return ()
        return (window_at(exon_centroid(gene), L),)
    if enh is None:
        raise ConfigError(
            f"anchor={mode} places the window from the enhancer, so it "
            "cannot be resolved for a gene on its own")
    if mode == "enhancer":
        return (window_at(enh.mid(), L),)
    if mode == "enhancer-tss-midpoint":
        return (window_at((enh.mid() + gene.provided_tss) // 2, L),)
    raise ConfigError(f"unknown anchor {mode!r}")


def major_tss(gene: Gene, theta: Dict[str, Any], rho: Resolved,
              backend: Backend) -> int:
    """The gene's dominant TSS: one forward pass per candidate, argmax.

    Returns the precomputed anchor when `--anchors` supplied one. That
    is the same quantity read from cache rather than recomputed; the
    search depends only on the model, the annotation and the anchor
    track set, so without the cache every run of every method would
    repeat the same forward passes over the whole cohort.
    """
    if gene.anchor_tss is not None:
        return gene.anchor_tss
    L = backend.meta.seq_len
    # The anchor's tracks, not the readout's. For every K562 arm these
    # are the same set; for the cell-type-agnostic arms they are not,
    # and using the readout's 638 tracks would argmax over a different
    # quantity than the one Avsec 2021 defines the major TSS by.
    tracks = backend.meta.tracks[(theta["tss_major_assay"],
                                 major_tss_cell_type(theta))]
    best, best_t = -math.inf, None
    for t in gene.gencode_tss:
        w = window_at(t, L)
        y = backend.forward(one_hot(w, gene, backend), replicate=1)
        b = bin_of(t, w, backend.meta)
        if b is None:
            continue
        v = reduce_over([y[b, k] for k in tracks], theta["track_agg"])
        if v > best:
            best, best_t = v, t
    if best_t is None:
        raise ConfigError(f"no usable candidate TSS for {gene.gid}")
    return best_t


def stable_seed(*parts: Any) -> int:
    """A reproducible 31-bit seed from arbitrary components.

    Not `hash()`: Python salts `str.__hash__` per process, so an ISM
    run seeded that way draws different perturbations on every
    invocation despite `seed: 0`. That bites hardest exactly where this
    is used. Avsec 2021 sets `ism_shuffles: 1`, so a pair's score is
    |f(ref) - f(one particular random sequence)| with no averaging to
    damp the variance, and the run would not be reproducible at all.
    """
    payload = "|".join(repr(x) for x in parts).encode()
    return int(hashlib.sha256(payload).hexdigest()[:8], 16) % (2 ** 31)


def bin_of(pos: int, w: Window, meta: ModelMeta) -> Optional[int]:
    """Genomic position to output-bin index, or None if unpredicted.

    Two grids exist and they are not the same:
      * the input or token grid, `seq_len // bin_size` wide (Enformer:
        1,536), which is what attention indexes, because attention runs
        before the crop;
      * the output grid, `n_bins` wide (Enformer: 896), which is what
        the predicted tracks index.

    The output grid is the centre of the input grid, offset by
    `meta.crop_bins`. Subtracting that offset is not optional: for a
    window centred on the anchor, `off // bin_size` gives 768 while the
    anchor's output bin is 448, so omitting it reads the signal
    40,960 bp away. Nothing raises, because 768 is a legal index into
    an 896-bin array, which is why this returns None rather than
    clamping. A clamp would turn "outside the predicted region" into
    "the last bin", and the caller could not tell the two apart.
    """
    off = pos - w.start
    if off < 0 or off >= w.length:
        return None
    b = off // meta.bin_size - meta.crop_bins
    return b if 0 <= b < meta.n_bins else None


# ---- A.2 the gene scalar --------------------------------------------------

def unsquash(y: np.ndarray, rho: Resolved) -> np.ndarray:
    """Invert the squashing a model applies to its training targets.

    Model generations differ in the form of that transform, so the
    branch is on `transform_generation` rather than on the model name.
    """
    cs, e, sc = rho.clip_soft, rho.exponent, rho.scale
    if rho.meta.transform_generation == "old":
        y = y / sc
        if cs is not None:
            y = np.where(y > cs, (y - cs) ** 2 + cs, y)
        return np.power(np.maximum(y, 0.0), 1.0 / e)
    if cs is not None:
        y = np.where(y > cs, (y - cs + 1) ** 2 + cs - 1, y)
    y = -1.0 + np.power(np.maximum(y + 1.0, 0.0), 1.0 / e)
    return y / sc


def readout_bins(gene: Gene, w: Window, theta: Dict[str, Any],
                 meta: ModelMeta) -> Tuple[int, ...]:
    """The output bins the gene scalar reads."""
    if theta["region"] == "exons":
        # Mirrors baskerville's gene `output_slice`. Its
        # `majority_overlap` argument defaults to False, and every
        # CRISPRi call site takes that default, so `any` is the
        # released behaviour. `fractional` has no meaning for a bin
        # *set*, and validation refuses it here.
        #
        # `- meta.crop_bins` moves the exon onto the output grid.
        # baskerville measures an exon from the first PREDICTED base
        # rather than the first input base, and subtracting the crop is
        # the same arithmetic. It is exact rather than approximate
        # because `crop_bins * bin_size` is a whole number of bins:
        # floor((lo - c*bs)/bs) == floor(lo/bs) - c. Getting it wrong
        # is expensive and silent: Borzoi's crop is 16 bins = 512 bp,
        # and reading the paper's 6,144-bin geometry instead puts the
        # readout 5,120 bins = 163,840 bp away.
        out = set()
        bs = meta.bin_size
        for begin, end in gene.exons:
            lo = max(0, begin - w.start)
            hi = max(0, min(end, w.start + w.length) - w.start)
            if lo >= hi:
                continue                          # exon outside window
            if theta["bin_overlap"] == "majority":  # >50% overlap
                b0, b1 = int(round(lo / bs)), int(round(hi / bs))
            else:                                  # any overlap
                b0, b1 = int(math.floor(lo / bs)), int(math.ceil(hi / bs))
            b0, b1 = b0 - meta.crop_bins, b1 - meta.crop_bins
            out.update(b for b in range(b0, b1) if 0 <= b < meta.n_bins)
        return tuple(sorted(out))
    k = theta["tss_width"] // meta.bin_size

    def bins_at(pos: int) -> Tuple[int, ...]:
        """The k bins centred on `pos`, or () if `pos` is unpredicted.

        Factored out so `tss` and `all-tss` cannot drift apart: they
        place an identical window, differing only in how many positions
        it is placed at.
        """
        i0 = bin_of(pos, w, meta)
        if i0 is None:
            return ()
        downstream = 1 if gene.strand == "+" else -1
        lo = i0 - (k - 1) // 2
        if k % 2 == 0 and downstream < 0:
            lo -= 1
        return tuple(b for b in range(lo, lo + k) if 0 <= b < meta.n_bins)

    if theta["region"] == "all-tss":
        # The readout half of `each-tss`, without the re-anchoring.
        # `anchor=each-tss` moves the input window to every TSS and
        # averages the resulting scores, at 15.7x the model calls; this
        # places one window and sums the readout over every TSS inside
        # it, at 1x. Separating the two halves is what makes either
        # interpretable: varying both at once measures a null
        # (-0.0143, p = 0.33) that cannot say which half was
        # responsible.
        #
        # `bin_of` returns None outside the predicted region, so "as
        # many TSSs as fit" is enforced by construction rather than by
        # a distance test, and the bound is the predicted extent
        # (Enformer 114,688 bp) rather than the 1.7x larger input
        # length (196,608 bp).
        #
        # The anchor's own bins are included unconditionally, because
        # `gene.provided_tss` is the screen's column and need not
        # appear in GENCODE's TSS set. A pure union over the annotation
        # could therefore omit the very bins the `tss` readout uses,
        # leaving `all-tss` not a superset of it. Adding the anchor
        # makes the two nest: same window, same anchor, more readout.
        out = set(bins_at(w.centre))
        for t in gene.gencode_tss:
            out.update(bins_at(t))
        return tuple(sorted(out))

    # `window-centre`, the default, reads wherever the anchor put the
    # window; `gene-tss` pins the readout to the gene while the anchor
    # still places the window.
    return bins_at(w.centre if theta["readout_anchor"] == "window-centre"
                   else gene.provided_tss)


# ---- THE FRAME GUARD ------------------------------------------------------------------------------------
# An augmentation moves the INPUT, so every read of the model's OUTPUT on that pass -- the gene's readout
# bins, the strand of its tracks, the attention query, the ISM lesion -- must be mapped into the same
# frame. Four layers make it impossible to get this wrong on any of the three models:
#   1. ONE MAPPING. `frame_for` is the only place an augmentation becomes a readout frame (strand,
#      whole-bin offset, mirror); GENE-SCALAR and the attention query take their bins from
#      `ReadoutFrame.map_bins`.
#   2. NO SILENT DEFAULT. attribution_map and ism_map run each model pass inside `augmented_frame(a)`;
#      a readout built there for any other augmentation (a forgotten `aug=` included) raises FrameError.
#   3. AN EXACT PROBE ON THE REAL GEOMETRY. `frame_selftest` drives the production readout, gradient-map,
#      ISM and attention code through `FrameProbeBackend`, which is exactly rc- and whole-bin-translation
#      equivariant on the model's own geometry, before the first model call of every run.
#   4. THE REAL MODEL AGREES WITH ITSELF. `FrameAgreementGate` compares each gene's rc readout with its
#      forward readout as the run goes, from the passes it makes anyway.


class FrameError(ConfigError):
    """An augmentation and the readout disagree about which frame the model's output is in."""


#: The fields of `Aug` the frame mapping knows (checked at import, under the `Aug` definition).
_FRAME_FIELDS = ("shift", "rc")


@dataclass(frozen=True)
class ReadoutFrame:
    """Where the gene's readout sits in the OUTPUT of a pass on an augmented input: the strand whose
    tracks carry the gene (swapped under rc), a whole-bin offset (`shift_bins`, applied first) and a
    mirror (rc, applied second) -- the order `augment` shifts and then reverse-complements in."""
    strand: str
    offset: int
    mirror: bool

    def map_bins(self, bins: Iterable[int], meta: "ModelMeta") -> Tuple[int, ...]:
        out = [b + self.offset for b in bins]
        if self.mirror:
            out = [meta.n_bins - 1 - b for b in out]
        bad = [b for b in out if not 0 <= b < meta.n_bins]
        if bad:
            raise FrameError(f"readout bins {bad[:5]} fall off the {meta.n_bins}-bin output under {self}")
        return tuple(sorted(out))


def shift_bins(shift: int, meta: "ModelMeta") -> int:
    """Whole OUTPUT bins an input shift of `shift` bp moves the gene's readout by: 0 for a shift under half
    a bin (the readout stays, as Basenji, Avsec 2021 and calico define shift augmentation), shift //
    bin_size for an exact multiple of the bin size (the readout follows; AlphaGenome's 1 bp heads make every
    shift one). Anything in between is refused: the readout would sit a fraction of a bin off."""
    bs = meta.bin_size
    if shift % bs == 0:
        return shift // bs
    if 2 * abs(shift) < bs:
        return 0
    raise FrameError(f"a {shift} bp shift on {meta.name}'s {bs} bp output bins is neither under half a bin "
                     f"nor a whole number of bins; use |shift| < {bs // 2 + (bs % 2)} or a multiple of {bs}")


#: The augmentation of the model pass currently running, innermost last (`augmented_frame`).
_FRAME_STACK: List[Optional["Aug"]] = []


def _aug_key(a: Optional["Aug"]) -> Tuple[int, bool]:
    return (0, False) if a is None else (int(a.shift), bool(a.rc))


class augmented_frame:
    """Context: every model pass inside runs on an input augmented by `a`. A readout built for any other
    augmentation inside it -- above all the `aug=None` a forgotten argument leaves -- raises FrameError."""

    def __init__(self, a: Optional["Aug"]):
        self.a = a

    def __enter__(self):
        _FRAME_STACK.append(self.a)
        return self

    def __exit__(self, *exc):
        _FRAME_STACK.pop()
        return False


def frame_for(gene: "Gene", aug: Optional["Aug"], meta: "ModelMeta") -> ReadoutFrame:
    """THE mapping from an augmentation to the frame the gene's readout must be read in. `aug=None` is the
    forward reference."""
    if _FRAME_STACK and _aug_key(aug) != _aug_key(_FRAME_STACK[-1]):
        raise FrameError(f"the readout of {gene.gid} was built for augmentation {aug!r} inside a model pass "
                         f"on an input augmented by {_FRAME_STACK[-1]!r}; pass `aug=` the pass's own one")
    if aug is None:
        return ReadoutFrame(gene.strand, 0, False)
    off = shift_bins(aug.shift, meta)
    if not aug.rc:
        return ReadoutFrame(gene.strand, off, False)
    return ReadoutFrame({"+": "-", "-": "+"}.get(gene.strand, gene.strand), off, True)


def readout_frame(gene: "Gene", aug: Optional["Aug"], meta: "ModelMeta") -> Tuple[str, bool]:
    """(strand to read, whether the readout bins are mirrored); the bins come from `frame_for(...).map_bins`."""
    fr = frame_for(gene, aug, meta)
    return fr.strand, fr.mirror


def gene_scalar_fn(gene: Gene, w: Window, theta: Dict[str, Any],
                   rho: Resolved, only_bin: Optional[int] = None,
                   per_track: bool = False, group_only: Optional[int] = None,
                   aug: Optional["Aug"] = None) -> Callable[[np.ndarray], Any]:
    """The gene's scalar readout, curried for differentiation.

    `only_bin` restricts the readout to a single bin. It exists for
    `bin_abs_order: before`, which needs one gradient per readout bin
    rather than one gradient of the summed readout, and is otherwise
    unused.

    `group_only` restricts an `assay=multi` readout to a single assay
    group, by index into MULTI_GROUPS. It exists for `--multi-channels`,
    which needs one backward pass per assay so that every weighting
    over the assays becomes a post-map reduction instead of a separate
    run. The scalar is then that assay's own term of the sum
    `assay=multi` otherwise returns, so summing the N gradients
    reproduces the single-pass map exactly; that identity is the wiring
    check on this path.

    `per_track` skips the cross-track reduction and returns a vector
    with one entry per resolved track instead of a float. It exists for
    `track_abs_order: before`, which needs each track's signed
    difference before the absolute value, and is otherwise unused. The
    returned callable is then not differentiable to a scalar, so only
    the ISM arm may ask for it.

    `aug` is the augmentation the INPUT went through (None = the forward reference). The readout bins and
    the strand mask are resolved in the forward frame and then moved into the augmented one by
    `frame_for`, so `only_bin` is always given in forward-frame coordinates.
    """
    meta = rho.meta
    frame = frame_for(gene, aug, meta)
    tracks = list(rho.tracks)
    if rho.strand_mask_on:
        # Paired with rho.track_strands rather than re-indexed out of
        # meta.track_strand: the tuple was resolved against this track
        # set, so it cannot pick up another set's strand.
        tracks = [t for t, st in zip(tracks, rho.track_strands)
                  if st in (None, frame.strand)]
    if not tracks:
        raise ConfigError(
            f"strand mask removed every track for {gene.gid} "
            f"({gene.strand})")
    bins = readout_bins(gene, w, theta, meta)
    if not bins:
        raise ConfigError(f"no readout bins for {gene.gid} in this window")
    if only_bin is not None:
        if only_bin not in bins:
            raise ConfigError(
                f"only_bin={only_bin} is not a readout bin of "
                f"{gene.gid}: {bins}")
        bins = (only_bin,)
    bins = frame.map_bins(bins, meta)      # into the augmented input's frame (identity for aug=None)

    def per_track_readout(z: np.ndarray) -> np.ndarray:
        """One value per resolved track, the cross-track step skipped.

        The same operations as the scalar path, one axis over: bins are
        axis 0 of z[bins], tracks are axis 1. The caller takes one
        absolute value per track.
        """
        tot = _reduce_axis(np.asarray(z, dtype=np.float64)[list(bins), :],
                           axis=0, op=FROZEN_BIN_AGG)
        if not theta["log_scalar"]:
            return tot
        arg = tot + FROZEN_PSEUDOCOUNT + LOG_EPS
        with np.errstate(divide="ignore", invalid="ignore"):
            lg = np.log2(arg) if meta.log_base == "2" else np.log(arg)
        return np.where(arg > 0.0, lg, -np.inf)

    def multi_assay_readout(z: np.ndarray) -> float:
        """`assay=multi`: log each assay separately, then sum,

            u = sum_a log( agg_bins( agg_tracks( E_a ) ) + pc )

        which makes the readout scale-free; ENFORMER_MULTI_GROUPS says
        why. Under `group_only` the sum stops at that one assay and its
        term is returned RAW, deliberately not logged: the caller forms
        the scale-free channel itself as dE_a / E_a. Returning
        log(E_a) here would make channel a scale-free already and
        divide channel N+a by E_a a second time, emptying out any
        weighting of the second half. Summing the logs is the
        combination step and has meaning only across groups; one
        group's readout is its raw value.
        """
        groups = MULTI_GROUPS.get(
            (meta.name, theta["assay"], theta["cell_type"]), ())
        n_want = sum(n for _, n in groups)
        if n_want != len(tracks):
            raise ConfigError(
                f"assay=multi group sizes sum to {n_want} but "
                f"{len(tracks)} tracks resolved for {gene.gid}: the "
                "contiguous-slice assumption is broken (a strand "
                "mask, or a changed declaration), so the per-assay "
                "split would silently mis-group tracks")
        acc, off = 0.0, 0
        for gi, (_name, gn) in enumerate(groups):
            if group_only is not None and gi != group_only:
                off += gn
                continue
            gz = z[:, off:off + gn]
            off += gn
            gper = _reduce_axis(gz, axis=1, op=theta["track_agg"])
            gtot = reduce_over([gper[b] for b in bins], FROZEN_BIN_AGG)
            if group_only is not None:
                return float(gtot)
            garg = gtot + FROZEN_PSEUDOCOUNT + LOG_EPS
            if garg <= 0.0:
                return -math.inf
            acc = acc + (math.log(garg, 2) if meta.log_base == "2"
                         else math.log(garg))
        return float(acc)

    def scalar(y: np.ndarray) -> Any:
        z = y[:, tracks]
        if theta["inverse_transform"]:
            z = unsquash(z, rho)
        if per_track:
            return per_track_readout(z)
        # Gated on the assay value, so every other arm takes the plain
        # path below bit for bit.
        if theta["assay"] == "multi":
            return multi_assay_readout(z)
        per_bin = _reduce_axis(z, axis=1, op=theta["track_agg"])
        total = reduce_over([per_bin[b] for b in bins], FROZEN_BIN_AGG)
        if not theta["log_scalar"]:
            return float(total)
        arg = total + FROZEN_PSEUDOCOUNT + LOG_EPS
        if arg <= 0.0:
            return -math.inf
        return float(math.log(arg, 2) if meta.log_base == "2"
                     else math.log(arg))

    # Advertise exactly which entries of `y` the scalar reads. A real
    # autograd backend ignores these; the stand-in uses them to
    # differentiate only the entries that matter.
    scalar.bins = bins            # type: ignore[attr-defined]
    scalar.tracks = tuple(tracks)  # type: ignore[attr-defined]
    # The transform constants belong to the track set, not to the run:
    # ('cage', 'k562') is (1.0, 1.0, 384) while ('rnaseq', 'k562') is
    # (0.3, 0.75, 384). A backend that mirrors this scalar has to
    # un-squash with this readout's own constants, because one backend
    # can differentiate two readouts while its own `rho` holds only one
    # of them.
    scalar.xform = (  # type: ignore[attr-defined]
        rho.scale, rho.exponent, rho.clip_soft)
    # The frame the bins and tracks were mapped into.
    scalar.frame = frame  # type: ignore[attr-defined]
    # The reduction chain, declared so a backend can form dU/dy in closed form when the scalar is
    # linear-then-log, and gate it against this closure (backend_enformer). `transform_generation ==
    # "none"` means un-squashing is the identity whatever `inverse_transform` says.
    scalar.ops = {"track_agg": theta["track_agg"],  # type: ignore[attr-defined]
                  "log_scalar": bool(theta["log_scalar"]),
                  "inverse_transform": bool(theta["inverse_transform"]),
                  "transform_generation": meta.transform_generation, "log_base": meta.log_base,
                  "multi": theta["assay"] == "multi", "per_track": bool(per_track),
                  "only_bin": only_bin, "pseudocount": FROZEN_PSEUDOCOUNT, "eps": LOG_EPS}
    # The un-logged readout T beside the scalar that is differentiated; backends report it per pass
    # (`_last_T`), which the frame agreement gate reads.
    if theta["log_scalar"] and theta["assay"] != "multi" and not per_track:
        scalar.linear = gene_scalar_fn(  # type: ignore[attr-defined]
            gene, w, dict(theta, log_scalar=False), rho, only_bin=only_bin,
            per_track=False, group_only=group_only, aug=aug)
    else:
        scalar.linear = scalar  # type: ignore[attr-defined]
    return scalar


def _reduce_axis(a: np.ndarray, axis: int, op: str) -> np.ndarray:
    """`reduce_over` one axis of an array instead of a sequence."""
    if op == "sum":
        return a.sum(axis=axis)
    if op == "mean":
        return a.mean(axis=axis)
    if op == "max":
        return a.max(axis=axis)
    raise ConfigError(f"unknown reduction {op!r}")


# ---- A.3 the attribution map ---------------------------------------------

# --- reference genome ------------------------------------------------------
# Set by --fasta, and deliberately not a parameter in PARAMS: which
# FASTA file sits on this disk is a runtime fact rather than a
# scientific choice, so it belongs in the manifest's runtime block and
# not in the cache key.
_REFERENCE: Dict[str, Any] = {"path": None, "handle": None}
_BASE_INDEX = {"A": 0, "C": 1, "G": 2, "T": 3, "a": 0, "c": 1, "g": 2, "t": 3}
#: The same map as a 256-entry table over byte values, for `one_hot`; -1 = not a base (N etc).
_BASE_LUT = np.full(256, -1, dtype=np.int64)
for _c, _i in _BASE_INDEX.items():
    _BASE_LUT[ord(_c)] = _i


def set_reference(path: Optional[str]) -> None:
    """Name the indexed FASTA that `one_hot` reads. Called once."""
    _REFERENCE["path"] = path
    _REFERENCE["handle"] = None
    if path and not os.path.exists(path):
        raise ConfigError(f"reference FASTA not found: {path}")


def _reference_handle():
    """The open FASTA, opened on first use."""
    if _REFERENCE["handle"] is None:
        if not _REFERENCE["path"]:
            raise ConfigError(
                "no reference genome. Pass --fasta <hg38.fa> (indexed).\n"
                "  Refusing to invent sequence: a synthetic fallback "
                "here would run the model on random DNA\n  and emit "
                "plausible, meaningless scores. gradient x input is "
                "defined against the REFERENCE\n  nucleotide, so "
                "without the real sequence the method is not the "
                "method.")
        import pyfaidx
        _REFERENCE["handle"] = pyfaidx.Fasta(_REFERENCE["path"], as_raw=True,
                                             sequence_always_upper=True)
    return _REFERENCE["handle"]


@lru_cache(maxsize=64)
def _reference_seq(chrom: str, start: int, length: int) -> str:
    """`length` bp from `start`, N-padded past either contig end.

    Windows are centred on an anchor and may overhang a telomere, so
    the clipping has to be explicit.
    """
    fa = _reference_handle()
    if chrom not in fa:
        raise ConfigError(f"{chrom!r} is not in {_REFERENCE['path']}")
    n = len(fa[chrom])
    lo, hi = max(0, start), min(n, start + length)
    seq = fa[chrom][lo:hi] if hi > lo else ""
    if start < 0:
        seq = "N" * (-start) + seq
    if start + length > n:
        seq = seq + "N" * (start + length - n)
    return seq[:length]


def one_hot(w: Window, gene: Gene, backend: Backend) -> np.ndarray:
    """Reference sequence of the window, one-hot (L, 4).

    `N`, and any other non-ACGT character, stays an all-zero row, which
    is what `gradient_map` assumes when it selects the reference
    channel.

    The stand-in backend gets deterministic pseudo-sequence, because
    its "model" is arithmetic on a random matrix and real DNA would
    tell it nothing. Every real backend needs a FASTA: a fallback there
    would run the model on invented sequence and emit plausible,
    meaningless numbers, the one failure mode that never announces
    itself.
    """
    if isinstance(backend, SyntheticBackend) or getattr(backend, "SYNTHETIC_SEQUENCE", False):
        digest = hashlib.sha256(
            f"{gene.chrom}:{w.start}:{w.length}".encode()).hexdigest()
        rng = np.random.RandomState(int(digest[:8], 16))
        idx = rng.randint(0, 4, size=w.length)
        x = np.zeros((w.length, 4), dtype=np.float64)
        x[np.arange(w.length), idx] = 1.0
        return x
    seq = _reference_seq(gene.chrom, w.start, w.length)
    x = np.zeros((w.length, 4), dtype=np.float64)
    # Byte-table lookup: identical to `_BASE_INDEX.get(c, -1)` per character (FASTA is ASCII) and ~200x
    # faster than a per-character generator over a 1 Mb window.
    idx = _BASE_LUT[np.frombuffer(seq.encode("latin-1"), dtype=np.uint8)]
    keep = idx >= 0
    x[np.nonzero(keep)[0], idx[keep]] = 1.0
    return x


def attribution_map(w: Window, gene: Gene, r: Any, a: Aug,
                    theta: Dict[str, Any], rho: Resolved,
                    backend: Backend) -> np.ndarray:
    """The positional attribution map for one augmentation.

    Gradient and attention only. ISM is not dispatched through here
    because it returns per-enhancer signed draws rather than a
    positional map, so there is nothing to un-augment; callers reach
    `ism_map` directly.
    """
    x = augment(one_hot(w, gene, backend), a)
    with augmented_frame(a):
        if theta["method"] == "gradient":
            M = gradient_map(x, gene, w, r, theta, rho, backend, a)
        elif theta["method"] == "attention":
            M = attention_map(x, w, r, theta, backend, gene, a)
        else:
            raise ConfigError("attribution_map does not handle "
                              f"method={theta['method']!r}")
    M = unaugment(M, a)
    return M


def gradient_map(x: np.ndarray, gene: Gene, w: Window, r: Any,
                 theta: Dict[str, Any], rho: Resolved,
                 backend: Backend, a: Optional[Aug] = None) -> np.ndarray:
    """The gradient of the gene scalar at the reference sequence.

    `bin_abs_order` decides where the absolute value falls relative to
    the readout-bin sum, which Avsec 2021's wording leaves open: "'CAGE
    at TSS' always means summing the absolute gradient values from
    three adjacent bins, as is also done in gene-focused model
    evaluation." Read by word order that is |.| per bin and then summed
    (`before`); read by the cross-reference it is the gradient of the
    3-bin readout, absolutised once (`after`), since in gene-focused
    evaluation the 3-bin sum is over predictions and there is no
    per-bin gradient to absolutise. The two differ only where a
    nucleotide pushes the three TSS bins in opposite directions, so the
    gap is expected to grow with distance as the signal weakens.

    `after` is the default and costs one backward pass; `before` costs
    one per readout bin.
    """
    # An all-zero row (N) argmaxes to A, so ambiguous bases read the
    # A channel rather than raising.
    ref = x.argmax(axis=1)
    idx = np.arange(x.shape[0])

    def at_ref(scalar: Callable[[np.ndarray], float]) -> np.ndarray:
        G = backend.scalar_gradient(x, r, scalar)
        if FREE_MEAN_SUBTRACT:
            # Keep both terms so that `mean_subtract` stays choosable
            # after the map exists: the variants are A (false) and
            # A - m (true), and these two columns span both exactly.
            # Positions stay on axis 0, since `reduce_pair` reads
            # M.shape[0] as the map length.
            return np.stack([G[idx, ref], G.mean(axis=1)], axis=1)     # (L, 2)
        if theta["mean_subtract"]:
            G = G - G.mean(axis=1, keepdims=True)
        return G[idx, ref]

    if MULTI_CHANNEL:
        # One backward pass per assay group, plus one forward for the
        # group readout values the scale-free channels divide by.
        # Reverse mode yields one linear functional of the Jacobian per
        # pass, so N assays genuinely cost N passes. What they buy is
        # the entire weighting axis, and every subset of it, for free.
        groups = MULTI_GROUPS.get(
            (rho.meta.name, theta["assay"], theta["cell_type"]), ())
        if not groups:
            raise ConfigError(
                "--multi-channels needs a MULTI_GROUPS entry for "
                f"({rho.meta.name}, {theta['assay']}, "
                f"{theta['cell_type']})")
        maps = [at_ref(gene_scalar_fn(gene, w, theta, rho, group_only=i, aug=a))
                for i in range(len(groups))]
        y = backend.forward(x, r)
        vals = [float(gene_scalar_fn(gene, w, theta, rho, group_only=i, aug=a)(y))
                for i in range(len(groups))]
        MULTI_SCALARS[gene.gid] = tuple(vals)
        # Guard each divisor rather than letting it produce inf: a
        # group whose readout is near zero has no scale to divide by,
        # and the inf would propagate into every scale-free column.
        scales = [v if abs(v) > LOG_EPS else 1.0 for v in vals]
        # (L, 2N): one column per group, then the scale-free copies.
        return np.stack(maps + [m / s for m, s in zip(maps, scales)],
                        axis=1)

    if theta["bin_abs_order"] == "before":
        bins = readout_bins(gene, w, theta, rho.meta)
        return sum(
            np.abs(at_ref(gene_scalar_fn(gene, w, theta, rho, only_bin=b, aug=a)))
            for b in bins)
    return at_ref(gene_scalar_fn(gene, w, theta, rho, aug=a))


def attention_map(x: np.ndarray, w: Window, r: Any, theta: Dict[str, Any],
                  backend: Backend, gene: Optional[Gene] = None,
                  a: Optional[Aug] = None) -> np.ndarray:
    """Attention from the TSS tokens, spread back over base pairs.

    The query is the same set of positions the gradient path reads, not
    a single token. Avsec 2021: "'CAGE at TSS' always means summing ...
    from three adjacent bins ... the bin overlapping the TSS and one
    flanking bin on each side". The "always" is load-bearing: the
    convention is the paper's definition of "at the TSS", so it governs
    the attention query as much as the gradient readout.

    Querying one token instead is the easy mistake, and a quiet one:
    the gradient path reaches `readout_bins` and so reads
    `tss_width / bin_size` = 3 bins, and a single-token attention query
    runs 0.047 high against Avsec 2021 Figure 2b at 3-12.5 kb where the
    gradient arm reproduces it to 0.004. The three tokens are also the
    ones the reference TensorFlow implementation averages, 767 to 769,
    which are `readout_bins` (447, 448, 449) shifted by the crop.

    `readout_bins` returns output-bin indices while attention lives on
    the uncropped token grid, so the crop offset is added back here
    rather than subtracted.
    """
    A = backend.attention(x, r, FROZEN_ATTENTION_AGGREGATION)
    meta = backend.meta
    if gene is not None:
        # The query width follows `tss_width` rather than a knob of its
        # own, so attention and gradient cannot ask about different
        # positions.
        bins = frame_for(gene, a, meta).map_bins(       # in x's frame
            readout_bins(gene, w, theta, meta), meta)
        q = [b + meta.crop_bins for b in bins]   # output bin -> token
        q = [i for i in q if 0 <= i < A.shape[0]]
    else:
        q = []
    if not q:                          # no gene, or all off-grid
        q = [min(A.shape[0] - 1, (w.length // 2) // meta.bin_size)]
    row = A[q].mean(axis=0)
    return np.repeat(row, meta.bin_size)[: w.length]


def perturb(x: np.ndarray, p: int, q: int, mode: str,
            rng: np.random.RandomState) -> np.ndarray:
    """A copy of `x` with the span [p, q) replaced, per `mode`."""
    xp = x.copy()
    seg = x[p:q]
    n = q - p
    if mode == "random-probs":
        # Uniform random values per channel, not a random base, so the
        # result is not a valid one-hot and, like a zero block, is
        # off-manifold: Enformer never saw such an input in training.
        #
        # It matches Fulco distance bin 2 best (0.582 against a 0.596
        # target, where `random-replace` gives 0.699), but that bin
        # holds 112 pairs with a bootstrap interval of [0.574, 0.805],
        # so every mode is consistent with it and the apparently
        # decisive gap is a third of the sampling interval. On Gasperini
        # it is the worst mode tested, worst |d| 0.127 against
        # `random-replace`'s 0.069.
        xp[p:q] = rng.rand(n, 4).astype(x.dtype)
    elif mode == "zero-replace":
        # The window is left all-zero, with no base set: the only mode
        # that removes sequence information without adding any.
        #
        # "Replaced by a random sequence" reads like an ablation but is
        # a substitution. A uniform-random block is not neutral, at 50%
        # GC with no CpG depletion, so the model scores it as a novel
        # sequence and |f(mod) - f(ref)| conflates "this element is
        # gone" with "an arbitrary new element sits here". Avsec's
        # companion BPNet paper adopts the same convention: all zeroes
        # for the reference input, chosen for the highest correlation
        # with in silico mutagenesis contribution scores.
        #
        # It stays an alternative rather than the default because Avsec
        # 2021 says a random sequence and not a blank, and the literal
        # reading reproduces 4/4 Fulco and 3/4 Gasperini bars. The Fulco
        # bin above does not separate the two: 0.699 there against 0.625
        # here, both inside the same interval.
        #
        # The mode is deterministic, so `rng` goes unused and
        # `ism_shuffles` is inert: every draw is the same blank. That
        # removes draw variance entirely, where the random modes swing
        # about 0.05 between seeds.
        xp[p:q] = 0.0
    elif mode == "random-replace":
        xp[p:q] = 0.0
        xp[np.arange(p, q), rng.randint(0, 4, size=n)] = 1.0
    elif mode == "genomic-replace":
        # A second reading of Avsec 2021's "replaced by a random
        # sequence": a random real genomic sequence rather than i.i.d.
        # uniform bases. The score becomes "how much more does this gene
        # depend on this element than on a typical piece of genome at
        # that position".
        #
        # Substituting arbitrary new regulatory content ought to make
        # this noisier per pair, but staying on-manifold dominates that
        # term: it is the joint-best predictor measured (Fulco bin 2:
        # 0.727, against 0.725 dinuc-shuffle, 0.699 random bases, 0.625
        # blank, 0.582 random values).
        #
        # The donor comes from the same input window, excluding the
        # perturbed span itself and the TSS neighbourhood so that the
        # promoter is never duplicated into the enhancer slot.
        L = x.shape[0]
        # The guard scales with the window: a fixed 5 kb would cover a
        # short synthetic window entirely, leaving no legal donor and
        # silently demoting this mode to a shuffle.
        guard = min(5000, L // 20)
        tss_lo, tss_hi = L // 2 - guard, L // 2 + guard
        for _ in range(64):
            r0 = int(rng.randint(0, L - n))
            if (not (r0 < q and p < r0 + n)
                    and not (r0 < tss_hi and tss_lo < r0 + n)):
                xp[p:q] = x[r0:r0 + n]
                break
        else:            # pathological window; fall back to a shuffle
            xp[p:q] = seg[rng.permutation(n)]
    elif mode == "mononuc-shuffle":
        xp[p:q] = seg[rng.permutation(n)]
    elif mode == "dinuc-shuffle":
        # Despite the name, this does not preserve dinucleotide
        # frequency. It is a block shuffle of adjacent 2-mers: only the
        # even-offset dinucleotides survive, and every one straddling a
        # block boundary is resampled. The name matches what Linder 2025
        # ships as `--dinuc_shuffle`; that implementation scatters where
        # this one gathers, which are inverse permutations and so
        # distributionally the same. Use `eulerian-dinuc-shuffle` for
        # the operation the rest of the field means by the phrase.
        pairs = [seg[i:i + 2] for i in range(0, n, 2)]
        order = rng.permutation(len(pairs))
        xp[p:q] = np.concatenate([pairs[i] for i in order], axis=0)[:n]
    elif mode == "eulerian-dinuc-shuffle":
        # The Altschul-Erickson Eulerian-path shuffle: the only mode
        # here that preserves dinucleotide frequency exactly, and what
        # deeplift, ChromBPNet, GRAMMAR and CREME all mean by
        # "dinucleotide shuffle". Over 300 real 2 kb enhancer windows it
        # reproduces the observed CpG obs/exp of 0.303 exactly, where
        # `dinuc-shuffle` above lands at 0.650, 2.14x high, and a
        # mononucleotide shuffle at 1.001. An ISM null is only
        # compositionally neutral, in the sense its logic assumes, under
        # this mode.
        xp[p:q] = dinuc_shuffle(seg, rng=rng).astype(x.dtype)
    else:
        raise ConfigError(f"unknown ism_mode {mode!r}")
    return xp


def ism_map(gene: Gene, enhancers: Sequence[Enhancer], w: Window,
            r: Any, a: Aug, theta: Dict[str, Any], rho: Resolved,
            backend: Backend) -> Tuple[Dict[str, np.ndarray], float]:
    """The signed ISM draws per enhancer, and the shared wild type. Every forward pass runs inside
    `augmented_frame(a)`."""
    with augmented_frame(a):
        return _ism_map_in_frame(gene, enhancers, w, r, a, theta, rho, backend)


def _ism_map_in_frame(gene: Gene, enhancers: Sequence[Enhancer], w: Window,
                      r: Any, a: Aug, theta: Dict[str, Any], rho: Resolved,
                      backend: Backend) -> Tuple[Dict[str, np.ndarray], float]:
    """The body of `ism_map`."""
    x = augment(one_hot(w, gene, backend), a)
    per_track = theta["track_abs_order"] == "before"                # A.3 1b
    # The lesion is placed in x's (augmented) frame below, so the readout must be too.
    scalar = gene_scalar_fn(gene, w, theta, rho, per_track=per_track, aug=a)
    u_wt = scalar(backend.forward(x, r))                            # A.3 2
    D: Dict[str, np.ndarray] = {}
    P = theta["ism_shuffles"]
    half = theta["ism_window"] // 2
    mode = FROZEN_ISM_WINDOW_MODE
    for e in enhancers:                                             # A.3 4
        m0 = e.mid() - w.start
        if not (0 <= m0 < w.length):
            continue
        # A.3 5-6  Place the lesion. The midpoint gates scoreability in
        # every mode, so the three modes score exactly the same pairs
        # and their rankings stay comparable; only the span differs.
        # `center` is Avsec 2021's reading.
        if mode == "center":
            m = augment_position(m0, a, w.length)                   # A.3 5
            p, q = m - half, m - half + theta["ism_window"]         # A.3 6
        elif mode == "start":
            s0 = e.start - w.start
            p, q = augment_interval(s0, s0 + theta["ism_window"],
                                    a, w.length)
        elif mode == "element":
            p, q = augment_interval(e.start - w.start, e.end - w.start,
                                    a, w.length)
        else:
            raise ConfigError(f"unknown ism_window_mode {mode!r}")
        want = q - p                              # intended span, pre-clip
        # Clip, do not discard. The enhancer midpoint is inside the
        # window, checked above, so the pair is scoreable and only part
        # of the span falls outside. Discarding instead sends those
        # pairs to `out_of_window` = 0, so Fulco pairs at
        # d = 96,980-98,302 bp, just inside Enformer's 98,304 bp
        # half-window, score exactly 0.0 while gradient and attention
        # score them normally. Clipping is also the likelier reading of
        # Avsec 2021: the natural implementation replaces
        # `seq[mid-1000 : mid+1000]`, and Python slicing truncates at
        # the array edge rather than raising. `max(0, p)` matters
        # independently: a negative p would wrap the slice to the far
        # end of the sequence.
        p, q = max(0, p), min(w.length, q)                          # A.3 7
        # Guard against the intended span, not against `ism_window`:
        # under `ism_window_mode: element` the intended span is the
        # element's own extent, which is legitimately shorter than half
        # a window, and guarding on `ism_window` would silently drop
        # every short element. For `center`, where want == ism_window,
        # this is the identical test, and a backstop: the midpoint is
        # already known to be inside the window, so at least half the
        # span survives the clip.
        if q - p < max(1, want // 2):
            continue
        # A.3 7b  The RNG is keyed on the pair, so each pair sees
        # independent noise and the draw variance averages out across
        # the ranking. Dropping the pair from the key would compare
        # every pair against the same block instead, making the noise
        # common-mode.
        rng = np.random.RandomState(
            stable_seed(theta["seed"], gene.gid, e.eid, r, a.shift, a.rc))
        # Shape () for the scalar readout, (T,) when
        # track_abs_order=before keeps the tracks apart.
        d = np.empty((P,) + np.shape(u_wt), dtype=np.float64)
        for j in range(P):                                          # A.3 8-10
            xp = perturb(x, p, q, theta["ism_mode"], rng)
            d[j] = u_wt - scalar(backend.forward(xp, r))
        D[e.eid] = d
    return D, u_wt                                                  # A.3 11


# ---- the frame guard, layers 3 and 4 ---------------------------------------

class _ProbeOutput:
    """A (n_bins, n_tracks) prediction held as columns.

    The gene scalar only ever reads `y[:, tracks]`, and a dense Borzoi
    (16,352 x 7,612) or AlphaGenome (1,048,576 x n) array would cost a
    gigabyte to say so.
    """

    def __init__(self, cols: Dict[int, np.ndarray], n_bins: int):
        self.cols = cols
        self.shape = (n_bins, (max(cols) + 1) if cols else 0)

    def __getitem__(self, key):
        rows, cols = key
        if isinstance(cols, (list, tuple, np.ndarray)):
            return np.stack([self.cols[int(c)][rows] for c in cols], axis=1)
        return self.cols[int(cols)][rows]

    def with_entry(self, b: int, t: int, v: float) -> "_ProbeOutput":
        cols = dict(self.cols)
        c = cols[t].copy()
        c[b] = v
        cols[t] = c
        return _ProbeOutput(cols, self.shape[0])


class FrameProbeBackend(Backend):
    """An exactly equivariant stand-in on a model's real geometry (layer 3).

    Track t of output bin b is the sum, over the input bases the bin
    covers, of a per-base weight:

        y[b, t] = sum_{i in bin b} H[strand(t)] . x_i

    with H['+'] = (1, 2, 3, 5) over (A, C, G, T), H['-'] = H['+'] composed
    with the complement = (5, 3, 2, 1), and an unstranded track
    complement-symmetric, (1, 5, 5, 1). H['+'] + H['-'] = (6, 5, 5, 6)
    depends on the base, so a readout that averages a '+' and a '-' track
    (strand_mask off) still sees the sequence. So, exactly and on integers:

        reverse complement:    y_rc[n_bins - 1 - b, t] = y[b, partner(t)]
        whole-bin shift by k:  y_k[b + k, t]           = y[b, t]

    ('+' <-> '-', unstranded <-> unstranded; the shift holds away from
    the zero-filled edge), because every model's crop is symmetric and a
    whole number of bins. A correct readout frame therefore gives an
    augmented pass the SAME gene scalar, the SAME un-augmented gradient
    map, the SAME ISM wild type and the SAME attention row as the forward
    pass. A wrong frame does not: a mirrored bin reads other bases, an
    un-swapped strand reads H['-'] where H['+'] belongs, and a one-bin
    offset reads a neighbouring bin.

    Only the weights are invented. The geometry (seq_len, bin_size,
    n_bins, crop, track indices and the strand of every resolved track)
    is the model's own, so this tests the frame arithmetic each real
    model runs. dU/dy is formed in closed form for the linear-then-log
    readouts every config uses and gated against the gene scalar itself
    by central differences; anything else is differenced entry by entry.
    """
    SYNTHETIC_SEQUENCE = True    # one_hot() gives it pseudo-sequence, no FASTA
    H = {"+": np.array([1.0, 2.0, 3.0, 5.0]),
         "-": np.array([5.0, 3.0, 2.0, 1.0]),
         None: np.array([1.0, 5.0, 5.0, 1.0])}
    MAX_ATTENTION_TOKENS = 4096

    def __init__(self, meta: ModelMeta, rho: Resolved):
        self.meta = meta
        self.strand_of = {int(t): st
                          for t, st in zip(rho.tracks, rho.track_strands)}

    def _profile(self, x: np.ndarray, h: np.ndarray) -> np.ndarray:
        bs, nb = self.meta.bin_size, self.meta.n_bins
        off = self.meta.crop_bins * bs
        return (x @ h)[off: off + nb * bs].reshape(nb, bs).sum(axis=1)

    def forward(self, x: np.ndarray, replicate: Any) -> Any:
        prof: Dict[Optional[str], np.ndarray] = {}
        cols: Dict[int, np.ndarray] = {}
        for t, st in self.strand_of.items():
            if st not in prof:
                prof[st] = self._profile(x, self.H[st])
            cols[t] = prof[st]
        return _ProbeOutput(cols, self.meta.n_bins)

    @staticmethod
    def _fd(scalar_fn, y, b: int, t: int) -> float:
        v = float(y.cols[t][b])
        h = 1e-3 * max(1.0, abs(v))
        return (float(scalar_fn(y.with_entry(b, t, v + h)))
                - float(scalar_fn(y.with_entry(b, t, v - h)))) / (2.0 * h)

    def _dU_dy(self, y, scalar_fn) -> np.ndarray:
        bins, tracks = list(scalar_fn.bins), list(scalar_fn.tracks)
        ops = getattr(scalar_fn, "ops", None) or {}
        if (ops and not ops.get("multi") and not ops.get("per_track")
                and ops.get("track_agg") in ("sum", "mean")):
            Y = np.stack([y.cols[t][bins] for t in tracks],
                         axis=1).astype(np.float64)
            if (ops["inverse_transform"]
                    and ops["transform_generation"] != "none"):
                sc, ex, cs = scalar_fn.xform
                rr = SimpleNamespace(scale=sc, exponent=ex, clip_soft=cs,
                                     meta=self.meta)
                hY = 1e-4 * np.maximum(1.0, np.abs(Y))
                dY = (unsquash(Y + hY, rr) - unsquash(Y - hY, rr)) / (2.0 * hY)
            else:
                dY = np.ones_like(Y)
            w_t = 1.0 if ops["track_agg"] == "sum" else 1.0 / len(tracks)
            d_total = 1.0
            if ops["log_scalar"]:
                arg = float(scalar_fn.linear(y)) + ops["pseudocount"] + ops["eps"]
                d_total = 1.0 / (arg * (math.log(2.0)
                                        if ops["log_base"] == "2" else 1.0))
            g = d_total * w_t * dY
            rng = np.random.RandomState(0)
            ok = True
            for _ in range(min(4, g.size)):      # gate against the closure
                i, j = rng.randint(len(bins)), rng.randint(len(tracks))
                fd = self._fd(scalar_fn, y, bins[i], tracks[j])
                if abs(fd - g[i, j]) > 1e-5 * max(1e-300, abs(fd), abs(g[i, j])):
                    ok = False
                    break
            if ok:
                return g
        if len(bins) * len(tracks) > 20_000:
            raise FrameError(
                "frame probe: the gene scalar's reduction is not the "
                f"linear-then-log form the probe differentiates in closed "
                f"form (ops={ops}), and {len(bins)} x {len(tracks)} readout "
                "entries are too many to difference one by one")
        g = np.zeros((len(bins), len(tracks)))
        for i, b in enumerate(bins):
            for j, t in enumerate(tracks):
                g[i, j] = self._fd(scalar_fn, y, b, t)
        return g

    def scalar_gradient(self, x: np.ndarray, replicate: Any,
                        scalar_fn: Callable[[np.ndarray], float]) -> np.ndarray:
        y = self.forward(x, replicate)
        g = self._dU_dy(y, scalar_fn)                               # (B, T)
        Ht = np.stack([self.H[self.strand_of[int(t)]]
                       for t in scalar_fn.tracks])
        wb = g @ Ht                        # (B, 4): dU/dx per base of bin b
        bs = self.meta.bin_size
        off = self.meta.crop_bins * bs
        G = np.zeros(x.shape, dtype=np.float64)
        for i, b in enumerate(scalar_fn.bins):
            G[off + b * bs: off + (b + 1) * bs, :] = wb[i]
        lin = getattr(scalar_fn, "linear", None)
        self._last_T = float(lin(y)) if lin is not None else float("nan")
        self._last_u = float(scalar_fn(y))
        return G

    def attention(self, x: np.ndarray, replicate: Any, how: str) -> np.ndarray:
        """Token q attends to token k with weight s(q) exp(-|q - k| / 64).

        s is complement-symmetric content, so the map is mirror
        equivariant: A_rc[n-1-q, n-1-k] = A[q, k]. On the input token
        grid, as `attention_map` expects.
        """
        bs = self.meta.bin_size
        n_tok = self.meta.seq_len // bs
        sv = (x @ self.H[None]).reshape(n_tok, bs).sum(axis=1)
        k = np.arange(n_tok)
        return sv[:, None] * np.exp(-np.abs(k[:, None] - k[None, :]) / 64.0)


#: The probe's reports for this process, keyed by the geometry and
#: readout each one certified.
_FRAME_SELFTEST_DONE: Dict[Any, Dict[str, Any]] = {}
#: The report of the run in progress; the manifest records it under
#: `runtime.frame_guard`.
FRAME_REPORT: Dict[str, Any] = {}


def _probe_genes(meta: ModelMeta, bs: int,
                 attempt: int = 0) -> Tuple[Window, List[Gene]]:
    """Two genes, one per strand, in one window.

    Their readouts are NOT symmetric about the window centre, so a
    mirrored or offset read lands on other bases and the probe cannot be
    blind. `attempt` moves the window by whole bins: the same geometry on
    another pseudo-sequence (`frame_selftest` searches for one with teeth).
    """
    c = 50_000_000 + 7 * bs + 3 + attempt * 1000 * bs
    w = window_at(c, meta.seq_len)
    span = max(bs, 32)
    exons = ((c - 90 * span, c - 77 * span), (c + 29 * span, c + 37 * span),
             (c + 300 * span + 5, c + 380 * span))
    lo = w.start + meta.crop_bins * bs + 4 * bs
    hi = w.start + w.length - meta.crop_bins * bs - 4 * bs
    exons = tuple((max(lo, a), min(hi, b)) for a, b in exons
                  if min(hi, b) > max(lo, a))
    out = []
    for strand in "+-":
        tss = c + (5 * bs + 17 if strand == "+" else -(9 * bs) - 11)
        extra = tuple(t for t in (c + 61 * span + 1, c - 140 * span)
                      if lo < t < hi)
        out.append(Gene(gid=f"FRAMEPROBE{strand}", chrom="chrFrameProbe",
                        strand=strand, provided_tss=tss,
                        gencode_tss=(tss,) + extra, exons=exons,
                        gencode=None))
    return w, out


def frame_selftest(meta: ModelMeta, theta: Dict[str, Any]) -> Dict[str, Any]:
    """Layer 3: certify this geometry and readout before any model call.

    Drives the production gene scalar, the gradient map (through
    `attribution_map`, so `augment` and `unaugment` too), the ISM lesion
    mapping and, where cheap, the attention map through
    `FrameProbeBackend`, for a '+' and a '-' gene, under reverse
    complement, whole-bin shifts both ways, their compositions and every
    whole-bin augmentation the run itself uses. Each augmented pass must
    reproduce the forward pass. The probe must also be able to FAIL:
    reading the forward frame on the rc pass has to change the readout,
    or the probe is blind for this geometry and says so. Raises
    FrameError; returns the report the manifest records.
    """
    key = (meta.name, meta.seq_len, meta.bin_size, meta.n_bins,
           FREE_MEAN_SUBTRACT, MULTI_CHANNEL,
           tuple(sorted((k, repr(v)) for k, v in theta.items()
                        if not str(k).startswith("_"))))
    if key in _FRAME_SELFTEST_DONE:
        return _FRAME_SELFTEST_DONE[key]
    th = dict(theta, replicates=1, tss_major=False)
    rho = resolve(th, meta)
    probe = FrameProbeBackend(meta, rho)
    bs = meta.bin_size
    augs = [Aug(0, True), Aug(bs, False), Aug(-2 * bs, False), Aug(bs, True),
            Aug(-3 * bs, True)]
    run_augs: List[Aug] = []
    if theta.get("tta") not in (None, "none"):
        k = theta.get("tta_shift_bp") or 0
        rcs = ((False, True) if ("rc" in theta["tta"]
                               or theta["tta"] == "avsec8") else (False,))
        run_augs = [Aug(sh, rc) for sh in range(-k, k + 1) for rc in rcs]
    for a in run_augs:
        if _aug_key(a) != (0, False) and a.shift % bs == 0 and a not in augs:
            augs.append(a)                     # exact for whole-bin shifts
    w, genes = _probe_genes(meta, bs)

    def close(u, v):
        u = np.asarray(u, dtype=np.float64)
        v = np.asarray(v, dtype=np.float64)
        return (u.shape == v.shape
                and bool(np.all(np.abs(u - v) <= 1e-9 * (1.0 + np.abs(v)))))

    def rc_teeth(g, w):
        """Does reading the forward frame on the rc pass change g's readout?"""
        try:
            f = gene_scalar_fn(g, w, th, rho, aug=None)
        except ConfigError:
            return True          # the loop below raises the clearer error
        x = one_hot(w, g, probe)
        return not close(f(probe.forward(augment(x, Aug(0, True)), 1)),
                         f(probe.forward(x, 1)))

    # A readout that averages both strands at the window centre (strand_mask
    # off) sees only part of each base under the probe's weights, and its rc
    # mirror differs from it by a bin or two at each edge: on some
    # pseudo-sequences those edges agree by chance and the probe has no
    # teeth (1 bp bins). Then move the window by whole bins; window 0 is the
    # default.
    attempt = 0
    while attempt < 8 and not any(rc_teeth(g, w) for g in genes):
        attempt += 1
        w, genes = _probe_genes(meta, bs, attempt)

    n_checks, teeth = 0, []
    L = meta.seq_len
    for g in genes:
        try:
            f0 = gene_scalar_fn(g, w, th, rho, aug=None)
        except FrameError:
            raise
        except ConfigError as e:
            raise FrameError(
                f"frame probe: no readout for the probe gene ({e}); the "
                "probe geometry needs updating for this readout") from e
        x0 = one_hot(w, g, probe)
        u0 = f0(probe.forward(x0, 1))
        # Teeth: the forward frame read off the rc pass must differ, or
        # this probe could not see a forward-frame readout at all.
        teeth.append(not close(f0(probe.forward(augment(x0, Aug(0, True)), 1)),
                               u0))
        for a in augs:
            ua = gene_scalar_fn(g, w, th, rho, aug=a)(
                probe.forward(augment(x0, a), 1))
            if not close(ua, u0):
                raise FrameError(
                    f"frame probe ({meta.name}, gene strand {g.strand}, {a}): "
                    f"the augmented pass reads {np.asarray(ua).ravel()[:3]} "
                    f"where the forward pass reads {np.asarray(u0).ravel()[:3]}"
                    " -- the readout is not in the augmented frame")
            n_checks += 1
        if th["method"] == "gradient":
            m0 = attribution_map(w, g, 1, Aug(0, False), th, rho, probe)
            scale = float(np.max(np.abs(m0)))
            if not scale > 0.0:
                raise FrameError(f"frame probe ({meta.name}): the forward "
                                 "gradient map is all zero")
            for a in augs:
                ma = attribution_map(w, g, 1, a, th, rho, probe)
                if (ma.shape != m0.shape
                        or float(np.max(np.abs(ma - m0))) > 1e-9 * scale):
                    raise FrameError(
                        f"frame probe ({meta.name}, gene strand {g.strand}, "
                        f"{a}): the un-augmented gradient map differs from "
                        "the forward map by "
                        f"{float(np.max(np.abs(ma - m0))) / scale:.3e} of its "
                        "scale")
                n_checks += 1
        if (th["method"] == "attention"
                and L // bs <= FrameProbeBackend.MAX_ATTENTION_TOKENS):
            m0 = attribution_map(w, g, 1, Aug(0, False), th, rho, probe)
            for a in [a for a in augs if a.rc and a.shift == 0]:
                ma = attribution_map(w, g, 1, a, th, rho, probe)
                if (float(np.max(np.abs(ma - m0)))
                        > 1e-9 * float(np.max(np.abs(m0)))):
                    raise FrameError(f"frame probe ({meta.name}, {a}): the "
                                     "attention query is not mirrored")
                n_checks += 1
        # The ISM lesion: the interval placed in the augmented frame must
        # hold the forward bases.
        for a in augs + run_augs:
            xa = augment(x0, a)
            for i, j in ((L // 2 - 1000, L // 2 + 1000),
                         (L // 3 + 7, L // 3 + 2055),
                         (5 * L // 8 + 1, 5 * L // 8 + 38)):
                lo, hi = augment_interval(i, j, a, L)
                want = x0[i:j][::-1, ::-1] if a.rc else x0[i:j]
                if hi - lo != j - i or not np.array_equal(xa[lo:hi], want):
                    raise FrameError(
                        f"frame probe ({meta.name}, {a}): the ISM lesion "
                        f"interval [{i}, {j}) maps to [{lo}, {hi}), which "
                        "does not hold the forward bases")
                n_checks += 1
    if not any(teeth):
        raise FrameError(
            f"frame probe ({meta.name}): reading the forward frame on the rc "
            "pass gives the same readout for both probe genes in "
            f"{attempt + 1} probe windows, so the probe could not detect a "
            "forward-frame readout for this geometry; make _probe_genes "
            "asymmetric for it")
    report = {"status": "passed", "model": meta.name,
              "geometry": {"seq_len": meta.seq_len, "bin_size": bs,
                           "n_bins": meta.n_bins, "crop_bins": meta.crop_bins},
              "readout": {"assay": th.get("assay"),
                          "cell_type": th.get("cell_type"),
                          "region": th.get("region"),
                          "strands": sorted({str(st)
                                             for st in rho.track_strands})},
              "method": th["method"], "checks": n_checks,
              "augmentations": [f"shift={a.shift},rc={a.rc}" for a in augs],
              "teeth": "reading the forward frame on the rc pass changes "
                       "the probe readout"}
    if attempt:
        report["probe_window"] = attempt   # window 0 had no teeth here
    _FRAME_SELFTEST_DONE[key] = report
    return report


class FrameAgreementGate:
    """Layer 4: the real model's gene readout must agree with itself
    across reverse complement.

    Enformer, Borzoi and AlphaGenome were all trained with rc
    augmentation, so with the readout in the right frame a gene's
    readout T on an rc pass is close to T on the forward pass at the
    same shift. Measured on real genes: Enformer CAGE@TSS has a median
    |log(T_rc / T_fwd)| of about 0.05, while a forward-frame readout on
    Borzoi RNA-seq at exons gives 2.8. The gate reads T from the passes
    the run makes anyway (`backend._last_T`, or the ISM wild type), so it
    costs no model call. After MIN_GENES genes, and again at the end, a
    median above the threshold stops the run.

    It is the backstop for what the exact probe cannot see: a backend
    that augments internally, a wrong strand table for real tracks, a
    crop the metadata misstates. It is NOT sensitive enough for a
    one-bin offset of an unstranded 3-bin readout (|log ratio| of about
    0.03-0.05 either way); layer 3 catches that one exactly.
    `SCORE_EG_FRAME_GATE_MAX` raises the threshold and
    `SCORE_EG_FRAME_GATE=off` disables the gate; the manifest records
    both.
    """
    MIN_GENES = 16

    def __init__(self, theta: Dict[str, Any]):
        self.enabled = (theta.get("tta") not in (None, "none")
                        and ("rc" in theta["tta"] or theta["tta"] == "avsec8")
                        and os.environ.get("SCORE_EG_FRAME_GATE",
                                           "on").lower() != "off")
        self.threshold = float(os.environ.get("SCORE_EG_FRAME_GATE_MAX", "0.5"))
        self.log_base: Optional[str] = None
        self.theta = theta
        self.d: List[float] = []
        self.checked = False

    def _linear(self, v: Any) -> float:
        v = float(np.mean(np.asarray(v, dtype=np.float64)))
        if self.theta["method"] == "ism" and self.theta.get("log_scalar"):
            return float(2.0 ** v if self.log_base == "2"
                         else math.exp(min(v, 700.0)))
        return v

    def observe(self, gid: str, augs: Sequence[Aug],
                T: Dict[Tuple[int, int, int], Any], meta: ModelMeta) -> None:
        """Record one gene's rc-vs-forward readout disagreement."""
        if not self.enabled:
            return
        self.log_base = meta.log_base
        diffs = []
        for (iw, ir, ia), v in T.items():
            a = augs[ia]
            if not a.rc:
                continue
            for ib, b in enumerate(augs):
                if not b.rc and b.shift == a.shift and (iw, ir, ib) in T:
                    t_rc = self._linear(v)
                    t_f = self._linear(T[(iw, ir, ib)])
                    if not (math.isfinite(t_rc) and math.isfinite(t_f)):
                        continue
                    eps = 1e-3 * (abs(t_rc) + abs(t_f)) / 2.0 + 1e-12
                    diffs.append(abs(math.log(abs(t_rc) + eps)
                                     - math.log(abs(t_f) + eps)))
                    break
        if diffs:
            self.d.append(float(np.median(diffs)))
            if len(self.d) == self.MIN_GENES:
                self.check()

    def check(self) -> None:
        self.checked = True
        if self.enabled and self.d and float(np.median(self.d)) > self.threshold:
            raise FrameError(
                f"FRAME AGREEMENT GATE: over {len(self.d)} genes the model's "
                "rc readout departs from its forward readout by a median "
                f"|log(T_rc / T_fwd)| of {float(np.median(self.d)):.3f} "
                f"(threshold {self.threshold}). A correct frame gives about "
                "0.05; a forward-frame readout gave 2.8 on Borzoi RNA-seq at "
                "exons. Stopping before more model time is spent on a wrong "
                "readout.")

    def report(self) -> Dict[str, Any]:
        if not self.enabled:
            why = ("no rc augmentation"
                   if self.theta.get("tta") in (None, "none")
                   or not ("rc" in self.theta["tta"]
                           or self.theta["tta"] == "avsec8")
                   else "SCORE_EG_FRAME_GATE=off")
            return {"status": "inactive", "why": why}
        if not self.d:
            return {"status": "no-pairs",
                    "why": "no gene had an rc pass with a same-shift forward "
                           "pass"}
        d = np.asarray(self.d)
        return {"status": ("passed" if float(np.median(d)) <= self.threshold
                           else "failed"),
                "n_genes": int(d.size),
                "median_abs_log_ratio": round(float(np.median(d)), 5),
                "p90_abs_log_ratio": round(float(np.quantile(d, 0.9)), 5),
                "max_abs_log_ratio": round(float(d.max()), 5),
                "threshold": self.threshold}


def check_frame_contract(theta: Dict[str, Any], backend: Backend) -> None:
    """Refuse, before any model call, augmentations whose frame cannot be
    made exact.

    That is a shift that is neither under half a bin nor a whole number
    of bins, and a per-track ISM statistic that would pair a '+' track
    with an arbitrary '-' track across orientations.
    """
    if theta.get("tta") in (None, "none"):
        return
    meta = backend.meta
    k = theta.get("tta_shift_bp") or 0
    for sh in range(-k, k + 1):
        shift_bins(sh, meta)
    has_rc = "rc" in theta["tta"] or theta["tta"] == "avsec8"
    if not has_rc:
        return
    rho = resolve(theta, meta)
    stranded = any(st in ("+", "-") for st in rho.track_strands)
    if (theta["method"] == "ism" and theta.get("track_abs_order") == "before"
            and stranded):
        raise FrameError(
            "track_abs_order=before keeps one ISM value per track, and under "
            "rc the '+' tracks are read on the '-' pass in list order, which "
            "pairs each with whichever '-' track sits at the same position, "
            "not necessarily its own library. No strand-pair table is "
            "declared for this track set, so the per-track pairing cannot be "
            "made exact; refusing.")


# ---- A.4 map -> pair score ------------------------------------------------

def map_grid(theta: Dict[str, Any], meta: ModelMeta) -> int:
    """The bp resolution of the attribution map.

    Gradient and ISM maps are per-nucleotide; attention lives on the
    transformer's token grid, 128 bp for Enformer.
    """
    return meta.bin_size if theta["method"] == "attention" else 1


def snap_to_grid(lo: int, hi: int, grid: int, rule: str) -> Tuple[int, int]:
    """Express the bp interval [lo, hi) on a `grid`-bp grid, per `rule`.

    Only bites when the map is coarser than base pairs, that is
    attention, whose tokens are 128 bp for Enformer: a nominal 2 kb
    window is 15.625 tokens and cannot be taken exactly. Returns bp
    bounds again, snapped to token edges, or unchanged for `fractional`,
    which weights the boundary tokens by their overlap instead.
    """
    if grid <= 1 or rule == "fractional":
        return lo, hi
    if rule == "fixed":
        # A constant, phase-independent token count: round(width/grid)
        # tokens centred on the interval's midpoint. The other rules
        # snap the two edges independently, so their count depends on
        # where the midpoint falls against the grid, which is not a
        # biological quantity. Over all 128 phases of a 2 kb window on a
        # 128 bp grid, `any` gives 16 or 17 tokens, `majority` 15 or 16,
        # and this rule exactly 16 every time.
        mid = (lo + hi) / 2.0
        n = max(1, int(round((hi - lo) / grid)))
        start = int(math.floor(mid / grid)) - (n - 1) // 2
        return start * grid, (start + n) * grid
    if rule == "majority":                    # >50% of the token inside
        return int(round(lo / grid)) * grid, int(round(hi / grid)) * grid
    return (int(math.floor(lo / grid)) * grid,            # any overlap
            int(math.ceil(hi / grid)) * grid)


#: The divisor ladder: every candidate `nu` for the map, stored so the
#: whole family is post-hoc. `global` is the mean, and is both the
#: default and the only published divisor (Linder's mean of |grad|).
#:
#: Why a ladder and not just `max`: any per-gene divisor is macro-inert,
#: and whether it moves the micro ranking depends on what it divides by.
#: These candidates run monotonically from a pure background statistic
#: to a pure signal one, which is the instrument that question needs:
#:
#:   median, trim99   background, robust: noise floor without the peak
#:   mean (= global)  background, pulled upward by the peak
#:   rms, std         background weighted toward large values
#:   p90 p95 p99 p999 the transition from background into signal
#:   max              pure signal, normally the promoter
#:
#: `l1` (sum |M|) is deliberately absent: the window length is the same
#: for every gene, so sum = L x mean is a global constant times the mean
#: and therefore rank-identical to it.
DIVISOR_KINDS: Tuple[str, ...] = ("global", "median", "trim99", "rms", "std",
                                  "p90", "p95", "p99", "p999", "max")


def map_divisors(M: np.ndarray) -> Dict[str, float]:
    """Every candidate `nu` for a 1-D map, in one O(L) family of passes.

    Called once per (pair, grid cell), so the quantiles go through a
    single `np.partition` carrying all five kth values rather than
    `np.percentile`, which sorts. O(L) against O(L log L) is worth the
    awkwardness at 196,608 positions for Enformer, and more so at
    Borzoi's 524,288 x 4 replicates.

    `nu` is always formed on the absolute map, even when `abs_map` is
    false, so every entry describes |M| rather than M. Read `std`
    accordingly: it is the spread of the absolute map, not of the
    signed one.
    """
    a = np.abs(M)
    n = a.size
    if n == 0:
        return {k: 1.0 for k in DIVISOR_KINDS}
    ks = [min(n - 1, max(0, int(round(p * (n - 1)))))
          for p in (0.50, 0.90, 0.95, 0.99, 0.999)]
    part = np.partition(a, ks)
    med, p90, p95, p99, p999 = (float(part[k]) for k in ks)
    # Trimmed mean below p99: the background with the signal peak
    # removed. The mask is on `a`, not on the partitioned copy, because
    # `np.partition` reorders and positional indexing into it is valid
    # only at the kth entries themselves.
    keep = a <= p99
    return {"global": float(a.mean()),
            "median": med,
            "trim99": float(a[keep].mean()) if keep.any() else float(a.mean()),
            "rms": float(np.sqrt(float((a * a).mean()))),
            "std": float(a.std()),
            "p90": p90, "p95": p95, "p99": p99, "p999": p999,
            "max": float(a.max())}


def _collapse_map(M: np.ndarray, theta: Dict[str, Any]) -> np.ndarray:
    """The channel collapse at the top of `reduce_pair`: a 1-D signed map.

    Factored out so `PostMapCache` can key on it; the arithmetic is the
    one `reduce_pair` performs, so a cached collapse is bit-identical to
    an uncached one.
    """
    # Two-channel map: form the mean_subtract variant first, because
    # |A - m| != |A| - |m| and the absolute value applies to whatever
    # this leaves. Positions stay on axis 0, so the length test that
    # follows in `reduce_pair` is unaffected. A 1-D map is untouched, and
    # the branch is on ndim rather than on a flag so the two
    # representations cannot be confused.
    if MULTI_CHANNEL and M.ndim == 2:
        # Design B, an (L, 2N) map. Mixing happens before the absolute
        # value because |sum_i w_i C_i| != sum_i w_i |C_i|: a mixture
        # of finished per-channel scores would discard exactly the
        # sign agreement that the mixture exists to exploit.
        if len(CO_W) != M.shape[1]:
            raise ConfigError(
                f"CO_W has {len(CO_W)} weights for a "
                f"{M.shape[1]}-channel map: the mix vector and the "
                "channel count must agree, or the weights silently "
                "address the wrong assays")
        return sum(CO_W[i] * M[:, i] for i in range(M.shape[1]))
    if M.ndim == 2 and M.shape[1] == 2:
        return M[:, 0] - M[:, 1] if theta["mean_subtract"] else M[:, 0]
    return M


def _collapse_map_key(M: np.ndarray, theta: Dict[str, Any]) -> Tuple[Any, ...]:
    """Which collapse `_collapse_map` applies to `M`, without forming it."""
    if MULTI_CHANNEL and M.ndim == 2:
        return ("multi", tuple(CO_W))
    if M.ndim == 2 and M.shape[1] == 2:
        return ("ms", bool(theta["mean_subtract"]))
    return ("raw",)


class PostMapCache:
    """Per-gene memo of everything `reduce_pair` derives from a map that
    does not depend on the enhancer.

    That is the merged maps (`merge_maps`), the channel collapse, the
    absolute map and the divisor ladder (`map_divisors`). Without it the
    free grid recomputes each of them once per (pair, cell, map): on a
    Borzoi arm that is about 130 cells by 5 maps per pair, each call a
    full-length |M|, a full-length collapse and an O(L) partition over
    524,288 positions, none of which depend on the enhancer. With it the
    post-map stage costs O(L) per (gene, variant) plus O(window) per
    (pair, cell). Measured on 10 genes with the full free grid: Borzoi
    412 s to 64 s and Enformer 30 s to 15 s, model load included.

    It changes no number. The same arrays go through the same operations
    in the same order, so a cached call returns the bit-identical value
    an uncached call does; `_test_post_map_cache` and
    `_test_definition_matches_driver` (the uncached `score_pair` against
    the cached driver) both check it. One cache lives per gene group in
    `run_experiment` and is dropped with the group, so its memory cost is
    one gene's maps, not the run's.

    Entries hold a reference to their source array, so an `id()` can
    never be recycled underneath a key, which is the hazard that makes a
    bare id-keyed cache silently wrong.
    """

    def __init__(self) -> None:
        self._merged: Dict[Tuple[Any, ...], Any] = {}
        self._prep: Dict[Tuple[Any, ...], Tuple[Any, ...]] = {}

    def merged(self, store: Dict[Tuple[int, int, int], Any], n_w: int,
               n_r: int, n_a: int, theta: Dict[str, Any]):
        """`merge_maps`, once per distinct merge per gene.

        Only `replicate_abs_order` and `tta_abs_order` reach the merge,
        so a grid that varies neither merges once however many cells it
        holds.
        """
        k = (id(store), theta["replicate_abs_order"], theta["tta_abs_order"],
             n_w, n_r, n_a)
        hit = self._merged.get(k)
        if hit is None or hit[0] is not store:
            hit = (store, merge_maps(store, n_w, n_r, n_a, theta))
            self._merged[k] = hit
        return hit[1]

    def prepared(self, M: np.ndarray, theta: Dict[str, Any]):
        """(M, the collapsed 1-D map, its absolute value, divisors or None).

        The key is formed without touching the array, so a hit costs a
        dict lookup and nothing O(L).
        """
        k = (id(M),) + _collapse_map_key(M, theta)
        ent = self._prep.get(k)
        if ent is None or ent[0] is not M:
            C = _collapse_map(M, theta)
            ent = (M, C, np.abs(C), None)
            self._prep[k] = ent
        return ent

    def divisors(self, M: np.ndarray, theta: Dict[str, Any]) -> Dict[str, float]:
        """`map_divisors` of the collapsed map, computed once."""
        ent = self.prepared(M, theta)
        if ent[3] is None:
            ent = ent[:3] + (map_divisors(ent[1]),)
            self._prep[(id(M),) + _collapse_map_key(M, theta)] = ent
        return ent[3]


_GAUSSIAN_WEIGHTS: Dict[Tuple[float, int], np.ndarray] = {}


def _gaussian_weights(sigma: float, h: int) -> np.ndarray:
    """exp(-d^2 / (2 sigma^2)) for offsets d = -h .. h, memoised per (sigma, h).

    The weights depend only on the offset from the enhancer midpoint, so
    `reduce_pair` slices this vector instead of evaluating the
    exponential once per (enhancer, cell). Same expression on the same
    float64 operands, so the slice equals the vector a per-call
    evaluation builds, bit for bit.
    """
    key = (float(sigma), int(h))
    om = _GAUSSIAN_WEIGHTS.get(key)
    if om is None:
        d = np.arange(-int(h), int(h) + 1)
        om = np.exp(-(d ** 2) / (2.0 * sigma * sigma))
        _GAUSSIAN_WEIGHTS[key] = om
    return om


def reduce_pair(M: np.ndarray, enh: Enhancer, w: Window, theta: Dict[str, Any],
                grid: int = 1, cache: Optional[PostMapCache] = None):
    """Reduce one attribution map to a single pair score, per A.4.

    `grid` is the map's native resolution in bp: 1 for a gradient or ISM
    map, the token size for attention. `cache` is the gene's
    `PostMapCache`: with it the collapse, |M| and the divisor are read
    from the memo, without it they are computed here. Both paths perform
    identical arithmetic.
    """
    M_src = M
    if cache is not None:
        _, M, A, _ = cache.prepared(M, theta)
    else:
        M, A = _collapse_map(M, theta), None
    m = enh.mid() - w.start                                         # A.4 1
    L = M.shape[0]
    if not (0 <= m < L):                                            # A.4 2
        return out_of_window(theta)
    if theta["abs_map"]:                                            # A.4 3
        N = A if A is not None else np.abs(M)
    else:
        N = M
    norm = theta["normalise"]                                       # A.4 4-6
    if norm == "none":
        nu = 1.0
    elif cache is not None:
        nu = cache.divisors(M_src, theta)[norm]
    else:
        nu = map_divisors(M)[norm]
    if nu == 0.0:
        nu = 1.0
    op = theta["enh_agg"]
    if op == "point":                                               # A.4 8
        s = float(N[m])
    elif op in ("sum", "mean", "max"):                              # A.4 9-11
        half = theta["enh_span"] // 2
        lo, hi = snap_to_grid(m - half, m + half, grid, theta["bin_overlap"])
        lo, hi = max(0, lo), min(L, hi)
        s = reduce_over(N[lo:hi], op)
        if op == "sum" and grid > 1:
            # The map replicates each token's value across its `grid`
            # bp, so dividing restores the sum of whole-token attention
            # scores. A global constant, so inert for ranking; done only
            # to make the number mean what it says.
            s /= grid
    elif op == "gaussian":                                          # A.4 12-16
        sigma = float(theta["gaussian_sigma"])
        h = theta["gaussian_truncate"] * theta["gaussian_sigma"]
        lo, hi = max(0, m - h), min(L, m + h + 1)
        omega = _gaussian_weights(sigma, h)[lo - (m - h): hi - (m - h)]
        s = float((omega * N[lo:hi]).sum() / omega.sum())
    else:
        raise ConfigError(f"unknown enh_agg {op!r}")
    return s / nu                                                   # A.4 17


def statistic(d: np.ndarray, u_wt, kind: str, axis: int) -> np.ndarray:
    """Reduce the ISM draws over the shuffle axis `axis`.

    `u_wt` broadcasts against `d` with the shuffle axis removed, so a
    relative statistic divides each draw by the wild type of its own
    (window, replicate, augmentation).
    """
    if kind == "abs_mean_diff":                              # A.3 2  P inside
        return np.abs(d.mean(axis=axis))
    if kind == "abs_diff":                                   # A.3 3  P outside
        return np.abs(d).mean(axis=axis)
    # nan rather than a division by ~0, which would be an infinity
    # the relative statistics then rank above every real score.
    denom = np.where(np.abs(u_wt) < LOG_EPS, np.nan, u_wt)
    if kind == "abs_mean_rel_diff":                          # A.3 4  P inside
        return np.abs(d.mean(axis=axis) / denom)
    if kind == "rel_abs_diff":                               # A.3 5  P outside
        return np.abs(d / denom).mean(axis=axis)
    raise ConfigError(f"unknown ism_statistic {kind!r}")


#: Axes of the ISM draw tensor `d`, in order.
AXIS = {"shuffle": 0, "tss": 1, "replicate": 2, "strand": 3}


def ism_combine(D, u_wt, enh: Enhancer, windows, reps, augs, theta,
                gene_id=None):
    """Assemble one pair's draws into a tensor and reduce them.

    The absolute value is taken inside `statistic`.
    """
    P = theta["ism_shuffles"]
    # Under `track_abs_order: before` the wild type and every draw carry
    # a trailing track axis. It is appended last so that AXIS, which
    # indexes shuffle/tss/replicate/strand positively, still addresses
    # the same axes downstream.
    tail = np.shape(next(iter(u_wt.values()))) if u_wt else ()
    grid = (len(windows), len(reps), len(augs))
    d = np.zeros((P,) + grid + tail, dtype=np.float64)
    wt = np.ones((1,) + grid + tail, dtype=np.float64)
    present = False
    for iw in range(len(windows)):                                    # A.3 1-3
        for ir in range(len(reps)):
            for ia in range(len(augs)):
                wt[0, iw, ir, ia] = u_wt[(iw, ir, ia)]
                draws = D[(iw, ir, ia)].get(enh.eid)
                if draws is None:
                    if theta["out_of_window"] == "drop":
                        return ABSENT
                    continue                              # imputed zero
                d[:, iw, ir, ia] = draws
                present = True
    if not present:                                                   # A.3 4
        return out_of_window(theta)

    # A.3 5b  Persist the raw signed draws, before any reduction
    # touches them. Everything below this line is pure post-processing
    # of `d` and `wt`, so recording the draws makes all four
    # `ism_statistic` values and every subset of `ism_inner_mean` free,
    # where discarding them means running the model again to ask a
    # different question of the same perturbations. The statistic is an
    # inferred reading of the paper's "permuting", exactly the kind of
    # setting a residual sends you back to. The cost is about 10 float64
    # per pair.
    if ISM_DRAWS is not None:
        ISM_DRAWS.append({"gene": gene_id, "enhancer": enh.eid,
                          "draws": np.asarray(d, dtype=np.float64).ravel(),
                          "wt": np.asarray(wt, dtype=np.float64).ravel(),
                          "shape": tuple(np.shape(d))})

    return ism_reduce_draws(d, wt, theta, bool(tail))


def ism_reduce_draws(d: np.ndarray, wt: np.ndarray, theta: Dict[str, Any],
                     tail: bool = False):
    """Everything downstream of the raw ISM draws, and nothing else.

    Split out of `ism_combine` so that replaying a persisted
    `--ism-draws` file runs the identical arithmetic rather than a
    second copy of it: re-scoring an arm under a different
    `ism_statistic` is then a call to this function rather than another
    pass over the model.
    """
    inner = set(theta["ism_inner_mean"])
    # A.3 6-8  Signed averaging over the inner axes. keepdims, so the
    # axis numbering in AXIS stays valid for the axes that follow. The
    # wild-type scalar is averaged over the same axes, so a relative
    # statistic divides by the matching u_wt.
    for name in ("strand", "replicate", "tss"):
        if name in inner:
            d = d.mean(axis=AXIS[name], keepdims=True)
            wt = wt.mean(axis=AXIS[name], keepdims=True)
    # A.3 9
    q = statistic(d, wt[0], theta["ism_statistic"], axis=AXIS["shuffle"])
    if tail:                                                        # A.3 10b
        # Collapse the track axis explicitly with track_agg, once the
        # |.| inside `statistic` has been taken per track. Letting
        # the terminal `q.reshape(-1)` absorb it would force a mean,
        # silently turning `track_agg: sum` into an average, and sum is
        # what `agnostic_absfirst` = sum_t |df_t| calls for. Doing it
        # here rather than after the strand and replicate collapses is
        # safe because those are means, and means commute with the
        # reduction.
        q = _reduce_axis(q, axis=q.ndim - 1, op=theta["track_agg"])
    # At this point q has axes (tss, replicate, strand); collapse
    # whatever the inner set left out.
    if "strand" not in inner:                                       # A.3 11
        q = q.mean(axis=2, keepdims=True)
    if "replicate" not in inner:                                    # A.3 12
        q = q.mean(axis=1, keepdims=True)
    if "tss" not in inner:                                          # A.3 13
        return reduce_over(q.reshape(-1), theta["anchor_agg"] or "mean")
    return float(q.reshape(-1).mean())


def merge_maps(store: Dict[Tuple[int, int, int], np.ndarray],
               n_w: int, n_r: int, n_a: int, theta: Dict[str, Any]
               ) -> Tuple[Dict[Tuple[int, int, int], np.ndarray], int, int]:
    """Average the signed maps over whichever axes take |.| last.

    Two axes, two independent parameters, the same question one axis
    apart:
      * `tta_abs_order = after`        -> collapse the augmentation axis
      * `replicate_abs_order = after`  -> collapse the replicate axis
    Returns the collapsed store and the counts `combine` should be
    given, so a caller cannot shrink one without the other. A no-op when
    the axis has length 1, which is why every Avsec 2021 arm is
    unaffected whatever these are set to.

    Order within this function does not matter: a mean over the
    augmentation axis and a mean over the replicate axis commute, so
    collapsing both gives the grand mean either way. That is not true of
    `|.|`, which is the whole reason these parameters exist.

    Linder 2025's reference code accumulates on both axes rather than
    averaging, so its maps differ from these by a constant factor. That
    factor is provably inert here, because `normalise = global` divides
    by `|M|.mean()` over the same map and so leaves the reduction
    exactly scale-invariant, asserted to 1e-12 in the self-test. Taking
    the mean keeps the map's units, which matters only under
    `normalise = none`.
    """
    merge_a = theta["tta_abs_order"] == "after" and n_a > 1
    merge_r = theta["replicate_abs_order"] == "after" and n_r > 1
    if not (merge_a or merge_r):
        return store, n_r, n_a
    out: Dict[Tuple[int, int, int], np.ndarray] = {}
    for iw in range(n_w):
        rs = [0] if merge_r else range(n_r)
        for ir in rs:
            aas = [0] if merge_a else range(n_a)
            for ia in aas:
                src = [store[(iw, jr, ja)]
                       for jr in (range(n_r) if merge_r else [ir])
                       for ja in (range(n_a) if merge_a else [ia])]
                out[(iw, ir, ia)] = np.mean(src, axis=0)
    return out, (1 if merge_r else n_r), (1 if merge_a else n_a)


def combine(per_wra: Dict[Tuple[int, int, int], Any],
            n_w: int, n_r: int, n_a: int, theta: Dict[str, Any]):
    """Pool a pair's scores over augmentation, replicate and anchor."""
    s2 = []
    for iw in range(n_w):
        s1 = []
        for ir in range(n_r):
            vals = [per_wra[(iw, ir, ia)] for ia in range(n_a)]
            if any(v is ABSENT for v in vals):
                return ABSENT
            s1.append(float(np.mean(vals)))                # A 16 augmentations
        s2.append(float(np.mean(s1)))                      # A 17 replicates
    return reduce_over(s2, theta["anchor_agg"] or "mean")  # A 18 anchors


# ---- B the driver ---------------------------------------------------------

def score_columns(score: Any, theta: Dict[str, Any]) -> Dict[str, Any]:
    """Expand a pair's score into row columns.

    On a plain arm the score is one float and this is `{"score": s}`.
    Every row goes through `pair_row` and so through here, including
    the out-of-window rows: the DictWriter rejects a ragged row.
    """
    if isinstance(score, dict):
        return dict(score)
    # A bare float reaching here under an active free grid is the
    # out-of-window site: that branch never computes a map, so it has no
    # variants to reduce. Every column still has to appear, and the
    # value is the same for all of them because being out of window is a
    # property of the pair, not of the pooling.
    if FREE_GRID or MULTI_CHANNEL:
        cols = ((free_grid_columns() if FREE_GRID else ())
                + (multi_mix_columns() if MULTI_CHANNEL else ()))
        return dict({c: score for c in cols}, score=score)
    return {"score": score}


#: The free grid: post-map reductions emitted from one attribution map,
#: set by `--free-grid`.
#:
#: `enh_agg` and `normalise` act on the finished map, so neither changes
#: the forward or backward pass that produced it. Emitting one column
#: per cell is what makes them free in practice: reducing each map to a
#: single `{"score": s}` and discarding it costs one model pass per
#: cell, and on a 196 kb Enformer arm the pass is the entire cost.
#: Columns also beat persisting the raw maps, which run to gigabytes per
#: arm against ~40 bytes per row.
#:
#: They are overlays rather than configs because `validate()` refuses
#: `enh_span` beside `enh_agg=gaussian` and requires it for
#: `enh_agg=sum`: the cells cannot all be expressed as one validated
#: theta. Applying them directly to `reduce_pair` is safe precisely
#: because they are post-map, so no overlay can reach the model, the
#: window, the readout or the map.
FREE_GRID: Tuple[Tuple[str, Dict[str, Any]], ...] = ()

#: Which axes the grid spans. Everything here is read by `reduce_pair`
#: or `combine`, strictly after the map exists, so the whole cross costs
#: no forward or backward pass. At the backbone's `anchor=single-tss`
#: two post-map parameters are inert: `anchor_agg`, because there is one
#: window and mean/max/sum of one element coincide, and `bin_overlap`,
#: because a gradient map has grid=1 and there is nothing to re-snap.
#:
#: The levels are a dial, not a discovery. The full cross of every
#: plausible level is ~136 columns, which turns pairwise deltas into
#: thousands of rows of multiple-comparison soup. These levels give 24
#: cells plus one diagnostic, to be read as main effects against the
#: backbone cell rather than as an all-pairs tournament.
FREE_GRID_NORMALISATIONS = ("global", "none")
#: Gaussian cells as (matched_span, sigma, truncate): a full factorial
#: of 4 widths x truncate {1, 2, 3}, with sigma solved so each cell has
#: the same second moment as the flat window in FREE_GRID_SPANS that
#: labels it. That pairing is the point: at matched width `enh_agg` is a
#: clean test of kernel shape, whereas an unmatched gaussian-vs-flat
#: comparison confounds shape with width.
#:
#: A flat window of full width W has sd = W/sqrt(12); a gaussian
#: truncated at t*sigma has sd = sigma*c(t), where c depends only on t
#: (c(1)=0.539, c(2)=0.880, c(3)=0.987). So sigma = W/(sqrt(12)*c(t)),
#: and the effective sd runs 144 to 866, a 6x span.
#:
#: Holding sd fixed and letting sigma absorb t is what makes t a pure
#: shape axis, giving a clean 4 (width) x 4 (peakedness) design with the
#: flat window as the uniform limit: peak/mean at fixed width is 1.00
#: flat, 1.17 at t=1, 1.67 at t=2, 2.40 at t=3. Varying t at fixed sigma
#: instead moves the width (sd 177/289/324 at sigma=328) and saturates
#: above t=2, which is a width sweep wearing a shape sweep's clothes.
#:
#: Matching the second moment does not match the support, and the gap
#: grows with t: to match a flat +/-1500 window t=2 needs +/-1968, where
#: t=4 would need +/-3464.
#:
#: Cells are named by the matched span, so the pairing is legible in the
#: column names: `score__g1000-*` is the gaussian of the same effective
#: width as `score__s1000-*`.
FREE_GRID_GAUSSIANS = (
    (500, 268, 1), (500, 164, 2), (500, 147, 3),         # sd ~144
    (1000, 535, 1), (1000, 329, 2), (1000, 293, 3),      # sd ~289
    (2000, 1070, 1), (2000, 657, 2), (2000, 586, 3),     # sd ~577
    (3000, 1605, 1), (3000, 985, 2), (3000, 878, 3),     # sd ~866
)
#: Flat-window full widths in bp, each matched by the gaussian cell of
#: the same label above.
FREE_GRID_SPANS = (500, 1000, 2000, 3000)
#: mean_subtract is the one axis on which Avsec 2021 (false) differs
#: from Linder 2025 and Avsec 2026 (true), so on a three-model
#: comparison it is the most load-bearing free axis there is.
FREE_GRID_MEAN_SUBTRACT = (True, False)

#: Set when the grid varies `mean_subtract`. Makes `gradient_map`
#: return a two-channel map
#: `(L, 2) = [g_at_reference_base, mean_over_the_four_bases]` instead of
#: the single reduced channel.
#:
#: Two channels rather than a post-hoc transform: writing A for the
#: reference-base gradient and m for the per-position mean over the four
#: bases, the two variants are exactly A (false) and A - m (true), both
#: from one backward pass. But `abs_map` absolutises the map inside
#: `reduce_pair`, and |A - m| != |A| - |m|, so a finished single-channel
#: map cannot yield the other variant's score. The choice has to be made
#: before the absolute value, which is why the map itself carries both.
#: Persisting the raw 4-channel gradient would work too, at 4x the
#: storage; only the reference base and the 4-base mean are ever read.
#:
#: Positions stay on axis 0. `reduce_pair` uses `M.shape[0]` as the map
#: length for its out-of-window test, so a `(2, L)` layout would make
#: every pair look unreachable: a silent, total corruption.
FREE_MEAN_SUBTRACT: bool = False

#: Design B's per-gene group readout values, one per assay:
#: the scale-free channels are per-gene rescalings, so without these the
#: weights cannot be re-interpreted after the fact.
MULTI_SCALARS: Dict[str, Tuple[float, ...]] = {}

#: Weights over the map's channels, applied by `reduce_pair` under
#: `--multi-channels`. Every variant of the
#: combination is a linear functional of the channels, which is why four
#: are stored:
#:   raw mixing   a*dC + (1-a)*dR              -> (a, 1-a, 0, 0)
#:   log mixing   a*dC/C + (1-a)*dR/R          -> (0, 0, a, 1-a),
#:                the scale-free geometric mean
#:   contrast     d[C/(C+R)] up to a per-gene constant that
#:                `normalise: global` divides out:
#:                (R*dC - C*dR)/(C+R)^2
#:                  = [CR/(C+R)^2] * (dC/C - dR/R)  -> (0, 0, 1, -1)
#:
#: Design B, set by `--multi-channels` on an `assay=multi` arm, makes
#: `gradient_map` take one backward pass per assay group and return an
#: (L, 2N) map, `[dE_1..dE_N, dE_1/E_1..dE_N/E_N]`, so every weighting
#: over the assays, and every subset, becomes a post-map reduction
#: instead of a separate run. That is N backward passes plus one forward
#: per gene, roughly 4.3x the cost of a single-scalar arm.
#:
#: It is worth that because fixing the weighting at equal-per-assay in
#: log space already measures +0.0405 (p=0.001) over log(CAGE) alone at
#: matched calibration: agreement carries signal.
#:
#: The wiring check: the equal-weight log cell of this map must
#: reproduce the single-scalar equal-weight arm, because
#: sum_a d log(E_a)/dx is d [sum_a log E_a]/dx. If it does not, the
#: per-group scalar is not one term of that sum and nothing else here is
#: interpretable.
MULTI_CHANNEL: bool = False

#: Weights over the channels: length 4 under the two-readout
#: multi-channel path, length 2N. A plain tuple
#: rather than a fixed arity so both widths share one `reduce_pair`
#: branch, which checks the width against the map it is handed.
CO_W: Tuple[float, ...] = (0.5, 0.5, 0.0, 0.0)

def build_multi_mix(n: int, names: Tuple[str, ...]
                    ) -> Tuple[Tuple[str, Tuple[float, ...]], ...]:
    """Design B's mix grid over 2N channels, as (name, weights).

    N is 4 for the K562 promoter set (CAGE, DNase, H3K4me3, RNAPII),
    so each weight vector is length 8: the first four weight the raw
    gradients, the last four the scale-free ones. Built here rather
    than written out, because an eight-element vector entered by hand
    is exactly the kind of literal that silently addresses the wrong
    assay.

    The cells, and what each answers:
      eqlog       equal weight, scale-free: must reproduce the
                  single-scalar equal-weight arm (the wiring check)
      eqraw       equal weight, raw: what a naive pooled track_agg does
      only-<a>    one assay alone, scale-free: the per-assay ablation
      drop-<a>    the other three: leave-one-out, complementing `only`
      cage-heavy  CAGE 0.7, rest 0.1 each: is CAGE load-bearing or
                  interchangeable?
    `drop-cage` doubles as the chromatin-only cell, the three marks at
    1/(n-1) each, so there is no separate cell for it.
    """
    zero = [0.0] * n
    out = [("eqlog", tuple(zero) + tuple([1.0 / n] * n)),
           ("eqraw", tuple([1.0 / n] * n) + tuple(zero))]
    for i, nm in enumerate(names):
        only = list(zero)
        only[i] = 1.0
        out.append((f"only-{nm}", tuple(zero) + tuple(only)))
        drop = [1.0 / (n - 1)] * n
        drop[i] = 0.0
        out.append((f"drop-{nm}", tuple(zero) + tuple(drop)))
    heavy = [0.1] * n
    heavy[0] = 0.7
    out.append(("cage-heavy", tuple(zero) + tuple(heavy)))
    return tuple(out)


MULTI_MIX: Tuple[Tuple[str, Tuple[float, ...]], ...] = ()
#: Names of the assay groups behind the current MULTI_MIX, used for
#: column naming.
MULTI_MIX_NAMES: Tuple[Tuple[str, int], ...] = ()


def multi_mix_columns() -> Tuple[str, ...]:
    """Columns the design-B grid contributes, MULTI_MIX x normalise."""
    cols = tuple(f"bmix-{nm}-{norm}" for nm, _ in MULTI_MIX
                 for norm in ("global", "none"))
    # Plus one readout value per assay: E_a is a per-gene constant, and
    # without it no weighting can be re-interpreted after the run.
    return cols + tuple(f"e_{nm}" for nm, _ in MULTI_MIX_NAMES)


def build_free_grid(theta: Dict[str, Any]
                    ) -> Tuple[Tuple[str, Dict[str, Any]], ...]:
    """The post-map cross of pooling and normalisation cells.

    Cells are named `score__<pooling>-<normalise>` with `-ms<0|1>` and
    `-r<order>` suffixes on the axes that are live, plus one `abs_map`
    diagnostic. Pooling levels are `gaussian` at each entry of
    FREE_GRID_GAUSSIANS and `sum` at each of FREE_GRID_SPANS.

    `mean_subtract` is included only when the two-channel map is active.
    Otherwise the map has already committed to one variant and the other
    is unrecoverable, so a column for it would silently repeat the
    committed value.
    """
    # Gaussian cells are named by the matched flat span and the
    # truncation, so a `g1000t2` cell pairs with the `s1000` flat cell;
    # the sigma is recoverable from FREE_GRID_GAUSSIANS.
    geoms = [(f"g{W}t{t}", {"enh_agg": "gaussian", "enh_span": None,
                            "gaussian_sigma": s, "gaussian_truncate": t})
             for W, s, t in FREE_GRID_GAUSSIANS]
    geoms += [(f"s{p}", {"enh_agg": "sum", "enh_span": p,
                         "gaussian_sigma": None, "gaussian_truncate": None})
              for p in FREE_GRID_SPANS]
    subs = (FREE_GRID_MEAN_SUBTRACT if FREE_MEAN_SUBTRACT
            else (theta["mean_subtract"],))
    # `replicate_abs_order` is free, because `merge_maps` decides it
    # after the maps exist, but it is a no-op when the replicate axis
    # has length 1. Crossing it in would double the column count with
    # exact duplicates, so it is added only where it is live: Borzoi at
    # replicates=4. Enformer and AlphaGenome keep the narrower grid.
    rao_live = theta["replicates"] > 1
    raos = ("before", "after") if rao_live else (theta["replicate_abs_order"],)
    grid = []
    for gname, geom in geoms:
        for norm in FREE_GRID_NORMALISATIONS:
            for ms in subs:
                for rao in raos:
                    name = (f"score__{gname}-{norm}"
                            + (f"-ms{int(ms)}" if FREE_MEAN_SUBTRACT else "")
                            + (f"-r{rao}" if rao_live else ""))
                    grid.append((name, dict(geom, normalise=norm,
                                            mean_subtract=ms,
                                            replicate_abs_order=rao)))
    # One abs_map diagnostic at the backbone cell rather than doubling
    # everything: abs_map=false keeps the signed map, so scores can be
    # negative and the ranking is a different object entirely. Worth one
    # number to see whether the sign carries information.
    backbone = (geoms[0][1] if theta["enh_agg"] == "gaussian"
                else geoms[-1][1])
    grid.append(("score__signed-backbone",
                 dict(backbone, normalise=theta["normalise"],
                      mean_subtract=theta["mean_subtract"],
                      abs_map=False)))
    return tuple(grid)


def free_grid_columns() -> Tuple[str, ...]:
    """Column names the grid contributes.

    Every row must carry exactly these. The `nu_*` columns are divisor
    values, not scores; they are listed here because the DictWriter
    needs every key on every row, including the out-of-window rows,
    which is also why `out_of_window` fills them with the same
    sentinel as the scores. A pair with no map has no divisor either,
    so the sentinel is honest.
    """
    if not FREE_GRID:
        return ()
    co = ()
    subs = sorted({dict({"mean_subtract": None}, **ov).get("mean_subtract")
                   for _, ov in FREE_GRID} - {None}) or [None]
    nus = tuple(
        f"nu_{k}" + (f"-ms{int(ms)}"
                     if FREE_MEAN_SUBTRACT and ms is not None else "")
        for ms in subs for k in DIVISOR_KINDS)
    # Dedupe while preserving order: without FREE_MEAN_SUBTRACT both ms
    # values name the same column.
    seen, uniq = set(), []
    for c in nus:
        if c not in seen:
            seen.add(c)
            uniq.append(c)
    return tuple(name for name, _ in FREE_GRID) + tuple(uniq) + co


def _mix_scores(mix: Sequence[Tuple[str, Tuple[float, ...]]], prefix: str,
                store: Dict[Tuple[int, int, int], Any], enh: Enhancer,
                windows: Sequence[Window], n_w: int, n_reps: int,
                n_augs: int, theta: Dict[str, Any],
                grid: int, cache: Optional[PostMapCache] = None
                ) -> Dict[str, Any]:
    """Score one pair under every `mix` cell crossed with `normalise`.

    The channels already exist, so a weighting is arithmetic: set CO_W,
    reduce the same maps again, pool. `merge_maps` goes through the
    gene's `PostMapCache`, so it runs once per distinct merge rather
    than once per cell, and each weighting's collapse is keyed on CO_W.

    `normalise` is crossed in because it decides whether the scale-free
    channels' per-gene divisor survives. Under `global` the score is
    Sum w|M| / mean|M|, so any per-gene rescaling of M cancels exactly;
    a channel can therefore read as nearly inert there while moving the
    score by an order of magnitude at `none`.
    """
    global CO_W
    out: Dict[str, Any] = {}
    saved = CO_W
    pm = cache if cache is not None else PostMapCache()
    try:
        for name, wts in mix:
            for norm in ("global", "none"):
                vtheta = dict(theta, normalise=norm)
                merged, n_r, n_a = pm.merged(store, n_w, n_reps, n_augs,
                                             vtheta)
                CO_W = wts
                per = {k: reduce_pair(M, enh, windows[k[0]], vtheta, grid,
                                      cache=pm)
                       for k, M in merged.items()}
                out[f"{prefix}-{name}-{norm}"] = combine(
                    per, n_w, n_r, n_a, vtheta)
    finally:
        CO_W = saved
    return out


def multi_mix_scores(store: Dict[Tuple[int, int, int], Any], enh: Enhancer,
                     windows: Sequence[Window], n_w: int, n_reps: int,
                     n_augs: int, theta: Dict[str, Any], grid: int,
                     gene_gid: Optional[str] = None,
                     cache: Optional[PostMapCache] = None) -> Dict[str, Any]:
    """Score one pair under every MULTI_MIX x normalise cell.

    The expensive
    object is the N backward passes, and every weighting of them is
    arithmetic.
    """
    out = _mix_scores(MULTI_MIX, "bmix", store, enh, windows, n_w, n_reps,
                      n_augs, theta, grid, cache=cache)
    # The per-assay readout values themselves: storing them keeps a
    # later re-weighting or pseudocount interpretable without going back
    # to the GPU.
    vals = MULTI_SCALARS.get(gene_gid or "", ())
    groups = MULTI_GROUPS.get((MODEL_META[theta["model"]].name,
                               theta["assay"], theta["cell_type"]), ())
    for i, (nm, _) in enumerate(groups):
        out[f"e_{nm}"] = float(vals[i]) if i < len(vals) else float("nan")
    return out


def divisor_columns(maps: Sequence[np.ndarray], ms: bool,
                    cache: Optional[PostMapCache] = None) -> Dict[str, float]:
    """One pair's `nu_*` columns: every divisor, averaged over its maps.

    This is what makes the WHOLE normalise family free rather than one
    more member of it. `reduce_pair` ends in `s / nu`, so the divisor
    factorises out of the pooling: the score under a stored
    `nu_<kind>-ms<k>` is the stored `score__<pooling>-none-ms<k>`
    divided by it. Ten kinds by sixteen poolings by two mean_subtract
    values is 320 derived scores out of 18 extra columns.

    EXACTNESS CAVEAT, and the reason this is not simply better in every
    case: `combine` averages the per-(window, replicate, augmentation)
    reductions and `nu` is applied INSIDE each one, so the quotient
    reproduces the pipeline exactly only when a pair has a single map.
    That holds for Enformer at replicates=1 / tta=none / single-tss and
    fails for Borzoi's four replicates or any tta arm. Where it fails
    the divisor has to be swept as a real `normalise` value, which is
    legal. `nu_global` is stored alongside so this is checkable rather
    than assumed: it must equal score__X-none / score__X-global to
    machine precision, and where it does not, the factorisation fails
    for that arm.

    `ms` collapses a two-channel map exactly as `reduce_pair` does, so
    that `nu` describes the same array the score was taken from. With
    `cache` the divisors are the ones the grid cells already computed
    for the same map, read rather than recomputed.
    """
    divs = []
    for M in maps:
        two_channel = M.ndim == 2 and M.shape[1] == 2
        if M.ndim == 2 and not two_channel:   # (L, 4): nu does not factorise
            divs = []
            break
        if cache is not None and not MULTI_CHANNEL:
            # the same collapse, keyed as `reduce_pair` keys it
            divs.append(cache.divisors(M, {"mean_subtract": ms}))
            continue
        Mc = (M[:, 0] - M[:, 1] if ms else M[:, 0]) if two_channel else M
        divs.append(map_divisors(Mc))
    suffix = f"-ms{int(ms)}" if FREE_MEAN_SUBTRACT else ""
    return {f"nu_{kind}{suffix}": (float(np.mean([d[kind] for d in divs]))
                                   if divs else float("nan"))
            for kind in DIVISOR_KINDS}


def free_grid_scores(store: Dict[Tuple[int, int, int], Any], enh: Enhancer,
                     windows: Sequence[Window], n_w: int, n_reps: int,
                     n_augs: int, theta: Dict[str, Any], grid: int,
                     gene_gid: Optional[str] = None,
                     cache: Optional[PostMapCache] = None) -> Dict[str, Any]:
    """Score one pair under every overlay, reusing its computed maps.

    Takes the UNMERGED `store`, not the merged maps, so that
    `replicate_abs_order` and `tta_abs_order` can vary too: both are
    decided inside `merge_maps`, which averages the signed maps over
    whichever axis takes |.| last. Post-merge maps would freeze those
    two axes at the config's values, and `replicate_abs_order` is live
    on every Borzoi arm (replicates=4) and is the axis Linder 2025's
    Fig 4c,d turns on.

    `merge_maps` runs once per DISTINCT merge, not once per cell: only
    two parameters reach it, so a grid that varies one of them merges
    twice however many cells it holds. Without that cache the merge,
    which copies every map, would run once per cell per pair. `cache` is
    the gene's `PostMapCache`, shared with the primary score and with
    every other enhancer of the gene.

    Reachability does not depend on the overlay: `reduce_pair`'s
    out-of-window test compares the enhancer midpoint against the map
    geometrically, before it reads `enh_agg` or `normalise` at all. A
    variant therefore cannot disagree with the primary about whether a
    pair is scoreable, and if one ever does that is a bug in this
    reasoning rather than a number to record.
    """
    out: Dict[str, Any] = {}
    pm = cache if cache is not None else PostMapCache()
    for name, overlay in FREE_GRID:
        vtheta = dict(theta, **overlay)
        merged, n_r, n_a = pm.merged(store, n_w, n_reps, n_augs, vtheta)
        per = {k: reduce_pair(M, enh, windows[k[0]], vtheta, grid, cache=pm)
               for k, M in merged.items()}
        out[name] = combine(per, n_w, n_r, n_a, vtheta)

    ms_values = sorted({dict(theta, **ov)["mean_subtract"]
                        for _, ov in FREE_GRID})
    merged0, _, _ = pm.merged(store, n_w, n_reps, n_augs, theta)
    for ms in ms_values:
        out.update(divisor_columns(list(merged0.values()), ms, cache=pm))

    return out


def score_pair(enh: Enhancer, gene: Gene, theta: Dict[str, Any], rho: Resolved,
               backend: Backend):
    """Score one pair: the definition, with no cache and no grouping.

    This is the contract `run_experiment` must meet for every pair and
    every configuration, and `_test_definition_matches_driver` checks
    that it does. The procedure stays literal even where that is
    wasteful, because its job is to be obviously correct rather than
    fast.
    """
    A = anchor_windows(enh, gene, theta, rho, backend)
    if not A:
        return out_of_window(theta)
    R = tuple(range(1, theta["replicates"] + 1))
    T = augmentations(theta, gene.gid)

    if theta["method"] == "ism":
        D, wt = {}, {}
        for iw, w in enumerate(A):
            for ir, r in enumerate(R):
                for ia, a in enumerate(T):
                    D[(iw, ir, ia)], wt[(iw, ir, ia)] = \
                        ism_map(gene, [enh], w, r, a, theta, rho, backend)
        return ism_combine(D, wt, enh, A, R, T, theta)

    maps: Dict[Tuple[int, int, int], Any] = {}
    for iw, w in enumerate(A):
        for ir, r in enumerate(R):
            for ia, a in enumerate(T):
                maps[(iw, ir, ia)] = attribution_map(
                    w, gene, r, a, theta, rho, backend)
    maps, n_r, n_a = merge_maps(maps, len(A), len(R), len(T), theta)
    grid = map_grid(theta, backend.meta)
    per = {k: reduce_pair(M, enh, A[k[0]], theta, grid)
           for k, M in maps.items()}
    return combine(per, len(A), n_r, n_a, theta)


def build_pair_table(pairs: List[Tuple[Enhancer, Gene]],
                     theta: Dict[str, Any]) -> List[Tuple[Enhancer, Gene]]:
    """Apply the configured pair filters to the loaded pair list."""
    f = theta["pair_filters"] or {}
    out = list(pairs)
    # Each gene must already carry the TSS set and exon set of THIS
    # GENCODE release. The loader attaches them; assert it, because a
    # mismatched annotation is invisible downstream and every anchor
    # and every exon readout depends on it.
    wanted = theta["gencode"]
    for _, g in out:
        got = getattr(g, "gencode", None)
        if got is not None and got != wanted:
            raise ConfigError(
                f"gene {g.gid} carries GENCODE {got} annotations "
                f"but θ.gencode={wanted}")
    if f.get("require_gene_in_annotation"):
        out = [(e, g) for e, g in out if g.gencode_tss and g.exons]
    if f.get("max_tss_distance"):
        lim = int(f["max_tss_distance"])
        out = [(e, g) for e, g in out if abs(e.mid() - g.provided_tss) <= lim]
    if f.get("min_tss_distance"):
        # Purely a compute optimisation for testing one distance bin,
        # and safe as one because a pair's score never depends on which
        # other pairs are present: a bin's auPRC from a
        # distance-restricted table is bit-identical to the full run's.
        # Measured against `provided_tss`, matching `max_tss_distance`
        # and the figure's own binning, and not against the anchor,
        # which can sit somewhere else entirely.
        lo = int(f["min_tss_distance"])
        out = [(e, g) for e, g in out if abs(e.mid() - g.provided_tss) >= lo]
    if f.get("min_elements_per_gene"):
        need = int(f["min_elements_per_gene"])
        counts: Dict[str, int] = {}
        for _, g in out:
            counts[g.gid] = counts.get(g.gid, 0) + 1
        out = [(e, g) for e, g in out if counts[g.gid] >= need]
    return out


def gpu_subset(theta: Dict[str, Any]) -> Dict[str, Any]:
    """Return the theta fields the artefact cache key depends on.

    Everything listed here changes the stored artefact itself, so two
    runs differing in any of them must not share a cached map. The
    parameters left out are pure reductions of a map already computed,
    and are what makes the free grid free.
    """
    keys = ("model", "replicates", "gencode", "anchor", "tss_major",
            "tss_major_assay", "tss_major_cell_type",
            "assay", "cell_type", "region", "tss_width", "track_agg",
            # readout_anchor moves READOUT-BINS off the window centre
            # onto the gene's own TSS, so it changes the differentiated
            # scalar and therefore the map. GPU-forcing for the same
            # reason bin_overlap is at region=exons.
            "readout_anchor",
            # GPU-forcing twice over: it moves EXON-CENTROID, so the
            # input window moves, and it changes READOUT-BINS, so the
            # differentiated scalar changes.
            "exon_source",
            # A second backward pass against a second scalar, so
            # `gradient_map` returns four channels instead of two: a map
            # cached without it cannot serve a run with it, because the
            # shapes differ. The key already separates the two settings
            # through its `tracks` component, so listing it here
            # changes no result. It is listed anyway so that the key is
            # faithful to its contract rather than correct by
            # coincidence.
            "strand_mask", "inverse_transform", "log_scalar", "method",
            # Both change the artefact and not merely its reduction:
            # bin_abs_order changes how many backward passes a gene
            # takes, and track_abs_order changes the stored draw from a
            # scalar to a per-track vector. The cache is per-run and
            # both are constant within a run, so listing them cannot
            # change a result; it keeps the key correct for a cache that
            # outlives one run.
            "bin_abs_order", "track_abs_order",
            # Not reductions at all: these decide which bases are
            # perturbed and with what, so every forward pass differs.
            "ism_window", "ism_mode", "ism_shuffles",
            # Also artefact-level: tta decides how many passes run and
            # which sequence each one sees, and tta_shift_bp sets the
            # offset those shifted sequences are taken at.
            "tta", "tta_shift_bp",
            "seed")
    d = {k: theta[k] for k in keys}
    if theta["method"] != "attention":
        # READOUT-BINS consumes it, so it changes the differentiated
        # scalar and therefore the map. On the attention path no gene
        # scalar is computed, the map does not depend on it, and the
        # parameter is free.
        d["bin_overlap"] = theta["bin_overlap"]
    return d


def cache_key(theta: Dict[str, Any], rho: Resolved, gene: Gene,
              w: Window, r: Any, a: Aug) -> str:
    """A short digest identifying one cached artefact."""
    blob = json.dumps({"theta": gpu_subset(theta), "tracks": list(rho.tracks),
                       "gene": gene.gid, "w": [w.start, w.length],
                       "r": str(r), "a": [a.shift, a.rc]}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def pair_row(rank: int, theta: Dict[str, Any], gene: Gene, enh: Enhancer,
             in_window: bool, score: Any) -> Dict[str, Any]:
    """One output row: the pair's identity, then its score columns.

    Every `rows.append` site goes through here, so the columns cannot
    drift between the scored and the out-of-window paths; the
    DictWriter rejects a ragged row.
    """
    row = {"_gene_rank": rank,
           "screen": theta["screen"], "gene": gene.gid,
           "enhancer": enh.eid, "chrom": enh.chrom,
           "start": enh.start, "end": enh.end,
           "distance_to_tss": enh.mid() - gene.provided_tss,
           "in_window": in_window}
    row.update(score_columns(score, theta))
    return row


@dataclass
class RunStats:
    """What one run did, for the manifest."""

    gpu_calls: int = 0
    cache_hits: int = 0
    pairs_scored: int = 0
    pairs_dropped: int = 0
    #: "i/n" when this process computed one shard of the gene list,
    #: else None. Recorded so that a partial score table can never be
    #: mistaken for a complete one.
    shard: Optional[str] = None


def run_experiment(pairs: List[Tuple[Enhancer, Gene]], theta: Dict[str, Any],
                   backend: Backend) -> Tuple[List[Dict[str, Any]], RunStats]:
    """Score every pair in `pairs`, sharing one artefact per gene.

    This is the driver `score_pair` defines the contract for: it caches
    the expensive artefact across the enhancers of a gene and rescores
    the free grid from it, and must still return, pair for pair, what
    the definition returns.
    """
    warns = validate(theta)
    rho = resolve(theta, backend.meta)
    # Anchor-table warnings are raised at load time, before this
    # function runs, so they arrive through the module accumulator
    # rather than from validate(). Without this the manifest reports
    # "warnings": [] for a run whose anchor provenance could not be
    # verified.
    rho.warnings = list(warns) + [w for w in ANCHOR_WARNINGS if w not in warns]
    # The frame guard, before any model call: refuse an augmentation
    # whose frame cannot be exact, certify the readout on the model's own
    # geometry with the exact probe, and arm the run-time agreement gate.
    check_frame_contract(theta, backend)
    FRAME_REPORT.clear()
    FRAME_REPORT.update(frame_selftest(backend.meta, theta))
    frame_gate = FrameAgreementGate(theta)
    reps = tuple(range(1, theta["replicates"] + 1))
    table = build_pair_table(pairs, theta)
    stats = RunStats()
    cache: Dict[str, Any] = {}
    rows: List[Dict[str, Any]] = []

    by_gene: Dict[str, List[Enhancer]] = {}
    genes: Dict[str, Gene] = {}
    order: List[str] = []
    for e, g in table:
        if g.gid not in by_gene:
            by_gene[g.gid], genes[g.gid] = [], g
            order.append(g.gid)
        by_gene[g.gid].append(e)

    # A `--only-genes` ID the cohort does not contain is a typo or an
    # outdated alias. Left alone it fails quietly: a table of zero rows
    # that a top-up step merges as "nothing to add", leaving the base
    # table short while every step exits 0.
    if ONLY_GENES is not None:
        missing = sorted(ONLY_GENES - by_gene.keys())
        if missing:
            raise ConfigError(
                "--only-genes names %d gene(s) absent from this "
                "cohort: %s. The cohort has %d genes; check "
                "GENE_ID_ALIASES and the screen's own gene column"
                % (len(missing), ", ".join(missing), len(by_gene)))

    enhancer_anchored = theta["anchor"] in ("enhancer",
                                            "enhancer-tss-midpoint")

    # Progress reporting is pure instrumentation and has to stay that
    # way: it reads `stats` and the clock, touches nothing else, and no
    # score can depend on it.
    #
    # It writes to stderr and flushes explicitly. Redirected stdout is
    # block-buffered at about 8 KB, so a long run's log sits at zero
    # bytes and a live run is indistinguishable from a stalled one. It
    # reports work DONE rather than work started, and extrapolates the
    # finish from the measured rate rather than from an assumed one.
    t_start = time.time()
    n_genes = len(order)
    try:
        every = int(os.environ.get("SCORE_EG_PROGRESS_EVERY", "200"))
    except ValueError:
        every = 200            # a bad env value must not kill a long run

    def report_progress(done: int) -> None:
        elapsed = time.time() - t_start
        rate = done / elapsed if elapsed > 0 else 0.0
        eta = (n_genes - done) / rate / 60.0 if rate > 0 else float("nan")
        sys.stderr.write(
            "progress %d/%d genes (%.1f%%)  %.0f s elapsed  %.2f genes/s  "
            "%d gpu calls  %d pairs scored  ETA %.0f min\n"
            % (done, n_genes, 100.0 * done / max(n_genes, 1), elapsed,
               rate, stats.gpu_calls, stats.pairs_scored, eta))
        sys.stderr.flush()

    def gather_artefacts(gene: Gene, group: Sequence[Enhancer],
                         windows: Sequence[Window], augs: Sequence[Aug]):
        """The maps for one gene, keyed (window, replicate, augment).

        Every enhancer of the group reads the same maps, which is the
        whole saving over `score_pair`: the expensive artefact is a
        property of the gene and the window, not of the pair. The
        second dict is the ISM wild type, empty for the other methods.
        """
        store: Dict[Tuple[int, int, int], Any] = {}
        wt: Dict[Tuple[int, int, int], float] = {}
        gate_T: Dict[Tuple[int, int, int], Any] = {}   # the agreement gate
        for iw, w in enumerate(windows):
            for ir, r in enumerate(reps):
                for ia, a in enumerate(augs):
                    key = cache_key(theta, rho, gene, w, r, a)
                    if key in cache:
                        stats.cache_hits += 1
                        artefact = cache[key]
                    elif theta["method"] == "ism":
                        artefact = ism_map(gene, group, w, r, a,
                                           theta, rho, backend)
                        stats.gpu_calls += (
                            1 + len(artefact[0]) * theta["ism_shuffles"])
                        cache[key] = artefact
                        gate_T[(iw, ir, ia)] = artefact[1]
                    else:
                        artefact = attribution_map(w, gene, r, a,
                                                   theta, rho, backend)
                        stats.gpu_calls += 1
                        cache[key] = artefact
                        gate_T[(iw, ir, ia)] = getattr(backend, "_last_T",
                                                       float("nan"))
                    if theta["method"] == "ism":
                        store[(iw, ir, ia)] = artefact[0]
                        wt[(iw, ir, ia)] = artefact[1]
                    else:
                        store[(iw, ir, ia)] = artefact
        frame_gate.observe(gene.gid, augs, gate_T, backend.meta)
        return store, wt

    def score_enhancer(gene: Gene, enh: Enhancer,
                       windows: Sequence[Window], augs: Sequence[Aug],
                       store: Dict[Tuple[int, int, int], Any],
                       wt: Dict[Tuple[int, int, int], float],
                       pm: PostMapCache):
        """One pair's score, plus the columns any free grid adds.

        The maps are rescored under every post-map overlay here, while
        the pair and its maps are both in hand: this is the only point
        in the program where they are. `pm` is the gene group's
        `PostMapCache`, so the merges, collapses, |M| and divisors are
        computed once for all its enhancers and grid cells.
        """
        if theta["method"] == "ism":
            return ism_combine(store, wt, enh, windows, reps, augs, theta,
                               gene_id=gene.gid)
        n_w, n_reps, n_augs = len(windows), len(reps), len(augs)
        grid = map_grid(theta, backend.meta)
        merged, n_r, n_a = pm.merged(store, n_w, n_reps, n_augs, theta)
        per = {k: reduce_pair(M, enh, windows[k[0]], theta, grid, cache=pm)
               for k, M in merged.items()}
        score = combine(per, n_w, n_r, n_a, theta)
        if FREE_GRID and score is not ABSENT:
            # `store`, not `merged`: the grid re-merges per variant, so
            # replicate_abs_order and tta_abs_order are free axes too.
            extra = free_grid_scores(store, enh, windows, n_w, n_reps,
                                     n_augs, theta, grid,
                                     gene_gid=gene.gid, cache=pm)
            absent = [k for k, v in extra.items() if v is ABSENT]
            if absent:
                raise AssertionError(
                    f"free-grid variant(s) {absent} returned "
                    f"OUT-OF-WINDOW for {gene.gid}/{enh.eid} while the "
                    "primary reduction did not. Reachability is decided "
                    "before enh_agg/normalise are read, so this cannot "
                    "happen unless reduce_pair's out-of-window test has "
                    "started depending on the pooling.")
            score = dict(extra, score=score)
        if MULTI_CHANNEL and score is not ABSENT:
            bmix = multi_mix_scores(store, enh, windows, n_w, n_reps,
                                    n_augs, theta, grid, gene_gid=gene.gid,
                                    cache=pm)
            base = score if isinstance(score, dict) else {"score": score}
            score = dict(bmix, **base)
        return score

    def emit_unreachable(rank: int, gene: Gene,
                         group: Sequence[Enhancer]) -> None:
        """Record pairs that no anchor window reaches.

        `in_window` is False by construction, written literally rather
        than through `pair_in_window`, so the column cannot silently
        disagree with the branch that produced it.
        """
        for e in group:
            s = out_of_window(theta)
            if s is ABSENT:
                stats.pairs_dropped += 1
                continue
            stats.pairs_scored += 1
            rows.append(pair_row(rank, theta, gene, e, False, s))

    # Sharding. `order` is built from the WHOLE pair table on every
    # shard, so `i` is a global gene rank and the shards partition the
    # genes without any of them needing to know what the others did.
    # Rows carry `_gene_rank`, and a stable sort on it rebuilds exactly
    # the table an unsharded run writes.
    #
    # Row order is not cosmetic here: the paper's bootstrap numbers its
    # resampling units in table order, so a permutation of the same rows
    # changes every interval and p-value. A shard-and-concatenate that
    # got the order wrong would move them while looking perfectly
    # healthy. Hence a stable sort on a recorded rank rather than a
    # concatenation, and hence a self-test that demands
    # sharded-then-merged be BIT-IDENTICAL to unsharded, not close.
    shard_i, shard_n = SHARD
    if shard_n > 1:
        stats.shard = f"{shard_i}/{shard_n}"
    for i, gid in enumerate(order):
        if shard_n > 1 and i % shard_n != shard_i:
            continue
        # Applied AFTER the shard test and against the same global `i`,
        # so a restricted run is a subset of the rows the unrestricted
        # run writes, in the same order.
        if ONLY_GENES is not None and gid not in ONLY_GENES:
            continue
        if every > 0 and i > 0 and i % every == 0:
            report_progress(i)
        gene, enhs = genes[gid], by_gene[gid]
        # Inside the gene loop, not hoisted above it: `tta = avsec8`
        # draws a fresh eight per input sequence, as Avsec 2021 augments
        # each sequence independently, and the four deterministic
        # settings return the same tuple every time. Hoisting would buy
        # nothing and would make the random draw wrong. `score_pair`
        # calls this with the same key, which is what keeps the
        # definition and the driver equal pair for pair.
        augs = augmentations(theta, gid)
        # Skip a gene with nothing scoreable, before spending a model
        # call on it.
        #
        # Every arm decides out-of-window by the same one-line test:
        # the enhancer MIDPOINT inside the window. `reduce_pair` returns
        # OUT-OF-WINDOW before it looks at `enh_span` at all, so a span
        # that overlaps the edge contributes nothing, and `ism_map`
        # applies the identical test. If no enhancer of this gene has
        # its midpoint in any window, every pair of the gene returns
        # OUT-OF-WINDOW whatever the method, and the pass that produced
        # the map was spent on nothing. On Gasperini two thirds of the
        # genes are in that state.
        #
        # This is neither a heuristic nor an approximation: the rows
        # written below are exactly the rows the full path writes, which
        # is why the self-test asserts bit-identity rather than
        # closeness.
        #
        # The check itself is free, being integer comparisons on
        # coordinates with no model and no sequence. The windows it
        # needs are REUSED below rather than recomputed: with
        # `--anchors` MAJOR-TSS is a dict lookup, but without it
        # MAJOR-TSS costs one forward pass per candidate TSS, so
        # computing the windows twice per gene would double that. The
        # saving must not be paid for on the path that has no
        # precomputed anchor.
        gene_windows = None
        if not enhancer_anchored:
            gene_windows = anchor_windows(None, gene, theta, rho, backend)
            if not any(0 <= e.mid() - w.start < w.length
                       for w in gene_windows for e in enhs):
                emit_unreachable(i, gene, enhs)
                continue
        # An enhancer-dependent anchor moves the whole block inside the
        # pair loop, since the window then differs per enhancer.
        groups = [[e] for e in enhs] if enhancer_anchored else [enhs]
        for group in groups:
            windows = (anchor_windows(group[0], gene, theta, rho, backend)
                       if enhancer_anchored else gene_windows)
            # Readout-anchor reachability. With `readout_anchor =
            # gene-tss` the readout is pinned to the gene's TSS while
            # the anchor places the window elsewhere, so the TSS can
            # fall outside the model's PREDICTED region: Enformer emits
            # only the central 114,688 bp of its 196,608 bp input.
            # `gene_scalar_fn` raises on an empty readout, and that
            # guard is unreachable while the readout sits at w.centre,
            # which is predicted by construction. Here it is reachable,
            # so the pair is resolved as OUT-OF-WINDOW exactly as an
            # unreachable enhancer is, rather than killing the arm on
            # its first distant pair.
            if theta.get("readout_anchor") == "gene-tss" and windows:
                if all(bin_of(gene.provided_tss, w_, backend.meta) is None
                       for w_ in windows):
                    emit_unreachable(i, gene, group)
                    continue
            store, wt = gather_artefacts(gene, group, windows, augs)
            # One post-map memo per gene group, dropped with it.
            pm = PostMapCache()
            for e in group:
                score = score_enhancer(gene, e, windows, augs, store, wt, pm)
                if score is ABSENT:
                    stats.pairs_dropped += 1
                    continue
                stats.pairs_scored += 1
                rows.append(pair_row(i, theta, gene, e,
                                     pair_in_window(e, windows), score))
    # The final line carries the measured rate for the whole run, so a
    # throughput can be read rather than guessed. It is gated on the run
    # being big enough to have been tracked at all: unconditionally it
    # would print one line per config through the self-test, where it
    # would bury a real FAIL, and a cohort smaller than one reporting
    # interval finishes before there is anything to report.
    if every > 0 and n_genes >= every:
        report_progress(n_genes)
    if not frame_gate.checked:
        frame_gate.check()             # a short run is checked at its end
    FRAME_REPORT["agreement"] = frame_gate.report()
    return rows, stats


# ============================================================================
# 7. CLI
# ============================================================================

def manifest(theta: Dict[str, Any], rho: Resolved, stats: RunStats,
             runtime: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Build the run's audit trail.

    `runtime` holds facts about the machine that ran it: which weights,
    which FASTA, which device. They are kept out of `config` so that
    they never enter the cache key, and recorded so that a score can be
    traced back to the checkpoint that produced it.
    """
    return {
        "runtime": runtime or {},
        "config": {k: (list(v) if isinstance(v, tuple) else v)
                   for k, v in theta.items() if not k.startswith("_")},
        "name": theta.get("_name"), "protocol": theta.get("_protocol"),
        # `config` above is the RESOLVED state; this is how it was
        # reached. Both are needed, because the resolved state alone
        # cannot say that a run was a one-flag ablation off a named
        # config.
        "overrides": list(theta.get("_overrides", ())),
        "resolved": {"model": rho.meta.name, "seq_len": rho.meta.seq_len,
                     "bin_size": rho.meta.bin_size, "tracks": list(rho.tracks),
                     "transform_generation": rho.meta.transform_generation,
                     "scale": rho.scale, "exponent": rho.exponent,
                     "clip_soft": rho.clip_soft, "log_base": rho.meta.log_base,
                     "strand_mask_on": rho.strand_mask_on},
        "warnings": rho.warnings,
        "stats": {"gpu_calls": stats.gpu_calls, "cache_hits": stats.cache_hits,
                  "pairs_scored": stats.pairs_scored,
                  "pairs_dropped": stats.pairs_dropped},
    }


SYNTH_SEQ_LEN = 8192


def toy_cohort(n_genes: int = 4, per_gene: int = 3,
               seq_len: int = SYNTH_SEQ_LEN) -> List[Tuple[Enhancer, Gene]]:
    """Build a small in-memory stand-in for a screen.

    Sized so that every pair is reachable under every anchor, and so
    that a 2 kb ISM window around any enhancer fits entirely inside the
    input. Out-of-window behaviour is tested separately, with an
    enhancer placed deliberately far away.
    """
    pairs = []
    for gi in range(n_genes):
        tss = seq_len // 2 + 137 * gi
        strand = "+" if gi % 2 == 0 else "-"
        exons = tuple((tss + 200 + 400 * k, tss + 200 + 400 * k + 200)
                      for k in range(3))
        gene = Gene(gid=f"G{gi}", chrom="chr1", strand=strand,
                    provided_tss=tss, gencode_tss=(tss, tss + 41),
                    exons=exons)
        for ei in range(per_gene):
            mid = tss + (-1) ** ei * (400 + 600 * ei)
            pairs.append((Enhancer("chr1", mid - 250, mid + 250,
                                   f"G{gi}_E{ei}"), gene))
    return pairs


def _synthetic_run(pairs: List[Tuple[Enhancer, Gene]],
                   theta: Dict[str, Any]
                   ) -> Tuple[List[Dict[str, Any]], RunStats]:
    """Score `pairs` under `theta` on a fresh stand-in backend."""
    return run_experiment(pairs, theta, make_backend(theta, "synthetic"))


#: The self-test's assertion sink, `check(label, condition, detail)`.
_Check = Callable[..., None]


def self_test() -> int:
    """Exercise every method, reduction and combiner without weights.

    Each call below is one named group of checks; `check` counts them
    and prints only the failures.
    """
    ok, fail = 0, 0

    def check(label: str, cond: bool, detail: str = "") -> None:
        nonlocal ok, fail
        if cond:
            ok += 1
        else:
            fail += 1
            print(f"  FAIL  {label}  {detail}")

    pairs = toy_cohort()

    base = fill_defaults({
        "model": "enformer", "screen": "gasperini2019", "gencode": "v32",
        "anchor": "single-tss", "assay": "cage", "cell_type": "k562",
        "region": "tss", "tss_width": 384, "track_agg": "sum",
        "log_scalar": False, "inverse_transform": False,
        "method": "gradient", "mean_subtract": False,
        "abs_map": True, "normalise": "none", "enh_agg": "sum",
        "enh_span": 2000,
    })
    # The ISM fixture the track and lever checks vary one field of.
    # `track_agg` is "mean" rather than `base`'s "sum" because
    # `track_abs_order=before` is defined only for the mean and the
    # validator enforces that, so the fixture has to opt in.
    ism_base = dict(base, method="ism", enh_agg="point", enh_span=None,
                    normalise="none", track_agg="mean", ism_shuffles=4,
                    ism_mode="random-replace",
                    ism_statistic="abs_mean_diff")

    _test_methods(check, pairs, base)
    _test_scale_invariance(check, pairs, base)
    _test_bin_abs_order(check, pairs, base)
    _test_pair_filters(check, pairs, base)
    _test_ism_statistics(check, pairs, base)
    _test_cage_track_sets(check, ism_base)
    _test_ism_levers(check, pairs, base, ism_base)
    _test_overrides(check, base, ism_base)
    _test_anchor_provenance(check, base)
    _test_unreachable_gene(check, pairs, base)
    _test_validator(check, base)
    _test_out_of_window(check, pairs, base)
    _test_free_grid(check, pairs, base)
    _test_free_replicate_order(check, pairs, base)
    _test_receptive_field(check, pairs, base)
    _test_output_bin_geometry(check)
    _test_bin_overlap(check, pairs, base)
    _test_ism_edge_clipping(check, pairs, base)
    _test_perturb(check)
    _test_definition_matches_driver(check, pairs, base)
    _test_post_map_cache(check, base)
    _test_frame_guard(check, pairs, base)
    _test_abs_order_vacuity(check, pairs, base)
    _test_shipped_configs(check)
    _test_backend_capabilities(check, base)

    print(f"\nself-test: {ok} passed, {fail} failed")
    return 0 if fail == 0 else 1


def _test_methods(check: _Check,
                  pairs: List[Tuple[Enhancer, Gene]],
                  base: Dict[str, Any]) -> None:
    """Every method and reduction scores every pair, finitely.

    Also pins the cache: one backward pass per gene, not per pair.
    """
    for method, extra in (("gradient", {}),
                          ("attention", {}),
                          ("ism", {"enh_agg": "point", "enh_span": None,
                                   "ism_shuffles": 1,
                                   "ism_mode": "random-replace",
                                   "ism_statistic": "abs_mean_diff"})):
        theta = dict(base, method=method, **extra)
        rows, stats = _synthetic_run(pairs, theta)
        check(f"{method}: all pairs scored", len(rows) == len(pairs),
              f"{len(rows)} of {len(pairs)}")
        check(f"{method}: scores finite",
              all(math.isfinite(r["score"]) for r in rows))
        check(f"{method}: scores non-negative",
              all(r["score"] >= 0 for r in rows))
        check(f"{method}: gpu_calls > 0", stats.gpu_calls > 0)

    # The cache serves every enhancer of a gene from one pass.
    theta = dict(base)
    rows, stats = _synthetic_run(pairs, theta)
    n_genes = len({g.gid for _, g in pairs})
    check("gradient: one backward pass per gene, not per pair",
          stats.gpu_calls == n_genes,
          f"{stats.gpu_calls} calls for {n_genes} genes, "
          f"{len(pairs)} pairs")

    # Every enhancer-window reduction runs and stays finite.
    for agg, extra in (("sum", {}), ("mean", {}), ("max", {}),
                       ("gaussian", {"gaussian_sigma": 300,
                                     "gaussian_truncate": 4})):
        theta = dict(base, enh_agg=agg, **extra)
        if agg == "gaussian":
            theta["enh_span"] = None
        rows, _ = _synthetic_run(pairs, theta)
        check(f"enh_agg={agg}", len(rows) == len(pairs)
              and all(math.isfinite(r["score"]) for r in rows))

    # A wide Gaussian approaches the global mean ratio.
    theta = dict(base, enh_agg="gaussian", enh_span=None,
                 gaussian_sigma=375, gaussian_truncate=4,
                 normalise="global")
    rows_g, _ = _synthetic_run(pairs, theta)
    check("normalise=global yields scale-free scores",
          all(0 < r["score"] < 1e3 for r in rows_g))


def _test_scale_invariance(check: _Check,
                           pairs: List[Tuple[Enhancer, Gene]],
                           base: Dict[str, Any]) -> None:
    """`normalise=global` divides out a positive rescaling of the map.

    That is what makes `log_scalar` inert for a gradient:
    d log(C+S)/dx = (1/(C+S)) dS/dx differs from dS/dx only by a
    per-gene positive constant. The exact statement is therefore about
    the reducer, so it is tested on `reduce_pair` directly and at
    machine precision, and only then end to end.
    """
    rng = np.random.RandomState(7)
    M = rng.randn(SYNTH_SEQ_LEN)
    e0, g0 = pairs[0]
    w0 = window_at(g0.provided_tss, SYNTH_SEQ_LEN)
    for agg, extra in (("sum", {"enh_span": 2000}),
                       ("mean", {"enh_span": 2000}),
                       ("gaussian", {"enh_span": None,
                                     "gaussian_sigma": 300,
                                     "gaussian_truncate": 4})):
        theta = dict(base, enh_agg=agg, normalise="global", **extra)
        s1 = reduce_pair(M, e0, w0, theta, 1)
        s2 = reduce_pair(1234.5 * M, e0, w0, theta, 1)
        check(f"normalise=global is scale-invariant ({agg})",
              abs(s1 - s2) < 1e-12, f"{s1!r} vs {s2!r}")
    theta = dict(base, enh_agg="sum", enh_span=2000, normalise="none")
    check("normalise=none is NOT scale-invariant",
          abs(reduce_pair(M, e0, w0, theta, 1)
              - reduce_pair(2.0 * M, e0, w0, theta, 1)) > 1e-9)

    # End to end the claim holds only to the accuracy of the stand-in
    # backend's finite-difference Jacobian, so the tolerance is loose
    # and the residual is reported rather than swallowed.
    raw = dict(base, log_scalar=False, normalise="global",
               inverse_transform=False)
    logged = dict(raw, log_scalar=True)
    rows_raw, _ = _synthetic_run(pairs, raw)
    rows_logged, _ = _synthetic_run(pairs, logged)
    worst = max(abs(x["score"] - y["score"])
                for x, y in zip(rows_raw, rows_logged))
    check("log is inert end-to-end for gradients under normalise=global",
          worst < 1e-4,
          f"max |delta| = {worst:.3e} (finite-difference limited)")
    print(f"  note: end-to-end log-vs-raw residual = {worst:.3e} "
          f"(central differences at h=1e-4; the identity is exact)")

    # mean_subtract, by contrast, must move the score.
    subtracted = dict(base, mean_subtract=True)
    rows_subtracted, _ = _synthetic_run(pairs, subtracted)
    check("mean_subtract changes the score",
          any(abs(x["score"] - y["score"]) > 1e-12
              for x, y in zip(rows_raw, rows_subtracted)))


def _test_bin_abs_order(check: _Check,
                        pairs: List[Tuple[Enhancer, Gene]],
                        base: Dict[str, Any]) -> None:
    """|.| before or after the readout-bin sum.

    The two orders differ only where a nucleotide pushes the readout
    bins in OPPOSITE directions: per position sum_b |g_b| >=
    |sum_b g_b|, with equality whenever the per-bin gradients share a
    sign. The stand-in's response kernel is positive everywhere, so its
    per-bin gradients are always sign-aligned and the two orders agree
    exactly. That makes it the wrong probe for the numerical difference
    and the right probe for the plumbing, so the plumbing is what is
    tested: `before` takes one gradient per readout bin and `after`
    exactly one. A live difference needs a real backend.
    """
    # Count by patching the instance's bound method rather than
    # wrapping the object: `one_hot` dispatches on
    # `isinstance(backend, SyntheticBackend)`, so a delegating wrapper
    # loses the pseudo-sequence and the run dies asking for a FASTA.
    cfg = dict(base, tss_width=384)          # 384 / 128 = 3 readout bins
    scores = {}
    for order, expect in (("after", 1), ("before", 3)):
        backend = make_backend(cfg, "synthetic")
        seen: List[Tuple[int, ...]] = []
        inner = backend.scalar_gradient

        def counted(x, r, scalar, _inner=inner, _seen=seen):
            _seen.append(tuple(getattr(scalar, "bins", ())))
            return _inner(x, r, scalar)

        backend.scalar_gradient = counted     # type: ignore[method-assign]
        rows, _ = run_experiment(pairs, dict(cfg, bin_abs_order=order),
                                 backend)
        scores[order] = [r["score"] for r in rows]
        n_genes = len({r["gene"] for r in rows})
        check(f"bin_abs_order={order} takes {expect} gradient(s) per gene",
              len(seen) == expect * n_genes,
              f"{len(seen)} calls over {n_genes} genes")
        check(f"bin_abs_order={order} reads {expect} distinct bin set(s)",
              len(set(seen)) == expect, f"saw {sorted(set(seen))}")
    rel = max(abs(a - b) / max(abs(a), 1e-30)
              for a, b in zip(scores["after"], scores["before"]))
    # The tolerance is finite-difference limited rather than algebraic:
    # the stand-in differentiates by central differences at h = 1e-4,
    # and `before` accumulates three of those where `after` takes one,
    # so the two agree to about 1e-8 rather than to machine epsilon.
    check("the two orders agree on the synthetic backend, whose "
          "per-bin gradients share a sign",
          rel < 1e-6, f"worst relative difference {rel:.3e}")
    # Reported on success as well as on failure. `bin_abs_order` is a
    # pre-registered parameter, so how much of this agreement is
    # algebraic and how much is finite-difference limited is exactly
    # what a reader needs to see.
    print(f"  note: bin_abs_order before-vs-after residual = {rel:.3e} "
          f"(central differences at h=1e-4; `before` accumulates three "
          f"where `after` takes one)")


def _test_pair_filters(check: _Check,
                       pairs: List[Tuple[Enhancer, Gene]],
                       base: Dict[str, Any]) -> None:
    """`pair_filters.min_tss_distance` cuts the table where it says.

    The filter exists to test ONE distance bin at a fraction of the
    cost, so a silent no-op wastes the saving and a silent over-cut
    corrupts the auPRC. Toy distances are 400, 1000 and 1600 bp, which
    pins all three behaviours: the bound is inclusive, it composes with
    `max_tss_distance` into a band, and it is measured against
    `provided_tss`.
    """
    def dists(sel):
        table = build_pair_table(pairs, dict(base, pair_filters=sel))
        return sorted(abs(e.mid() - g.provided_tss) for e, g in table)

    band = {"min_tss_distance": 1000, "max_tss_distance": 1000}
    check("min_tss_distance keeps only pairs at or beyond the bound",
          dists({"min_tss_distance": 1000})
          == [1000, 1000, 1000, 1000, 1600, 1600, 1600, 1600],
          f"got {dists({'min_tss_distance': 1000})}")
    check("min_tss_distance composes with max_tss_distance into a "
          "closed band",
          dists(band) == [1000] * 4, f"got {dists(band)}")
    check("min_tss_distance above every pair empties the table",
          dists({"min_tss_distance": 10_000}) == [])
    check("no filter keeps everything", len(dists({})) == len(pairs))


def _test_ism_statistics(check: _Check,
                         pairs: List[Tuple[Enhancer, Gene]],
                         base: Dict[str, Any]) -> None:
    """The four ISM statistics are reachable, finite and distinct."""
    seen = {}
    for stat in ("abs_mean_diff", "abs_diff", "abs_mean_rel_diff",
                 "rel_abs_diff"):
        theta = dict(base, method="ism", enh_agg="point", enh_span=None,
                     ism_shuffles=10, ism_mode="dinuc-shuffle",
                     ism_statistic=stat, log_scalar=False,
                     normalise="none")
        rows, _ = _synthetic_run(pairs, theta)
        seen[stat] = [r["score"] for r in rows]
        check(f"ism_statistic={stat}", len(rows) == len(pairs)
              and all(math.isfinite(v) and v >= 0 for v in seen[stat]))
    check("abs_mean_diff <= abs_diff pointwise (Jensen)",
          all(m <= d + 1e-12
              for m, d in zip(seen["abs_mean_diff"], seen["abs_diff"])))

    # A Borzoi RNA-seq arm, where the inner mean has axes to take.
    strand_only = fill_defaults({
        "model": "borzoi", "screen": "gasperini2019", "gencode": "v41",
        "anchor": "gene-midpoint", "assay": "rnaseq", "cell_type": "k562",
        "region": "exons", "tss_width": None,
        "track_agg": "mean",
        "inverse_transform": True, "log_scalar": True,
        "method": "ism", "enh_agg": "point", "ism_shuffles": 16,
        "ism_mode": "dinuc-shuffle", "ism_statistic": "abs_mean_diff",
        "normalise": "none", "replicates": 4, "tta": "rc",
        "ism_inner_mean": ["strand"]})
    both = dict(strand_only, ism_inner_mean=("strand", "replicate"))
    rows_strand, _ = _synthetic_run(pairs, strand_only)
    rows_both, _ = _synthetic_run(pairs, both)
    check("ism_inner_mean={strand} vs {strand,replicate} differ",
          any(abs(x["score"] - y["score"]) > 1e-12
              for x, y in zip(rows_strand, rows_both)))
    check("Borzoi ISM arm scores every pair",
          len(rows_strand) == len(pairs))


def _test_cage_track_sets(check: _Check, ism_base: Dict[str, Any]) -> None:
    """Each CAGE cell-type key resolves to the tracks it names.

    Three claims per model, each of which a plausible wiring bug would
    break:

      1. The strand mask still bites. GENE-SCALAR keeps a track whose
         strand is `None`, so declaring 1,276 tracks and forgetting
         their strands makes `strand_mask=auto` silently inert: the arm
         would average 1,276 tracks while its name says 638, and no
         score-table column would show it. Asserted as the read count
         per strand, not as "the map is populated".
      2. The ten draws are Enformer's ten draws. The two target files
         are index-aligned by biological sample, so ordinal k names
         Enformer 4675+k and Borzoi (2k, 2k+1), the same FANTOM sample.
         If that alignment stops holding, the two bands stop being
         paired and the cross-model comparison quietly becomes a
         comparison of different cell types.
      3. ("cage","k562") still resolves to its own tracks, or every
         reported CAGE number refers to a different readout.
    """
    # The singleton keys isolate exactly one FANTOM library each.
    for ct, want in (("k562", (4828, 5111)),
                     ("k562-cnhs11250", (4828,)),
                     ("k562-encode", (5111,))):
        rho = resolve(dict(ism_base, cell_type=ct), ENFORMER_META)
        got = tuple(rho.tracks)
        check("cell_type=%s resolves to %s" % (ct, want), got == want,
              "got %s" % (got,))

    borzoi = dict(ism_base, model="borzoi")
    for ct, n_declared, n_read in (("k562", 4, 2), ("all", 1276, 638),
                                   ("not-k562", 1272, 636),
                                   ("rand2-s0", 4, 2)):
        rho = resolve(dict(borzoi, cell_type=ct), BORZOI_META)
        check("borzoi cell_type=%s declares %d tracks" % (ct, n_declared),
              len(rho.tracks) == n_declared, "got %d" % len(rho.tracks))
        check("borzoi cell_type=%s: strand mask is ON" % ct,
              rho.strand_mask_on)
        check("borzoi cell_type=%s: every track has a declared strand" % ct,
              all(st is not None for st in rho.track_strands))
        for gs in ("+", "-"):
            kept = sum(1 for st in rho.track_strands if st in (None, gs))
            check("borzoi cell_type=%s: a '%s' gene reads %d tracks"
                  % (ct, gs, n_read), kept == n_read, "got %d" % kept)
    k562_borzoi = resolve(dict(borzoi, cell_type="k562"), BORZOI_META)
    check("borzoi ('cage','k562') still resolves to its own four tracks",
          tuple(k562_borzoi.tracks) == (306, 307, 872, 873))
    for i in range(10):
        name = "rand2-s%d" % i
        ords = BORZOI_CAGE_RAND2_ORDINALS[name]
        enformer_draw = ENFORMER_META.tracks[("cage", name)]
        check("%s: Borzoi draw is Enformer's draw, same FANTOM samples"
              % name, tuple(4675 + k for k in ords) == enformer_draw,
              "ordinals %s -> %s, Enformer has %s"
              % (ords, tuple(4675 + k for k in ords), enformer_draw))
        check("%s: expands to both strands of both samples" % name,
              BORZOI_META.tracks[("cage", name)]
              == tuple(sorted([2 * k for k in ords]
                              + [2 * k + 1 for k in ords])))
        check("%s: draws no K562 CAGE track" % name,
              not set(BORZOI_META.tracks[("cage", name)])
              & {306, 307, 872, 873})
    check("the ten Borzoi draws cover 20 distinct samples",
          len({t // 2 for i in range(10)
               for t in BORZOI_META.tracks[("cage", "rand2-s%d" % i)]})
          == 20)
    # The (even, odd) = (+, -) rule the whole block rests on, stated
    # where it can fail loudly.
    check("Borzoi CAGE strand map follows (even, odd) = (+, -) for "
          "all 1,276",
          all(BORZOI_META.track_strand[t] == ("+" if t % 2 == 0 else "-")
              for t in BORZOI_CAGE_BLOCK))
    check("Borzoi CAGE block is 638 samples",
          len(BORZOI_CAGE_BLOCK) == 1276)
    check("not-k562 is the block minus K562's own two samples",
          set(BORZOI_META.tracks[("cage", "not-k562")])
          == set(BORZOI_CAGE_BLOCK) - {306, 307, 872, 873})

    # AlphaGenome carries the same three hazards plus two of its own:
    #
    #   4. The expansion rule is (k, k+273), not Borzoi's (2k, 2k+1).
    #      Both are "the pair of strands for sample k" and they
    #      disagree everywhere except k=0. Borzoi's rule here would
    #      pair sample k's '+' with a DIFFERENT sample's '-', and
    #      because both indices are real tracks nothing would raise.
    #   5. The head is 640 channels but only 546 are real. The other 94
    #      are organism padding shared with the mouse head: emitted,
    #      and meaningless. A raw `range(640)` slice would average junk
    #      into the agnostic arm and look entirely normal.
    alphagenome = dict(ism_base, model="alphagenome", assay="cage")
    for ct, n_declared, n_read in (("k562", 2, 1), ("all", 546, 273),
                                   ("not-k562", 544, 272),
                                   ("rand1-s0", 2, 1)):
        rho = resolve(dict(alphagenome, cell_type=ct), ALPHAGENOME_META)
        check("alphagenome cell_type=%s declares %d tracks"
              % (ct, n_declared),
              len(rho.tracks) == n_declared, "got %d" % len(rho.tracks))
        check("alphagenome cell_type=%s: strand mask is ON" % ct,
              rho.strand_mask_on)
        check("alphagenome cell_type=%s: every track has a declared "
              "strand" % ct,
              all(st is not None for st in rho.track_strands))
        for gs in ("+", "-"):
            kept = sum(1 for st in rho.track_strands if st in (None, gs))
            check("alphagenome cell_type=%s: a '%s' gene reads %d tracks"
                  % (ct, gs, n_read), kept == n_read, "got %d" % kept)
    k562_alphagenome = _alphagenome_cage_tracks(
        [ALPHAGENOME_CAGE_K562_ORDINAL])
    check("alphagenome ('cage','k562') still resolves to its own two "
          "tracks", k562_alphagenome == (150, 423),
          "got %s" % (k562_alphagenome,))
    for name, k in ALPHAGENOME_CAGE_RAND1_ORDINALS.items():
        check("%s: expands by the CONSTANT OFFSET rule (k, k+273)" % name,
              _alphagenome_cage_tracks([k]) == (k, k + 273),
              "ordinal %d -> %s" % (k, _alphagenome_cage_tracks([k])))
        check("%s: draws no K562 CAGE track" % name,
              not set(_alphagenome_cage_tracks([k])) & {150, 423})
    check("the ten AlphaGenome draws are ten DISTINCT samples",
          len(set(ALPHAGENOME_CAGE_RAND1_ORDINALS.values())) == 10,
          "ten size-1 draws from 272 collide with ~15% probability, so "
          "this is checked rather than implied")
    every_set = ([range(ALPHAGENOME_CAGE_N_SAMPLES),
                  ALPHAGENOME_CAGE_NOT_K562_ORDINALS]
                 + [[v] for v
                    in ALPHAGENOME_CAGE_RAND1_ORDINALS.values()])
    check("no CAGE set leaks a padding channel (546-639 are not real)",
          all(i <= 545 for ords in every_set
              for i in _alphagenome_cage_tracks(ords)),
          "AlphaGenome's CAGE head is 640 wide but only 0-545 carry a "
          "strand")
    check("alphagenome not-k562 is the 273 samples minus K562's own one",
          set(ALPHAGENOME_CAGE_NOT_K562_ORDINALS)
          == set(range(ALPHAGENOME_CAGE_N_SAMPLES))
          - {ALPHAGENOME_CAGE_K562_ORDINAL})
    check("AlphaGenome draws are NOT Borzoi's: the panels differ in size",
          ALPHAGENOME_CAGE_N_SAMPLES != 638,
          "273 vs 638; only 4 of the 20 samples the other two trunks "
          "drew exist in AlphaGenome's panel at all, so the three bands "
          "are independent and a figure must say so")


def _test_ism_levers(check: _Check,
                     pairs: List[Tuple[Enhancer, Gene]],
                     base: Dict[str, Any],
                     ism_base: Dict[str, Any]) -> None:
    """The ISM parameters that decide what is perturbed and how."""
    # ism_window. The perturbation span is placed from the element
    # midpoint, and that midpoint floors for an even-width element, so
    # under reverse complement the floor lands on the other side and
    # the span shifts by 1 bp. `tta` is "none" here, so that never
    # bites. A wider window must actually widen the lesion.
    narrow = dict(ism_base, ism_window=500)
    wide = dict(ism_base, ism_window=2000)
    s_narrow = [r["score"] for r in _synthetic_run(pairs, narrow)[0]]
    s_wide = [r["score"] for r
              in run_experiment(pairs, wide,
                                make_backend(ism_base, "synthetic"))[0]]
    check("a 2000 bp lesion differs from a 500 bp one",
          any(abs(x - y) > 1e-12 for x, y in zip(s_narrow, s_wide)))

    # track_abs_order. Two algebraic facts pin the implementation. For
    # ONE track the two orders are the same expression, so they must
    # agree to machine precision, which catches a per-track path that
    # reduces over the wrong axis. For many tracks
    # mean_T |y_T| >= |mean_T y_T| by the triangle inequality, with no
    # assumption about the model, so `before` can never score below
    # `after`; a reversed or misplaced |.| breaks that on some pair.
    single = dict(ism_base, cell_type="k562-cnhs11250")
    after_one = [r["score"] for r
                 in run_experiment(pairs,
                                   dict(single, track_abs_order="after"),
                                   make_backend(single, "synthetic"))[0]]
    before_one = [r["score"] for r
                  in run_experiment(pairs,
                                    dict(single,
                                         track_abs_order="before"),
                                    make_backend(single, "synthetic"))[0]]
    worst = max(abs(x - y) for x, y in zip(after_one, before_one))
    check("track_abs_order is inert on a single track",
          worst < 1e-12, "worst |delta| = %.3e" % worst)
    after_all = [r["score"] for r
                 in run_experiment(pairs,
                                   dict(ism_base, cell_type="all",
                                        track_abs_order="after"),
                                   make_backend(ism_base,
                                                "synthetic"))[0]]
    before_all = [r["score"] for r
                  in run_experiment(pairs,
                                    dict(ism_base, cell_type="all",
                                         track_abs_order="before"),
                                    make_backend(ism_base,
                                                 "synthetic"))[0]]
    slack = min(y - x for x, y in zip(after_all, before_all))
    check("track_abs_order=before never scores below after "
          "(triangle inequality)",
          slack > -1e-12, "worst shortfall %.3e" % slack)

    # track_agg must reach the per-track path. The readout this
    # reproduces sums 638 tracks (sum_t |df_t|), so a track axis
    # collapsed by an implicit mean would be 638 times too small, and
    # that factor cancels in auPRC and would never be noticed.
    meta = make_backend(ism_base, "synthetic").meta
    n_tracks = len(resolve(dict(ism_base, cell_type="all"), meta).tracks)
    before_sum = [r["score"] for r
                  in run_experiment(pairs,
                                    dict(ism_base, cell_type="all",
                                         track_abs_order="before",
                                         track_agg="sum"),
                                    make_backend(ism_base,
                                                 "synthetic"))[0]]
    worst = max(abs(s - n_tracks * m)
                for s, m in zip(before_sum, before_all) if abs(m) > 1e-12)
    check("track_agg=sum on the per-track path is exactly %d x the mean"
          % n_tracks, worst < 1e-9,
          "worst |delta| = %.3e over %d tracks" % (worst, n_tracks))

    # The draws NEST, which is what makes the ism_shuffles ladder free:
    # stable_seed keys on the pair and never on the draw index, so
    # P=10's first four draws ARE P=4's, and a P=4 run is a prefix of a
    # P=10 run rather than an independent experiment.
    backend = make_backend(ism_base, "synthetic")
    rho = resolve(ism_base, backend.meta)
    enh, gene = pairs[0]
    window = anchor_windows(None, gene, ism_base, rho, backend)[0]
    fwd = Aug(shift=0, rc=False)
    draws4 = ism_map(gene, [enh], window, 1, fwd,
                     dict(ism_base, ism_shuffles=4), rho,
                     backend)[0][enh.eid]
    draws10 = ism_map(gene, [enh], window, 1, fwd,
                      dict(ism_base, ism_shuffles=10), rho,
                      backend)[0][enh.eid]
    check("ism_shuffles draws nest: P=4 is the prefix of P=10",
          len(draws4) == 4 and len(draws10) == 10
          and np.allclose(draws10[:4], draws4, rtol=0, atol=0),
          "P=4 %s vs P=10[:4] %s"
          % (np.round(draws4, 6), np.round(draws10[:4], 6)))

    # tss_width=128 is the single-bin readout. It must be exactly the
    # TSS bin, and unlike every even width it must NOT depend on
    # strand: the downstream tie-break applies only when k is even and
    # there is no middle bin to land on.
    backend = make_backend(base, "synthetic")
    rho = resolve(base, backend.meta)
    for gene in {g.gid: g for _, g in pairs}.values():
        window = anchor_windows(None, gene, base, rho, backend)[0]
        one_bin = readout_bins(gene, window, dict(base, tss_width=128),
                               backend.meta)
        three_bin = readout_bins(gene, window, dict(base, tss_width=384),
                                 backend.meta)
        check("tss_width=128 reads exactly the TSS bin (%s, %s)"
              % (gene.gid, gene.strand),
              len(one_bin) == 1
              and one_bin[0] == bin_of(window.centre, window,
                                       backend.meta),
              "got %s" % (one_bin,))
        check("the 1-bin readout is the middle of the 3-bin readout (%s)"
              % gene.gid, three_bin[1] == one_bin[0],
              "%s vs %s" % (three_bin, one_bin))


def _test_overrides(check: _Check,
                    base: Dict[str, Any],
                    ism_base: Dict[str, Any]) -> None:
    """`--set` is indistinguishable from writing the value in the config.

    So the test is an equality against the config form, plus one case
    per kind of value. A string that survived coercion untyped would
    fail the domain check later, and the error would then point at the
    value rather than at the parsing.
    """
    flags = ["ism_shuffles=4", "tss_major=true", "cell_type=all",
             "ism_inner_mean=strand,replicate",
             'pair_filters={"min_tss_distance": 1000}',
             "tss_width=null"]
    ov = apply_overrides({"model": "enformer"}, flags)
    check("--set coerces int",
          ov["ism_shuffles"] == 4 and isinstance(ov["ism_shuffles"], int))
    check("--set coerces bool", ov["tss_major"] is True)
    # No float-coercion check: no parameter currently declares
    # kind="float". Add one here if a float parameter is introduced.
    check("--set coerces a comma set",
          ov["ism_inner_mean"] == ("strand", "replicate"))
    check("--set coerces a JSON dict",
          ov["pair_filters"] == {"min_tss_distance": 1000})
    check("--set coerces null", ov["tss_width"] is None)
    named = apply_overrides({"model": "enformer"},
                            ["normalise=none", "tta=none"])
    check("--set keeps `none` where the parameter declares it as a value",
          named["normalise"] == "none" and named["tta"] == "none")
    check("--set records itself for the manifest",
          ov.get("_overrides") == flags)
    yaml_form = fill_defaults(dict(ism_base, ism_mode="genomic-replace"))
    set_form = fill_defaults(
        apply_overrides(dict(ism_base), ["ism_mode=genomic-replace"]))
    set_form.pop("_overrides", None)
    check("--set is equivalent to writing the value in the config",
          set_form == yaml_form)
    for bad_set in (["nonesuch=1"], ["ism_shuffles"], ["tss_major=maybe"]):
        try:
            apply_overrides({}, bad_set)
            check("--set rejects %s" % (bad_set,), False, "accepted it")
        except ConfigError:
            check("--set rejects %s" % (bad_set,), True)
    # An override reaches the SAME validator, so it cannot arrive at a
    # state a config file could not.
    try:
        validate(fill_defaults(
            apply_overrides(dict(base), ["track_abs_order=before"])))
        check("--set is still validated", False,
              "gradient + track_abs_order=before was allowed")
    except ConfigError:
        check("--set is still validated", True)


def _test_anchor_provenance(check: _Check, base: Dict[str, Any]) -> None:
    """An anchor table is checked against the ANCHOR's cell type.

    A table is built with one readout cell type and consumed by a run
    with another, so `validate_anchors` must compare against
    `tss_major_cell_type` and not `cell_type`. Compare against
    `cell_type` and every cell-type-agnostic run is rejected against
    the K562 anchors that are exactly what it should use.
    """
    anchored = dict(base, tss_major=True, tss_major_assay="cage")
    check("major_tss_cell_type follows cell_type when unset",
          major_tss_cell_type(dict(anchored, cell_type="all")) == "all")
    check("major_tss_cell_type overrides when set",
          major_tss_cell_type(dict(anchored, cell_type="all",
                                   tss_major_cell_type="k562")) == "k562")

    prov = {"model": "eleutherai/enformer-official-rough",
            "gencode": "v32", "tss_major_assay": "cage",
            "cell_type": "k562", "screen": "gasperini",
            "use_tf_gamma": "true"}
    agnostic = dict(anchored, cell_type="all", tss_major_cell_type="k562",
                    screen="gasperini2019")
    label = "K562 anchors are ACCEPTED for a cell-type-agnostic run"
    try:
        warns = validate_anchors(dict(prov), agnostic, "x.tsv")
        check(label, not warns, "warned: %s" % warns)
    except ConfigError as exc:
        check(label, False, str(exc))
    try:
        validate_anchors(dict(prov, cell_type="hepg2"), agnostic,
                         "x.tsv")
        check("a genuinely wrong anchor cell type still raises", False,
              "accepted it")
    except ConfigError:
        check("a genuinely wrong anchor cell type still raises", True)
    # An ABSENT field is unknown, not matching: silence there would
    # hide a genuine provenance mismatch behind a clean run.
    missing = {k: v for k, v in prov.items() if k != "gencode"}
    warns = validate_anchors(missing, agnostic, "x.tsv")
    check("a missing provenance field warns rather than passing "
          "silently",
          any("records no gencode" in w for w in warns),
          "got %s" % warns)
    # use_tf_gamma has three states, not two: recorded true, recorded
    # false, and never recorded. The third has to warn like the second.
    # It is fixed at True on the theta side, so only the anchor table's
    # value varies, and it is the half that warns.
    for val, want in (("true", False), ("false", True),
                      ("unrecorded", True), ("", True)):
        warns = validate_anchors(dict(prov, use_tf_gamma=val), agnostic,
                                 "x.tsv")
        hit = any("use_tf_gamma" in w for w in warns)
        check("use_tf_gamma=%r %s"
              % (val, "warns" if want else "is silent"),
              hit == want, "warnings: %s" % warns)
    try:
        validate(fill_defaults(dict(base, tss_major=False,
                                    tss_major_cell_type="k562")))
        check("tss_major_cell_type without tss_major is refused", False,
              "accepted it")
    except ConfigError:
        check("tss_major_cell_type without tss_major is refused", True)


def _test_unreachable_gene(check: _Check,
                           pairs: List[Tuple[Enhancer, Gene]],
                           base: Dict[str, Any]) -> None:
    """A gene no window reaches, under skipping, sharding and topping up.

    The cohort is the toy one plus a gene whose only enhancers sit
    10 Mb away, which is the case the skip exists for: no window can
    contain them, every arm returns OUT-OF-WINDOW, and the model call
    that produced the map was spent on nothing. Each claim below is
    exact rather than approximate, because these are optimisations on
    the hot path of every arm and a score that moved by 1e-15 would
    still be a correctness bug.
    """
    far_gene = Gene(gid="FARGENE", chrom="chr1", strand="+",
                    provided_tss=SYNTH_SEQ_LEN // 2,
                    gencode_tss=(SYNTH_SEQ_LEN // 2,),
                    exons=((SYNTH_SEQ_LEN // 2,
                            SYNTH_SEQ_LEN // 2 + 200),))
    far_pairs = list(pairs) + [
        (Enhancer("chr1", 10 ** 7, 10 ** 7 + 500, "FAR1"), far_gene),
        (Enhancer("chr1", 10 ** 7 + 900, 10 ** 7 + 1400, "FAR2"),
         far_gene)]
    ism_cfg = dict(base, method="ism", enh_agg="point", enh_span=None,
                   normalise="none", ism_shuffles=1,
                   ism_mode="random-replace",
                   ism_statistic="abs_mean_diff")
    n_far_genes = len({g.gid for _, g in far_pairs})
    for label, cfg in (("gradient", dict(base)),
                       ("attention", dict(base, method="attention")),
                       ("ism", ism_cfg)):
        rows_far, stats_far = run_experiment(
            far_pairs, cfg, make_backend(cfg, "synthetic"))
        seen_genes = {r["gene"] for r in rows_far}
        check("skip fires on a gene with nothing scoreable (%s)" % label,
              stats_far.gpu_calls > 0
              and len(seen_genes) == n_far_genes,
              "gpu_calls=%d genes=%d"
              % (stats_far.gpu_calls, len(seen_genes)))
        far_rows = [r for r in rows_far if r["gene"] == "FARGENE"]
        check("the skipped gene still gets its rows, at OUT-OF-WINDOW "
              "(%s)" % label,
              len(far_rows) == 2
              and all(r["score"] == 0.0 for r in far_rows),
              "%s" % [(r["enhancer"], r["score"]) for r in far_rows])
        rows_ref, _ = run_experiment(pairs, cfg,
                                     make_backend(cfg, "synthetic"))
        ref = {(r["gene"], r["enhancer"]): r["score"] for r in rows_ref}
        got = {(r["gene"], r["enhancer"]): r["score"] for r in rows_far
               if r["gene"] != "FARGENE"}
        check("adding a skipped gene leaves every other score "
              "BIT-IDENTICAL (%s)" % label, ref == got,
              "differ on %s"
              % [k for k in ref if ref.get(k) != got.get(k)][:3])

    # Sharding then merging. The failure mode here is specifically
    # ORDER rather than value: the paper's bootstrap numbers its
    # resampling units in table order, so a merged table holding all the
    # right numbers in the wrong order moves the published intervals. A
    # value-only check would pass while they drifted, so compare the row
    # SEQUENCE and not a dict keyed by pair.
    global SHARD

    def without_rank(rows):
        return [{k: v for k, v in r.items() if k != "_gene_rank"}
                for r in rows]

    for label, cfg in (("gradient", dict(base)), ("ism", ism_cfg)):
        try:
            SHARD = (0, 1)
            whole, _ = _synthetic_run(far_pairs, cfg)
            for n_shards in (2, 3):
                parts = []
                for i in range(n_shards):
                    SHARD = (i, n_shards)
                    rows, stats = _synthetic_run(far_pairs, cfg)
                    check("shard %d/%d records its identity (%s)"
                          % (i, n_shards, label),
                          stats.shard == "%d/%d" % (i, n_shards),
                          "got %r" % stats.shard)
                    parts.extend(rows)
                # What a shard-combining step does: a stable sort on
                # the recorded rank.
                parts.sort(key=lambda r: r["_gene_rank"])
                check("%d shards merge BIT-IDENTICALLY to one run, in "
                      "order (%s)" % (n_shards, label),
                      without_rank(parts) == without_rank(whole),
                      "%d vs %d rows" % (len(parts), len(whole)))
                ranks = [r["_gene_rank"] for r in parts]
                check("%d shards partition the genes (%s)"
                      % (n_shards, label),
                      len(set(ranks))
                      == len({r["_gene_rank"] for r in whole}),
                      "%d distinct ranks" % len(set(ranks)))
        finally:
            SHARD = (0, 1)

    # The claim a top-up rests on: a score is a pure function of (gene,
    # enhancer, theta), so scoring one gene alone gives bit-identical
    # rows to scoring it inside the whole cohort. If any cohort-wide
    # quantity ever leaks into a row, whether a normaliser taken over
    # all pairs, a rank, or a running statistic, this is the test that
    # fails, and it fails before a topped-up table can be published.
    global ONLY_GENES
    try:
        whole, _ = _synthetic_run(far_pairs, base)
        present = list(dict.fromkeys(r["gene"] for r in whole))
        for gid in present:
            ONLY_GENES = frozenset({gid})
            part, _ = _synthetic_run(far_pairs, base)
            check("--only-genes %s == that gene's rows in a full run"
                  % gid, part == [r for r in whole if r["gene"] == gid],
                  "%d rows vs %d in the full table"
                  % (len(part), sum(1 for r in whole
                                    if r["gene"] == gid)))
        rebuilt = []
        for gid in present:
            ONLY_GENES = frozenset({gid})
            rows, _ = _synthetic_run(far_pairs, base)
            rebuilt.extend(rows)
        rebuilt.sort(key=lambda r: r["_gene_rank"])
        check("per-gene runs splice back into the full table "
              "BIT-IDENTICALLY", rebuilt == whole,
              "%d vs %d rows" % (len(rebuilt), len(whole)))
        ONLY_GENES = frozenset({"NO_SUCH_GENE"})
        try:
            _synthetic_run(far_pairs, base)
            check("--only-genes with an unknown gene ID is fatal", False,
                  "it returned instead of raising")
        except ConfigError:
            check("--only-genes with an unknown gene ID is fatal", True)
    finally:
        ONLY_GENES = None

    dropping = dict(base, out_of_window="drop")
    rows_dropped, stats_dropped = _synthetic_run(far_pairs, dropping)
    far_kept = sum(1 for r in rows_dropped if r["gene"] == "FARGENE")
    check("out_of_window=drop drops the skipped gene's pairs",
          far_kept == 0 and stats_dropped.pairs_dropped >= 2,
          "dropped=%d, FARGENE rows=%d"
          % (stats_dropped.pairs_dropped, far_kept))


def _test_validator(check: _Check, base: Dict[str, Any]) -> None:
    """The validator refuses what it must, and only that."""
    ism_cfg = dict(base, method="ism", enh_agg="point", enh_span=None)
    bad = [
        ("relative statistic on a logged scalar",
         dict(ism_cfg, ism_statistic="rel_abs_diff", log_scalar=True)),
        # No "attention on model X" case here: all three trunks do
        # attend, so refusing by metadata would encode a false claim
        # about the model. The real guard is a capability check in
        # `check_backend` against a backend that does not IMPLEMENT
        # `attention`, which `_test_backend_capabilities` exercises.
        #
        # unified-t1's readout-scheme pairing. The model is Borzoi
        # deliberately, not `base`'s Enformer: Enformer owns no RNA-seq
        # track set, so on Enformer the rnaseq case is refused several
        # rules earlier for having no track set and the assertion would
        # pass for entirely the wrong reason. Borzoi owns both
        # ("cage","k562") and ("rnaseq","k562"), so both cells here are
        # representable and only the pairing rule can reject them.
        ("cage@exons under unified-t1",
         dict(base, model="borzoi", _protocol="unified-t1",
              assay="cage", region="exons", tss_width=None)),
        ("rnaseq@tss under unified-t1",
         dict(base, model="borzoi", _protocol="unified-t1",
              assay="rnaseq", region="tss", tss_width=384)),
        ("tss_width not a multiple of the bin size",
         dict(base, tss_width=300)),
        ("tss_width set while region=exons",
         dict(base, region="exons", tss_width=384)),
        ("region=tss without tss_width",
         dict(base, tss_width=None)),
        ("gaussian without sigma",
         dict(base, enh_agg="gaussian", enh_span=None)),
        ("enh_span set on a gaussian window",
         dict(base, enh_agg="gaussian", gaussian_sigma=300,
              gaussian_truncate=4)),
        ("sum without enh_span",
         dict(base, enh_agg="sum", enh_span=None)),
        ("anchor_agg without each-tss",
         dict(base, anchor_agg="mean")),
        ("each-tss without anchor_agg",
         dict(base, anchor="each-tss")),
        ("shuffles inside ism_inner_mean",
         dict(ism_cfg, ism_inner_mean=("shuffles",))),
        ("track_abs_order=before on the gradient arm",
         dict(base, track_abs_order="before")),
        ("strand inside ism_inner_mean without rc",
         dict(ism_cfg, ism_inner_mean=("strand",), tta="none")),
        ("ism without point aggregation",
         dict(base, method="ism", enh_agg="sum", enh_span=2000)),
        ("point aggregation without ism",
         dict(base, enh_agg="point", enh_span=None)),
        ("inverse_transform on alphagenome",
         dict(base, model="alphagenome", assay="rnaseq",
              inverse_transform=True)),
        ("tss_major without an assay",
         dict(base, tss_major=True)),
        ("shift TTA without a shift size",
         dict(base, tta="shift")),
        ("bin_overlap=fractional off the attention path",
         dict(base, method="gradient", bin_overlap="fractional")),
    ]
    for label, theta in bad:
        try:
            validate(theta)
            check(f"refuses: {label}", False, "accepted it")
        except ConfigError:
            check(f"refuses: {label}", True)

    # The readout-scheme rule is SCOPED to unified-t1, and refuses for
    # the stated reason. The scoping is what matters: an incoherent
    # readout must stay expressible when no protocol claims it, because
    # a reproduction config has to be free to mirror what a paper
    # actually did. If the rule ever leaks out of unified-t1 it
    # silently removes that freedom, and nothing else here would
    # notice.
    incoherent = dict(base, model="borzoi", assay="rnaseq", region="tss",
                      tss_width=384)
    scoped = "rnaseq@tss is ACCEPTED with no _protocol"
    try:
        validate(incoherent)
        check(scoped, True)
    except ConfigError as e:
        check(scoped, False, str(e))
    for proto in ("avsec2021", "linder2025", "avsec2026"):
        try:
            validate(dict(incoherent, _protocol=proto))
            check(f"the rule does not touch _protocol={proto}", True)
        except ConfigError as e:
            check(f"the rule does not touch _protocol={proto}", False,
                  str(e))
    try:
        validate(dict(incoherent, _protocol="unified-t1"))
        check("the refusal names the readout scheme", False,
              "accepted it")
    except ConfigError as e:
        check("the refusal names the readout scheme",
              "readout scheme" in str(e) and "rnaseq@exons" in str(e),
              str(e))
    for assay, region, width, model in (("cage", "tss", 384, "borzoi"),
                                        ("rnaseq", "exons", None,
                                         "borzoi"),
                                        ("cage", "tss", 384, "enformer")):
        theta_ok = dict(base, model=model, _protocol="unified-t1",
                        assay=assay, region=region, tss_width=width)
        try:
            validate(theta_ok)
            check(f"unified-t1 accepts {assay}@{region} on {model}", True)
        except ConfigError as e:
            check(f"unified-t1 accepts {assay}@{region} on {model}",
                  False, str(e))

    # Unknown config keys are rejected.
    try:
        fill_defaults({"model": "enformer", "sigma": 300})
        check("rejects unknown key", False, "accepted 'sigma'")
    except ConfigError:
        check("rejects unknown key", True)


def _test_out_of_window(check: _Check,
                        pairs: List[Tuple[Enhancer, Gene]],
                        base: Dict[str, Any]) -> None:
    """`out_of_window` and the `in_window` column agree on reachability.

    `pair_in_window` claims that reachability can be read off a row
    without knowing the policy. That holds only if its geometric
    predicate agrees with the one `reduce_pair` applies to
    `M.shape[0]`, which is a different expression of what should be the
    same fact. Rather than assume it, the two are pinned together:
    `drop` keeps exactly the rows `zero` marks reachable. If a map's
    length ever stops equalling its window's length, this fails here
    instead of silently mislabelling coverage in a headline comparison.
    """
    far = list(pairs) + [(Enhancer("chr1", 10 ** 7, 10 ** 7 + 500,
                                   "FAR"), pairs[0][1])]
    kept: Dict[str, List[Dict[str, Any]]] = {}
    for policy, expect in (("zero", len(far)), ("drop", len(pairs))):
        theta = dict(base, out_of_window=policy)
        rows, _ = run_experiment(far, theta,
                                 make_backend(theta, "synthetic"))
        kept[policy] = rows
        check(f"out_of_window={policy}", len(rows) == expect,
              f"{len(rows)} rows, expected {expect}")

    def key(row):
        return (row["gene"], row["enhancer"])

    for policy in ("zero", "drop"):
        missing = sum("in_window" not in r for r in kept[policy])
        check(f"every row carries in_window (out_of_window={policy})",
              missing == 0, f"{missing} rows lack it")
    dropped = {key(r) for r in kept["drop"]}
    reachable = {key(r) for r in kept["zero"] if r["in_window"]}
    check("out_of_window=drop keeps exactly the rows marked in_window",
          dropped == reachable,
          "drop=%s vs reachable=%s"
          % (sorted(dropped), sorted(reachable)))
    check("every surviving row under drop is in_window=True",
          all(r["in_window"] for r in kept["drop"]))
    unreached = [r for r in kept["zero"] if not r["in_window"]]
    check("the deliberately-distal pair is marked unreachable, and "
          "only it",
          [key(r) for r in unreached] == [(pairs[0][1].gid, "FAR")],
          f"{[key(r) for r in unreached]}")
    far_row = unreached[0]
    check("an unreachable pair scores 0.0 AND says so",
          float(far_row["score"]) == 0.0
          and far_row["in_window"] is False,
          f"score={far_row['score']} in_window={far_row['in_window']}")


def _test_free_grid(check: _Check,
                    pairs: List[Tuple[Enhancer, Gene]],
                    base: Dict[str, Any]) -> None:
    """The post-map overlays are free, and they are also correct.

    Scoring the {gaussian,sum} x {global,none} grid from one set of
    maps must give BIT-IDENTICAL numbers to running each cell as its
    own experiment, and must not cost an extra model call. "Close" is
    not the bar: these columns are meant to be usable in a table
    interchangeably with dedicated runs, so any discrepancy at all
    means the overlay is not the same computation.

    The same bar covers `mean_subtract`, whose two variants must come
    out of ONE backward pass. That rests on an algebraic identity, the
    variants being A and A - m for A the reference-base gradient and m
    the per-position four-base mean, and on the variant being formed
    BEFORE the absolute value, since |A - m| != |A| - |m|.
    """
    # This also guards the trap in `build_free_grid`: `gaussian` and
    # `sum` need mutually exclusive window parameters, so a cell that
    # forgets to null the other's leaves a leftover `enh_span` or
    # `gaussian_sigma` in the overlay. That failure is silent,
    # producing plausible numbers from the wrong geometry, and only a
    # comparison against a real run catches it.
    global FREE_GRID
    grid_base = dict(base, enh_agg="gaussian", enh_span=None,
                     gaussian_sigma=329, gaussian_truncate=2,
                     normalise="global")
    FREE_GRID = build_free_grid(grid_base)
    try:
        grid_rows, grid_stats = _synthetic_run(pairs, grid_base)
        check("free grid emits every 2x2 column on every row",
              all(all(c in r for c in free_grid_columns())
                  for r in grid_rows),
              f"columns={free_grid_columns()}")
        FREE_GRID = ()
        plain_rows, plain_stats = _synthetic_run(pairs, grid_base)
        check("free grid costs NO extra model call",
              grid_stats.gpu_calls == plain_stats.gpu_calls,
              f"{grid_stats.gpu_calls} with grid vs "
              f"{plain_stats.gpu_calls} without")
        check("free grid leaves the primary `score` untouched",
              [float(r["score"]) for r in grid_rows]
              == [float(r["score"]) for r in plain_rows])
        cells = ([(f"g{W}t{t}",
                   {"enh_agg": "gaussian", "enh_span": None,
                    "gaussian_sigma": s, "gaussian_truncate": t})
                  for W, s, t in FREE_GRID_GAUSSIANS]
                 + [(f"s{p}",
                     {"enh_agg": "sum", "enh_span": p,
                      "gaussian_sigma": None, "gaussian_truncate": None})
                    for p in FREE_GRID_SPANS])
        for name, geom in cells:
            for norm in FREE_GRID_NORMALISATIONS:
                cell = dict(grid_base, normalise=norm, **geom)
                validate(cell)      # each cell is a legal config itself
                dedicated, _ = _synthetic_run(pairs, cell)
                col = f"score__{name}-{norm}"
                worst = max(abs(float(a[col]) - float(b["score"]))
                            for a, b in zip(grid_rows, dedicated))
                check(f"free-grid {col} == a dedicated run of that cell",
                      worst == 0.0, f"max |delta| = {worst:.3e}")
    finally:
        FREE_GRID = ()

    # The two-channel map, where mean_subtract becomes a free axis.
    global FREE_MEAN_SUBTRACT
    FREE_MEAN_SUBTRACT = True
    FREE_GRID = build_free_grid(grid_base)
    try:
        ms_rows, ms_stats = _synthetic_run(pairs, grid_base)
        check("two-channel grid still costs no extra model call",
              ms_stats.gpu_calls == plain_stats.gpu_calls,
              f"{ms_stats.gpu_calls} vs {plain_stats.gpu_calls}")
        check("grid includes BOTH mean_subtract variants",
              any(c.endswith("-ms1") for c in free_grid_columns())
              and any(c.endswith("-ms0") for c in free_grid_columns()),
              f"{free_grid_columns()}")
        for ms in (True, False):
            for name, geom in (
                    ("g1000t2", {"enh_agg": "gaussian", "enh_span": None,
                                 "gaussian_sigma": 329,
                                 "gaussian_truncate": 2}),
                    ("s2000", {"enh_agg": "sum", "enh_span": 2000,
                               "gaussian_sigma": None,
                               "gaussian_truncate": None})):
                cell = dict(grid_base, mean_subtract=ms,
                            normalise="global", **geom)
                validate(cell)
                dedicated, _ = _synthetic_run(pairs, cell)
                col = f"score__{name}-global-ms{int(ms)}"
                worst = max(abs(float(a[col]) - float(b["score"]))
                            for a, b in zip(ms_rows, dedicated))
                check(f"two-channel {col} == a dedicated "
                      f"mean_subtract={ms} run",
                      worst == 0.0, f"max |delta| = {worst:.3e}")
        # The two variants must actually differ, or the checks above
        # are vacuous.
        spread = max(abs(float(r["score__g1000t2-global-ms1"])
                         - float(r["score__g1000t2-global-ms0"]))
                     for r in ms_rows)
        check("the two mean_subtract variants are genuinely different",
              spread > 1e-12, f"max |delta| = {spread:.3e}")
    finally:
        FREE_MEAN_SUBTRACT = False
        FREE_GRID = ()


def _test_free_replicate_order(check: _Check,
                               pairs: List[Tuple[Enhancer, Gene]],
                               base: Dict[str, Any]) -> None:
    """`replicate_abs_order` is a free axis, where it is live at all.

    It is decided inside `merge_maps`, that is after the maps exist, so
    it is free, but only if the grid is handed the UNMERGED store. Two
    things to establish: that the axis appears exactly where it is live
    (replicates > 1) and nowhere else, since at one replicate
    `merge_maps` is a no-op and the extra columns would be exact
    duplicates; and that each cell reproduces a dedicated run of that
    `replicate_abs_order` bit for bit.
    """
    global FREE_GRID
    rep_base = dict(base, model="borzoi", replicates=4,
                    replicate_abs_order="before", enh_agg="gaussian",
                    enh_span=None, gaussian_sigma=329,
                    gaussian_truncate=2, normalise="global")
    FREE_GRID = build_free_grid(rep_base)
    try:
        cols = free_grid_columns()
        check("replicate_abs_order IS an axis when replicates>1",
              any(c.endswith("-rbefore") for c in cols)
              and any(c.endswith("-rafter") for c in cols),
              f"{cols[:4]}")
        rep_rows, _ = _synthetic_run(pairs, rep_base)
        for order in ("before", "after"):
            cell = dict(rep_base, replicate_abs_order=order)
            validate(cell)
            dedicated, _ = _synthetic_run(pairs, cell)
            col = f"score__g1000t2-global-r{order}"
            worst = max(abs(float(a[col]) - float(b["score"]))
                        for a, b in zip(rep_rows, dedicated))
            check(f"free-grid {col} == a dedicated "
                  f"replicate_abs_order={order} run",
                  worst == 0.0, f"max |delta| = {worst:.3e}")
        spread = max(abs(float(r["score__g1000t2-global-rbefore"])
                         - float(r["score__g1000t2-global-rafter"]))
                     for r in rep_rows)
        check("the two replicate_abs_order variants are genuinely "
              "different", spread > 1e-12, f"max |delta| = {spread:.3e}")
    finally:
        FREE_GRID = ()
    # And it must be absent at replicates=1, where merge_maps is a
    # no-op.
    FREE_GRID = build_free_grid(dict(base, model="borzoi", replicates=1))
    check("replicate_abs_order is OMITTED at replicates=1 (it would be "
          "duplicate columns)",
          not any("-r" in c.rsplit("-", 1)[-1]
                  for c in free_grid_columns()),
          f"{free_grid_columns()[:4]}")
    FREE_GRID = ()


def _test_receptive_field(check: _Check,
                          pairs: List[Tuple[Enhancer, Gene]],
                          base: Dict[str, Any]) -> None:
    """The stand-in backend has a receptive field to attribute over.

    A property of the stand-in rather than of the algorithm. Without a
    receptive field a distal enhancer's gradient would be exactly zero
    and every distal assertion elsewhere would hold vacuously.
    """
    for method in ("gradient", "attention"):
        theta = dict(base, method=method)
        rows, _ = _synthetic_run(pairs, theta)
        by_gene: Dict[str, List[Tuple[int, float]]] = {}
        for r in rows:
            by_gene.setdefault(r["gene"], []).append(
                (abs(int(r["distance_to_tss"])), float(r["score"])))
        monotone = all(
            all(s0 > s1
                for (_, s0), (_, s1) in zip(sorted(v), sorted(v)[1:]))
            for v in by_gene.values())
        check(f"{method}: scores decay with |distance| (backend has a "
              f"receptive field)",
              monotone, str({g: [round(s) for _, s in sorted(v)]
                             for g, v in list(by_gene.items())[:1]}))


def _test_output_bin_geometry(check: _Check) -> None:
    """Output-bin geometry on the real model metadata.

    These models predict only the CENTRE of their input, so an
    input-bin index is not an output-bin index. Using one as the other
    reads the target signal a crop width away and raises nothing,
    because the wrong index is still in range: on Enformer it would
    place the "CAGE at TSS" scalar 40,960 bp downstream of the TSS. The
    stand-in backend is uncropped and cannot exercise this, so it is
    asserted against the real geometries instead. Borzoi's pin is 16
    bins, its prediction-time crop, and not the much wider crop it was
    trained with; prediction time is what the CRISPRi runners use.
    """
    exon_readout = dict(region="exons", bin_overlap="any")
    for meta, want_crop in ((ENFORMER_META, 320), (BORZOI_META, 16),
                            (ALPHAGENOME_META, 0)):
        check(f"{meta.name}: crop_bins == {want_crop}",
              meta.crop_bins == want_crop, f"got {meta.crop_bins}")
        check(f"{meta.name}: n_bins x bin_size + 2*crop == seq_len",
              meta.n_bins * meta.bin_size
              + 2 * meta.crop_bins * meta.bin_size == meta.seq_len)
        window = Window(start=0, length=meta.seq_len)
        centre = meta.seq_len // 2
        check(f"{meta.name}: anchor at window centre -> middle output "
              f"bin",
              bin_of(centre, window, meta) == meta.n_bins // 2,
              f"got {bin_of(centre, window, meta)}, "
              f"want {meta.n_bins // 2}")
        first = meta.crop_bins * meta.bin_size
        last = first + meta.n_bins * meta.bin_size - 1
        check(f"{meta.name}: first predicted bp -> output bin 0",
              bin_of(first, window, meta) == 0,
              f"got {bin_of(first, window, meta)}")
        check(f"{meta.name}: last predicted bp -> output bin n_bins-1",
              bin_of(last, window, meta) == meta.n_bins - 1)
        if meta.crop_bins:
            check(f"{meta.name}: inside the leading crop -> None "
                  f"(not clamped)",
                  bin_of(first - 1, window, meta) is None,
                  f"got {bin_of(first - 1, window, meta)}")
            check(f"{meta.name}: inside the trailing crop -> None "
                  f"(not clamped)",
                  bin_of(meta.seq_len - 1, window, meta) is None,
                  f"got {bin_of(meta.seq_len - 1, window, meta)}")

        # READOUT-BINS' exon branch must live on the SAME grid as
        # bin_of. Nothing else catches a mismatch, because every Avsec
        # 2021 arm uses region=tss and goes through bin_of. A one-base
        # exon must land in exactly the bin bin_of names.
        for pos in (first, centre, last):
            gene = Gene(gid="X", chrom="chrN", strand="+",
                        provided_tss=centre, gencode_tss=(centre,),
                        exons=((pos, pos + 1),))
            want = bin_of(pos, window, meta)
            got = readout_bins(gene, window, exon_readout, meta)
            check(f"{meta.name}: one-base exon at {pos} -> output bin "
                  f"{want} (crop applied)",
                  got == (want,), f"got {got}, want {(want,)}")
        if meta.crop_bins:
            gene = Gene(gid="X", chrom="chrN", strand="+",
                        provided_tss=centre, gencode_tss=(centre,),
                        exons=((0, meta.bin_size),))
            check(f"{meta.name}: exon inside the leading crop -> no "
                  f"readout bin",
                  readout_bins(gene, window, exon_readout, meta) == (),
                  "an un-cropped exon index would have returned bin 0")


def _test_bin_overlap(check: _Check,
                      pairs: List[Tuple[Enhancer, Gene]],
                      base: Dict[str, Any]) -> None:
    """`bin_overlap`: a bp interval read on a coarse grid.

    Enformer's attention lives on 128 bp tokens, so a 2 kb window is
    15.625 tokens and cannot be taken exactly. The three readings must
    differ, and must differ only on the grid.
    """
    bin_size = 128
    ones = np.ones(SYNTH_SEQ_LEN)      # counts tokens when summed
    window = Window(0, SYNTH_SEQ_LEN)
    seen_counts = {}
    for rule in ("any", "majority", "fractional"):
        theta = dict(base, method="attention", enh_agg="sum",
                     enh_span=2000, normalise="none", bin_overlap=rule)
        counts = set()
        for phase in range(0, bin_size, 16):
            mid = SYNTH_SEQ_LEN // 2 + phase
            enh = Enhancer("chr1", mid - 250, mid + 250, "E")
            counts.add(round(reduce_pair(ones, enh, window, theta,
                                         bin_size), 3))
        seen_counts[rule] = sorted(counts)
    check("bin_overlap=any gives 16-17 whole tokens",
          seen_counts["any"] == [16.0, 17.0], str(seen_counts["any"]))
    check("bin_overlap=majority gives 15-16 whole tokens",
          seen_counts["majority"] == [15.0, 16.0],
          str(seen_counts["majority"]))
    check("bin_overlap=fractional gives exactly 2000/128 tokens, "
          "phase-independent",
          seen_counts["fractional"] == [15.625],
          str(seen_counts["fractional"]))
    check("the three readings genuinely differ",
          len({tuple(v) for v in seen_counts.values()}) == 3)
    rows_any, _ = _synthetic_run(pairs, dict(base, bin_overlap="any"))
    rows_maj, _ = _synthetic_run(pairs,
                                 dict(base, bin_overlap="majority"))
    check("bin_overlap does not touch a bp-resolution gradient window",
          all(abs(x["score"] - y["score"]) < 1e-12
              for x, y in zip(rows_any, rows_maj)))
    # It does change the exon readout, so it is GPU-forcing off the
    # attention path.
    check("bin_overlap is in the cache key off the attention path",
          "bin_overlap" in gpu_subset(dict(base, method="gradient"))
          and "bin_overlap"
          not in gpu_subset(dict(base, method="attention")))


def _test_ism_edge_clipping(check: _Check,
                            pairs: List[Tuple[Enhancer, Gene]],
                            base: Dict[str, Any]) -> None:
    """ISM clips a perturbation window at the input edge.

    An enhancer whose midpoint is inside the window is scoreable even
    when its 2 kb perturbation span crosses the edge, so the span is
    clipped rather than the pair dropped. On Fulco a dozen pairs sit in
    exactly that band, just inside Enformer's 98,304 bp half-window;
    dropping them would score them 0.0 while gradient and attention
    scored them normally.

    ISM-MAP is called directly with an explicit window, which isolates
    the clipping arithmetic from anchor resolution and asserts what
    matters: that the enhancer gets a perturbation draw at all.
    Asserting a score > 0 end to end would pass vacuously, since the
    stand-in's response 98 kb from the anchor underflows to 0.0 either
    way.
    """
    theta = dict(base, method="ism", enh_agg="point", enh_span=None,
                 ism_shuffles=1, ism_mode="random-replace",
                 ism_statistic="abs_mean_diff", normalise="none",
                 log_scalar=False, seed=0)
    backend = make_backend(theta, "synthetic")
    gene = pairs[0][1]
    window = window_at(gene.provided_tss, backend.meta.seq_len)
    # Inside the window; its 2 kb span overhangs the edge by 750 bp.
    mid = window.start + window.length - 250
    edge = Enhancer(gene.chrom, mid - 150, mid + 150, "EDGE")
    draws, _ = ism_map(gene, [edge], window, 1, Aug(shift=0, rc=False),
                       theta, resolve(theta, backend.meta), backend)
    check("ISM clips an edge-overhanging perturbation window instead "
          "of dropping the pair", "EDGE" in draws,
          "the pair got no perturbation draw: it was discarded, not "
          "clipped")


def _test_perturb(check: _Check) -> None:
    """Every `ism_mode` is a valid, local, in-place edit.

    Cheap invariants, but the ISM arm is the one reproducing Avsec
    2021's Figure 2b, and a perturbation that silently leaked outside
    [p, q) or stopped being a one-hot would look exactly like a
    disagreement with the published recipe rather than a bug.
    """
    rng = np.random.RandomState(0)
    length = 8192
    x = np.zeros((length, 4), dtype="float32")
    bases = np.random.RandomState(1).randint(0, 4, size=length)
    x[np.arange(length), bases] = 1.0
    p, q = 3000, 5000
    for mode in ("random-probs", "zero-replace", "random-replace",
                 "genomic-replace", "mononuc-shuffle", "dinuc-shuffle",
                 "eulerian-dinuc-shuffle"):
        out = perturb(x, p, q, mode, rng)
        if mode == "random-probs":
            check("perturb random-probs: window in [0,1), not a "
                  "one-hot, nothing else touched",
                  bool(out[p:q].min() >= 0.0)
                  and bool(out[p:q].max() < 1.0)
                  and not bool(np.all(out[p:q].sum(axis=1) == 1.0))
                  and bool(np.array_equal(out[:p], x[:p]))
                  and bool(np.array_equal(out[q:], x[q:])))
        elif mode == "zero-replace":
            check("perturb zero-replace: the window is blank, nothing "
                  "else is",
                  bool(np.all(out[p:q] == 0.0))
                  and bool(np.all(out[:p].sum(axis=1) == 1.0))
                  and bool(np.all(out[q:].sum(axis=1) == 1.0)))
        else:
            check(f"perturb {mode}: still a one-hot",
                  bool(np.all(out.sum(axis=1) == 1.0)))
        check(f"perturb {mode}: leaves everything outside [p,q) "
              f"untouched",
              bool(np.array_equal(out[:p], x[:p])
                   and np.array_equal(out[q:], x[q:])))
        check(f"perturb {mode}: does not alias the input array",
              out is not x)
    check("perturb zero-replace is deterministic: the seed cannot "
          "matter",
          all(np.array_equal(
              perturb(x, p, q, "zero-replace", np.random.RandomState(k)),
              perturb(x, p, q, "zero-replace", np.random.RandomState(0)))
              for k in range(4)))

    def dinuc_counts(one_hot):
        """The 16 overlapping-dinucleotide counts of a one-hot."""
        tokens = one_hot.argmax(1)
        counts = np.zeros(16)
        for a, b in zip(tokens[:-1], tokens[1:]):
            counts[a * 4 + b] += 1
        return counts

    # The distinction the two similarly named modes turn on.
    # `eulerian-dinuc-shuffle` must reproduce the dinucleotide count
    # vector exactly, and `dinuc-shuffle` must not: were they to agree,
    # the two modes would have collapsed into one and the sweep would
    # be measuring nothing.
    ref = dinuc_counts(x[p:q])
    check("perturb eulerian-dinuc-shuffle preserves dinucleotide "
          "counts EXACTLY",
          all(bool(np.array_equal(
              dinuc_counts(perturb(x, p, q, "eulerian-dinuc-shuffle",
                                   np.random.RandomState(k))[p:q]), ref))
              for k in range(6)))
    check("perturb eulerian-dinuc-shuffle preserves base composition "
          "too (it must, a fortiori)",
          bool(np.array_equal(
              np.sort(perturb(x, p, q, "eulerian-dinuc-shuffle",
                              np.random.RandomState(3))[p:q].argmax(1)),
              np.sort(x[p:q].argmax(1)))))
    check("perturb dinuc-shuffle does NOT preserve dinucleotide counts "
          "(it is a block shuffle)",
          any(not bool(np.array_equal(
              dinuc_counts(perturb(x, p, q, "dinuc-shuffle",
                                   np.random.RandomState(k))[p:q]), ref))
              for k in range(6)))
    check("perturb eulerian-dinuc-shuffle actually reorders (it is not "
          "the identity)",
          any(not bool(np.array_equal(
              perturb(x, p, q, "eulerian-dinuc-shuffle",
                      np.random.RandomState(k))[p:q], x[p:q]))
              for k in range(4)))
    check("ism_shuffles admits 5 (half of Avsec 2026's published 10)",
          5 in PARAMS["ism_shuffles"].choices)
    check("eulerian-dinuc-shuffle is a declared ism_mode",
          "eulerian-dinuc-shuffle" in PARAMS["ism_mode"].choices)
    check("perturb mononuc-shuffle preserves base composition",
          bool(np.array_equal(
              np.sort(perturb(x, p, q, "mononuc-shuffle",
                              rng)[p:q].argmax(1)),
              np.sort(x[p:q].argmax(1)))))
    # genomic-replace must copy real sequence from OUTSIDE the
    # perturbed span and away from the TSS, so that the promoter can
    # never be duplicated into the enhancer slot.
    span = q - p
    guard = min(5000, length // 20)
    donor_ok = []
    for seed in range(12):
        segment = perturb(x, p, q, "genomic-replace",
                          np.random.RandomState(seed))[p:q].argmax(1)
        hits = [start for start in range(length - span)
                if np.array_equal(x[start:start + span].argmax(1),
                                  segment)]
        donor_ok.append(bool(hits) and all(
            not (start < q and p < start + span)
            and not (start < length // 2 + guard
                     and length // 2 - guard < start + span)
            for start in hits))
    check("perturb genomic-replace copies real sequence from outside "
          "[p,q) and off the TSS", all(donor_ok))
    check("perturb genomic-replace guard leaves a legal donor even on "
          "a short window", guard * 2 + span < length)


def _test_definition_matches_driver(check: _Check,
                                    pairs: List[Tuple[Enhancer, Gene]],
                                    base: Dict[str, Any]) -> None:
    """`score_pair` and `run_experiment` agree, pair for pair."""
    equality_cases = {
        "gradient, single-tss": dict(base),
        "gradient, each-tss": dict(base, anchor="each-tss",
                                   anchor_agg="mean"),
        "gradient, each-tss max": dict(base, anchor="each-tss",
                                       anchor_agg="max"),
        "gradient, enhancer-anchored": dict(base, anchor="enhancer"),
        "gradient, rc+shift TTA": dict(base, tta="rc+shift",
                                       tta_shift_bp=3),
        # The case that guards the random draw. `avsec8` is the only
        # tta whose augmentation set depends on something other than
        # theta, so it is the only one that can make the definition and
        # the driver disagree: score_pair scores one pair in isolation
        # while run_experiment walks a gene list, and any RNG that
        # streamed with that walk would give them different eights.
        # Equality here asserts that the draw is keyed on the gene and
        # on nothing else.
        "gradient, Avsec 2021 random-8 TTA": dict(base, tta="avsec8",
                                                  tta_shift_bp=3),
        "gradient, 4 replicates": dict(base, replicates=4),
        "attention": dict(base, method="attention"),
        "ism, P=10": dict(base, method="ism", enh_agg="point",
                          enh_span=None, ism_shuffles=10,
                          normalise="none", log_scalar=False),
        "ism, each-tss + rc + 4 reps": dict(
            base, method="ism", enh_agg="point", enh_span=None,
            ism_shuffles=10, normalise="none", log_scalar=False,
            anchor="each-tss", anchor_agg="mean", tta="rc",
            replicates=4, ism_inner_mean=("strand",)),
        "ism, inner mean over tss": dict(
            base, method="ism", enh_agg="point", enh_span=None,
            ism_shuffles=10, normalise="none", log_scalar=False,
            anchor="each-tss", anchor_agg="mean",
            ism_inner_mean=("tss",)),
        # MERGE-AUGMENTATIONS is a branch in both the definition and
        # the driver, so their agreement has to be re-established on
        # it, and in more than one case, because the merge interacts
        # with the loops around it: replicates, and an
        # enhancer-dependent anchor that moves the whole block inside
        # the pair loop.
        "gradient, tta_abs_order=after": dict(base, tta="rc",
                                              tta_abs_order="after"),
        "gradient, tta_abs_order=after x 4 reps": dict(
            base, tta="rc", tta_abs_order="after", replicates=4),
        "gradient, tta_abs_order=after, enhancer-anchored": dict(
            base, tta="rc", tta_abs_order="after", anchor="enhancer"),
        "attention, tta_abs_order=after": dict(
            base, method="attention", tta="rc", tta_abs_order="after"),
        # The same for the replicate axis, including BOTH axes at once,
        # which is what Linder 2025's Borzoi gradient arm is.
        # `merge_maps` collapses two axes in one pass and its output
        # shape has to stay consistent with what COMBINE is then told.
        "gradient, replicate_abs_order=after": dict(
            base, replicates=4, replicate_abs_order="after"),
        "gradient, both abs_orders=after": dict(
            base, replicates=4, replicate_abs_order="after", tta="rc",
            tta_abs_order="after"),
        "gradient, both after + each-tss max": dict(
            base, replicates=4, replicate_abs_order="after", tta="rc",
            tta_abs_order="after", anchor="each-tss",
            anchor_agg="max"),
        "gradient, both after + enhancer-anchored": dict(
            base, replicates=4, replicate_abs_order="after", tta="rc",
            tta_abs_order="after", anchor="enhancer"),
    }
    for label, theta in equality_cases.items():
        backend = make_backend(theta, "synthetic")
        rho = resolve(theta, backend.meta)
        rho.warnings = validate(theta)
        rows, _ = _synthetic_run(pairs, theta)
        worst, n = 0.0, 0
        for row in rows:
            enh, gene = next((e, g) for e, g in pairs
                             if e.eid == row["enhancer"])
            defn = score_pair(enh, gene, theta, rho, backend)
            worst = max(worst, abs(float(defn) - float(row["score"])))
            n += 1
        check(f"definition == implementation ({label})",
              n == len(pairs) and worst < 1e-12,
              f"{n} pairs, max |delta| = {worst:.3e}")


def _test_post_map_cache(check: _Check, base: Dict[str, Any]) -> None:
    """The per-gene `PostMapCache` and the Gaussian memo change no number.

    Every reduction, divisor and mean_subtract variant is scored on the
    same maps with and without the cache and must agree exactly, not
    approximately: the cache reorders no arithmetic, so any difference
    at all means it served the wrong array.
    """
    rng = np.random.default_rng(7)
    L = 6000
    maps = {"1-D": rng.standard_normal(L),
            "two-channel": rng.standard_normal((L, 2))}
    w = Window(start=0, length=L)
    enhs = [Enhancer("chrS", c - 250, c + 250, f"e{c}")
            for c in (40, 900, 3001, 5990)]
    pm = PostMapCache()
    n_cmp, bad = 0, []
    for mname, M in maps.items():
        for ms in (False, True):
            for abs_map in (True, False):
                for norm in ("none",) + tuple(DIVISOR_KINDS):
                    for agg, extra in (("point", {"enh_span": None}),
                                       ("sum", {}), ("mean", {}),
                                       ("max", {}),
                                       ("gaussian", {"enh_span": None,
                                                     "gaussian_sigma": 300,
                                                     "gaussian_truncate": 4})):
                        th = dict(base, mean_subtract=ms, abs_map=abs_map,
                                  normalise=norm, enh_agg=agg, **extra)
                        for e in enhs:
                            a = reduce_pair(M, e, w, th, 1)
                            b = reduce_pair(M, e, w, th, 1, cache=pm)
                            n_cmp += 1
                            if not (a == b or (a != a and b != b)):
                                bad.append((mname, ms, abs_map, norm, agg,
                                            e.eid, a, b))
    check("post-map cache: reduce_pair identical with and without it",
          not bad, f"{len(bad)} of {n_cmp} differ, e.g. {bad[:2]}")
    for ms in (False, True):
        ref = divisor_columns(list(maps.values()), ms)
        got = divisor_columns(list(maps.values()), ms, cache=pm)
        check(f"post-map cache: divisor columns identical (ms={ms})",
              ref == got)
    # The memo's slice against the per-call expression it stands in for.
    sigma, h, m, L2 = 300.0, 1200, 700, 5000
    lo, hi = max(0, m - h), min(L2, m + h + 1)
    idx = np.arange(lo, hi)
    direct = np.exp(-((idx - m) ** 2) / (2.0 * sigma * sigma))
    check("Gaussian memo: slice equals the per-call weights bit for bit",
          np.array_equal(_gaussian_weights(sigma, h)[lo - (m - h): hi - (m - h)],
                         direct))
    # A merge served from the cache is the merge itself.
    store = {(0, r, 0): rng.standard_normal(L) for r in range(3)}
    th = dict(base, replicate_abs_order="after")
    m1, n_r1, n_a1 = merge_maps(store, 1, 3, 1, th)
    m2, n_r2, n_a2 = pm.merged(store, 1, 3, 1, th)
    check("post-map cache: merged maps identical",
          (n_r1, n_a1) == (n_r2, n_a2) and m1.keys() == m2.keys()
          and all(np.array_equal(m1[k], m2[k]) for k in m1))
    check("post-map cache: a second request is served, not recomputed",
          pm.merged(store, 1, 3, 1, th)[0] is m2)


def _test_frame_guard(check: _Check,
                      pairs: List[Tuple[Enhancer, Gene]],
                      base: Dict[str, Any]) -> None:
    """The readout follows the augmentation, on every real geometry.

    The synthetic backend has no rc symmetry, so it cannot see a readout
    left in the forward frame on an rc pass; the exact probe has that
    symmetry. On each model's real geometry the probe must certify an
    rc + shift readout, and must CATCH the forward-frame readout when
    `frame_for` is replaced by one that ignores the augmentation.
    """
    cases = ((ENFORMER_META, dict(model="enformer", assay="cage",
                                  cell_type="k562", region="tss",
                                  tss_width=384)),
             (BORZOI_META, dict(model="borzoi", assay="rnaseq",
                                cell_type="k562", region="exons",
                                anchor="gene-midpoint", tss_width=None,
                                log_scalar=True)),
             (ALPHAGENOME_META, dict(model="alphagenome", assay="cage",
                                     cell_type="k562", region="tss",
                                     tss_width=384)))
    for meta, over in cases:
        th = fill_defaults(dict(base, **over, tta="rc+shift", tta_shift_bp=3,
                                method="gradient", screen="gschwind",
                                gencode="v41"))
        _FRAME_SELFTEST_DONE.clear()
        try:
            rep = frame_selftest(meta, th)
            check(f"{meta.name}: frame probe certifies rc+shift",
                  rep.get("status") == "passed", str(rep))
        except Exception as e:                                  # noqa: BLE001
            check(f"{meta.name}: frame probe certifies rc+shift", False,
                  f"{type(e).__name__}: {e}")
        orig = globals()["frame_for"]
        globals()["frame_for"] = (
            lambda gene, aug, meta_: ReadoutFrame(gene.strand, 0, False))
        _FRAME_SELFTEST_DONE.clear()
        try:
            frame_selftest(meta, th)
            check(f"{meta.name}: frame probe catches the forward-frame rc "
                  "readout", False, "probe passed")
        except FrameError:
            check(f"{meta.name}: frame probe catches the forward-frame rc "
                  "readout", True)
        finally:
            globals()["frame_for"] = orig
            _FRAME_SELFTEST_DONE.clear()

    # A shift between half a bin and a whole bin is refused before any
    # model call; under half a bin the readout stays, a whole bin moves it.
    check("shift_bins: under half a bin stays", shift_bins(3, ENFORMER_META) == 0)
    check("shift_bins: a whole bin follows",
          shift_bins(-256, ENFORMER_META) == -2)
    try:
        shift_bins(64, ENFORMER_META)
        check("shift_bins: half a bin is refused", False, "accepted")
    except FrameError:
        check("shift_bins: half a bin is refused", True)
    # A readout built for one augmentation inside a pass on another raises.
    g = pairs[0][1]
    try:
        with augmented_frame(Aug(0, True)):
            frame_for(g, None, ENFORMER_META)
        check("augmented_frame: a forgotten aug= raises", False, "no error")
    except FrameError:
        check("augmented_frame: a forgotten aug= raises", True)
    # A synthetic rc run carries a passed certificate and an active gate.
    th = dict(base, tta="rc")
    _synthetic_run(pairs, th)
    check("run_experiment records the probe certificate",
          FRAME_REPORT.get("status") == "passed", str(FRAME_REPORT)[:200])
    check("run_experiment records the agreement gate",
          FRAME_REPORT.get("agreement", {}).get("status")
          in ("passed", "no-pairs"), str(FRAME_REPORT.get("agreement")))


def _test_abs_order_vacuity(check: _Check,
                            pairs: List[Tuple[Enhancer, Gene]],
                            base: Dict[str, Any]) -> None:
    """What the `abs_order` equality cases do and do not establish.

    The four `tta_abs_order=after` cases in
    `_test_definition_matches_driver` would pass identically if the
    branch were a no-op. They are not vacuous: the two orientations'
    maps really do differ, so MERGE-AUGMENTATIONS averages two
    different arrays. But their signs agree at every position, so
    Jensen is tight, mean_a |M_a| == |mean_a M_a| exactly, and the two
    orderings coincide. Nothing on the stand-in can separate them.

    That sign agreement is structural rather than luck. The stand-in's
    forward is linear in the one-hot, so dy[b,t]/dx[i,k] = K[k,t]*w[i]
    and the reference-channel map is w[i]*f(ref[i]): positionally
    separable, with a positive envelope w. Reversing the sequence
    permutes the envelope and so changes every magnitude, but a
    position's sign depends only on its reference base, which the
    reverse-complement pass reads at the same locus. `mean_subtract`
    does not help, subtracting a per-position multiple of the same
    w[i] and leaving w[i]*(f(ref[i]) - mean f).

    Whether the ordering moves a SCORE is therefore a question only a
    real model can answer. The checks below record that, rather than
    letting four green lines elsewhere stand as evidence that it does.
    """
    theta_before = dict(base, tta="rc", tta_abs_order="before",
                        normalise="none")
    theta_after = dict(base, tta="rc", tta_abs_order="after",
                       normalise="none")
    backend = make_backend(theta_before, "synthetic")
    rho = resolve(theta_before, backend.meta)
    e0, g0 = pairs[0]
    w0 = anchor_windows(e0, g0, theta_before, rho, backend)[0]
    orient = [attribution_map(w0, g0, 1, a, theta_before, rho, backend)
              for a in augmentations(theta_before, g0.gid)]
    spread = float(np.abs(orient[0] - orient[1]).max())
    scale = float(np.abs(orient[0]).max())
    check("the two orientations' maps are genuinely different arrays "
          "(so MERGE-MAPS is executed, not skipped)",
          len(orient) == 2 and spread > 0.1 * scale,
          f"max |M_fwd - M_rc| = {spread:.3e} against "
          f"max |M_fwd| = {scale:.3e}")
    nz = np.abs(orient[0]) > 0
    check("...but their signs agree everywhere, so Jensen is tight and "
          "the two orderings "
          "COINCIDE on the stand-in -- the score-level test needs a "
          "real model",
          bool(nz.any())
          and np.all(np.sign(orient[0][nz]) == np.sign(orient[1][nz])),
          "signs differ somewhere, which would make the score-level "
          "check below meaningful")
    # The replicate axis has the same status for the same reason: the
    # four maps differ in magnitude while their signs agree, so
    # `replicate_abs_order` is vacuous here too. Asserted rather than
    # assumed, because a stand-in whose signs ever varied would make
    # the `after` equality cases meaningful and this reasoning wrong.
    theta_rep = dict(base, replicates=4, normalise="none")
    backend_rep = make_backend(theta_rep, "synthetic")
    rho_rep = resolve(theta_rep, backend_rep.meta)
    w_rep = anchor_windows(e0, g0, theta_rep, rho_rep, backend_rep)[0]
    fwd = augmentations(theta_rep, g0.gid)[0]
    reps = [attribution_map(w_rep, g0, r, fwd, theta_rep, rho_rep,
                            backend_rep) for r in (1, 2, 3, 4)]
    spread = float(np.abs(reps[0] - reps[1]).max())
    check("the four replicate maps are genuinely different arrays",
          spread > 0.1 * float(np.abs(reps[0]).max()),
          f"max |M_r1 - M_r2| = {spread:.3e}")
    nz = np.abs(reps[0]) > 0
    check("...and their signs agree across all four, so "
          "replicate_abs_order is vacuous here too",
          bool(nz.any())
          and all(np.all(np.sign(reps[0][nz]) == np.sign(reps[k][nz]))
                  for k in (1, 2, 3)),
          "signs differ across replicates, which would make the "
          "score-level difference measurable on the stand-in")
    rows_before, _ = _synthetic_run(pairs, theta_before)
    rows_after, _ = _synthetic_run(pairs, theta_after)
    s_before = {r["enhancer"]: r["score"] for r in rows_before}
    s_after = {r["enhancer"]: r["score"] for r in rows_after}
    check("un-normalised: mean_a |M_a| >= |mean_a M_a| for every pair "
          "(Jensen, with equality here)",
          all(s_before[k] >= s_after[k] - 1e-9
              for k in s_before if k in s_after),
          "found a pair where the signed-mean-then-abs is LARGER, "
          "which is impossible")


def _test_shipped_configs(check: _Check) -> None:
    """Every shipped config validates and resolves."""
    cfg_dir = os.path.join(_HERE, "configs")
    if os.path.isdir(cfg_dir):
        names = sorted(f for f in os.listdir(cfg_dir)
                       if f.endswith((".yaml", ".yml", ".json")))
        check("configs/ is not empty", bool(names))
        for name in names:
            try:
                theta = fill_defaults(
                    load_config(os.path.join(cfg_dir, name)))
                validate(theta)
                resolve(theta)
                check(f"config {name}", True)
            except Exception as exc:                    # noqa: BLE001
                check(f"config {name}", False,
                      f"{type(exc).__name__}: {exc}")


def _test_backend_capabilities(check: _Check, base: Dict[str, Any]) -> None:
    """Optional backend methods: capability detection and fail-fast.

    The contract is that only `forward` is mandatory, and a backend
    serves exactly the methods its implemented functions support.
    Enforced rather than documented, because the failure it prevents
    (NotImplementedError mid-run, after weights are loaded and 78k
    pairs are read) is expensive and easy to introduce.
    """
    meta = MODEL_META["enformer"]

    class ForwardOnly(Backend):
        def __init__(self):
            self.meta = meta

        def forward(self, x, replicate):
            return np.zeros((self.meta.n_bins, 8))

    class ForwardAndGradient(ForwardOnly):
        def scalar_gradient(self, x, replicate, scalar_fn):
            return np.zeros_like(x)

    capable = SyntheticBackend(meta)
    caps_capable = backend_capabilities(capable)
    caps_forward = backend_capabilities(ForwardOnly())
    check("capabilities: SyntheticBackend implements all three",
          all(caps_capable.values()), str(caps_capable))
    check("capabilities: forward-only backend detected as forward-only",
          caps_forward == {"forward": True, "scalar_gradient": False,
                           "attention": False},
          str(caps_forward))
    check("capabilities: UnavailableBackend declares nothing, despite "
          "overriding all three",
          backend_capabilities(UnavailableBackend(meta, "x"))
          == {"forward": False, "scalar_gradient": False,
              "attention": False})

    def refusal(backend, method):
        """The message `check_backend` refuses with, or None."""
        theta = dict(base, method=method)
        if method == "ism":
            theta = fill_defaults(dict(theta, ism_shuffles=1,
                                       ism_mode="shuffle"))
        try:
            check_backend(theta, backend)
            return None
        except ConfigError as exc:
            return str(exc)

    # Forward-only: ISM runs, gradient and attention are refused BY
    # NAME.
    check("fail-fast: forward-only serves ism",
          refusal(ForwardOnly(), "ism") is None)
    for method, needed in (("gradient", "scalar_gradient"),
                           ("attention", "attention")):
        msg = refusal(ForwardOnly(), method)
        check(f"fail-fast: forward-only refuses {method}",
              msg is not None)
        check(f"fail-fast: the {method} message names {needed}()",
              bool(msg) and needed in msg, msg or "")
        check(f"fail-fast: the {method} message says what CAN run",
              bool(msg) and "ism" in msg, msg or "")
    # Adding scalar_gradient turns on the gradient arm, and only it.
    check("fail-fast: +scalar_gradient enables gradient",
          refusal(ForwardAndGradient(), "gradient") is None)
    check("fail-fast: +scalar_gradient still refuses attention",
          refusal(ForwardAndGradient(), "attention") is not None)
    for method in ("gradient", "attention", "ism"):
        check(f"fail-fast: full backend serves {method}",
              refusal(capable, method) is None)
    # An unwired model is refused up front, with its requirement quoted.
    msg = refusal(UnavailableBackend(meta, "a Borzoi checkpoint"), "ism")
    check("fail-fast: UnavailableBackend refused before the run",
          msg is not None)
    check("fail-fast: its message quotes the requirement",
          bool(msg) and "Borzoi checkpoint" in msg, msg or "")


def _build_parser() -> argparse.ArgumentParser:
    """The command line. One process scores one arm."""
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config",
                    help="YAML or JSON file of parameter values")
    ap.add_argument(
        "--set", action="append", default=[], metavar="KEY=VALUE",
        help="override one parameter, repeatable. Applied after the "
             "config and validated identically, so an override cannot "
             "reach a state a config could not, and --manifest records "
             "the theta it produced")
    ap.add_argument("--out",
                    help="TSV of per-pair scores (default: stdout)")
    ap.add_argument("--manifest",
                    help="write the resolved run manifest here as JSON")
    ap.add_argument("--backend", default="real",
                    choices=("real", "synthetic"))
    ap.add_argument("--fasta",
                    help="indexed reference FASTA, required by a real "
                         "backend")
    ap.add_argument("--anchors",
                    help="TSV of precomputed MAJOR-TSS per gene; "
                         "without it every candidate TSS costs a "
                         "forward pass")
    ap.add_argument("--device", default="cuda",
                    help="torch device for a real backend")
    ap.add_argument("--print-config", action="store_true",
                    help="resolve and validate θ, print it, run nothing")
    ap.add_argument("--self-test", action="store_true",
                    help="check the algorithm's numerical identities "
                         "and exit; needs no data and no accelerator")
    ap.add_argument(
        "--ism-draws", metavar="PATH.npz", default=None,
        help="method=ism only: write the raw per-shuffle signed "
             "differences here, which leaves every ism_statistic and "
             "every ism_inner_mean subset recoverable as "
             "post-processing rather than as another full ISM wave")
    ap.add_argument(
        "--shard", metavar="I/N", default=None,
        help="score only the genes whose global rank %% N == I, to "
             "spread one arm over N accelerators. Each shard writes a "
             "partial table carrying a `_gene_rank` column, and a "
             "stable sort on that column rebuilds the table an "
             "unsharded run writes. Row order is load-bearing, so "
             "merge -- never concatenate")
    ap.add_argument(
        "--only-genes", metavar="GID[,GID...]", default=None,
        help="score only these gene IDs, each keeping its GLOBAL gene "
             "rank so the rows are a subsequence of a full run's. For "
             "topping up an otherwise-complete table. A gene ID absent from the cohort is a fatal "
             "error, not an empty table")
    ap.add_argument(
        "--free-grid", action="store_true",
        help="also emit every post-map reduction in FREE_GRID as a "
             "`score__<pooling>-<normalise>` column, computed from the "
             "same attribution maps as `score` and so costing no "
             "further model call. Gradient and attention arms only: an "
             "ISM arm reduces at enh_agg=point and has no window to "
             "re-pool, where --ism-draws is the same idea")
    ap.add_argument(
        "--multi-channels", action="store_true",
        help="DESIGN B, assay=multi only: one backward pass per assay "
             "instead of one for their summed log, returning an "
             "(L, 2N) map on which every weighting over the assays, "
             "and every subset, is a post-map reduction. Roughly N "
             "times the cost")
    ap.add_argument(
        "--free-mean-subtract", action="store_true",
        help="add mean_subtract to the free grid by making the "
             "gradient map carry TWO channels, the reference-base "
             "gradient and the per-position mean over the four bases, "
             "so both variants come from one backward pass. Requires "
             "bin_abs_order=after. Implies --free-grid")
    return ap


def _apply_scope_flags(args: argparse.Namespace,
                       ap: argparse.ArgumentParser) -> None:
    """Set the switches that narrow what this process computes.

    None of the three is a scientific choice, so none is a Param: they
    say which rows one process computes and what it records along the
    way, never what a row means.
    """
    global ISM_DRAWS, SHARD, ONLY_GENES
    if args.ism_draws:
        ISM_DRAWS = []
    if args.shard:
        try:
            i, n = (int(v) for v in args.shard.split("/"))
        except ValueError:
            ap.error(f"--shard must look like I/N, got {args.shard!r}")
        if not (n >= 1 and 0 <= i < n):
            ap.error(f"--shard {args.shard}: need N >= 1 and 0 <= I < N")
        SHARD = (i, n)
    if args.only_genes:
        gids = tuple(g.strip() for g in args.only_genes.split(",")
                     if g.strip())
        if not gids:
            ap.error("--only-genes was given but lists no gene IDs")
        ONLY_GENES = frozenset(gids)


def _configure_multi_channels(theta: Dict[str, Any],
                              free_mean_subtract: bool) -> None:
    """Switch design B on: one channel pair per assay group."""
    global MULTI_CHANNEL, MULTI_MIX, MULTI_MIX_NAMES, CO_W
    if theta["assay"] != "multi":
        raise ConfigError(
            "--multi-channels requires assay=multi: it splits the "
            "summed-log readout into one backward pass per assay "
            "GROUP, and there are no groups for "
            f"assay={theta['assay']!r}")
    if free_mean_subtract:
        raise ConfigError(
            "--multi-channels is incompatible with "
            "--free-mean-subtract: "
            "the extra channels occupy the same slot")
    grp = MULTI_GROUPS.get((MODEL_META[theta["model"]].name,
                            theta["assay"], theta["cell_type"]), ())
    if not grp:
        raise ConfigError(
            "--multi-channels: no MULTI_GROUPS entry for "
            f"({theta['model']}, {theta['assay']}, "
            f"{theta['cell_type']})")
    MULTI_CHANNEL = True
    MULTI_MIX_NAMES = grp
    MULTI_MIX = build_multi_mix(len(grp), tuple(n for n, _ in grp))
    # The primary `score` column is the eqlog mix: no weight on the
    # raw channels, 1/N on the log ones.
    CO_W = tuple([0.0] * len(grp) + [1.0 / len(grp)] * len(grp))
    print(f"design B: {len(grp)} assay channels x 2 = "
          f"{2*len(grp)}-channel map, {len(MULTI_MIX)} mixes x 2 "
          f"normalise = {len(MULTI_MIX)*2} columns "
          f"({', '.join(n for n, _ in grp)})", file=sys.stderr)


def _configure_free_grid(theta: Dict[str, Any],
                         free_mean_subtract: bool) -> None:
    """Switch the post-map reduction grid on."""
    global FREE_GRID, FREE_MEAN_SUBTRACT
    if theta["method"] not in ("gradient", "attention"):
        raise ConfigError(
            "--free-grid is meaningless for "
            f"method={theta['method']!r}: only the gradient and "
            "attention arms produce a re-poolable map. An ISM arm "
            "already reduces at enh_agg=point, so there is no window "
            "to re-weight; use --ism-draws, which is the same idea "
            "for ism_statistic. Refusing rather than emitting four "
            "identical columns.")
    if free_mean_subtract:
        # Forming the mean_subtract variant after the fact needs the
        # map to stay linear, and bin_abs_order=before takes |.| per
        # readout bin before the bins are summed.
        if theta["bin_abs_order"] == "before":
            raise ConfigError(
                "--free-mean-subtract is incompatible with "
                "bin_abs_order=before: that setting takes |.| per "
                "readout bin inside GRADIENT-MAP, before the bins are "
                "summed, so the mean_subtract variant can no longer "
                "be formed after the fact (|A - m| != |A| - |m|). Use "
                "bin_abs_order=after, which is the backbone.")
        FREE_MEAN_SUBTRACT = True
    FREE_GRID = build_free_grid(theta)
    cols = free_grid_columns()
    print(f"free grid: {len(cols)} post-map cells from the same maps "
          "(no extra model calls)"
          + (" [two-channel map: mean_subtract is a free axis]"
             if free_mean_subtract else ""), file=sys.stderr)
    print("  " + ", ".join(cols), file=sys.stderr)


def _output_fields(sharded: bool) -> List[str]:
    """The score table's columns, in the order they are written.

    `in_window` sits among the pair attributes rather than among the
    score columns because it describes the pair -- whether the
    enhancer falls inside the model's input window -- and not any
    reduction of the map. Readers key columns by name, so a table that
    lacks the column means "unknown", never False.

    `_gene_rank` rides only on a shard's partial table, so an
    unsharded run writes the same columns whether or not sharding
    exists.
    """
    fields = ["screen", "gene", "enhancer", "chrom", "start", "end",
              "distance_to_tss", "in_window", "score"]
    if FREE_GRID:
        fields += list(free_grid_columns())
    if MULTI_CHANNEL:
        fields += list(multi_mix_columns())
    if sharded:
        return ["_gene_rank"] + fields
    return fields


def _runtime_record(args: argparse.Namespace,
                    backend: Backend) -> Dict[str, Any]:
    """What the manifest records about how the run was invoked.

    `--backend synthetic` builds its pairs from `toy_cohort()` and
    never calls `load_screen`, so neither the anchor table nor the
    FASTA is read. Recording their paths anyway would leave the
    manifest asserting an input that never reached the computation.
    """
    used_cohort = args.backend != "synthetic"
    runtime = {"backend": args.backend,
               "fasta": args.fasta if used_cohort else None,
               "anchors": args.anchors if used_cohort else None,
               "cohort": ("screen table" if used_cohort else
                          "toy_cohort() -- anchors/fasta NOT read"),
               "device": args.device,
               # the frame guard's certificate for this run: the probe's
               # report and the agreement gate's verdict
               "frame_guard": dict(FRAME_REPORT)}
    if hasattr(backend, "provenance"):
        runtime.update(backend.provenance())
    return runtime


def _write_ism_draws(path: str, theta: Dict[str, Any]) -> None:
    """Write the run's raw per-shuffle ISM differences to `path`."""
    if theta["method"] != "ism":
        raise ConfigError(
            f"--ism-draws is meaningless for method={theta['method']!r}"
            ": only the ISM path produces per-shuffle draws. Refusing "
            "rather than writing an empty file, which would look like "
            "a run that recorded nothing.")
    if not ISM_DRAWS:
        raise ConfigError(
            "--ism-draws was requested but no draws were recorded. "
            "Every pair was out-of-window, or ism_combine was not "
            "reached -- either way the run did not do what the flag "
            "assumes.")
    shp = ISM_DRAWS[0]["shape"]
    if any(r["shape"] != shp for r in ISM_DRAWS):
        raise ConfigError(
            "per-pair draw shapes differ; the npz would be ragged")
    np.savez_compressed(
        path,
        gene=np.array([r["gene"] for r in ISM_DRAWS]),
        enhancer=np.array([r["enhancer"] for r in ISM_DRAWS]),
        draws=np.stack([r["draws"] for r in ISM_DRAWS]),
        wt=np.stack([r["wt"] for r in ISM_DRAWS]),
        # (shuffle, tss, replicate, strand[, track]). Stored so a
        # reader can reshape without guessing the axis order.
        draw_shape=np.array(shp),
        ism_shuffles=np.array(theta["ism_shuffles"]),
        ism_mode=np.array(theta["ism_mode"]),
        ism_statistic_used=np.array(theta["ism_statistic"]),
        ism_inner_mean_used=np.array(list(theta["ism_inner_mean"]),
                                     dtype=object),
        seed=np.array(theta["seed"]))
    print(f"wrote {len(ISM_DRAWS):,} pairs of raw ISM draws to {path}",
          file=sys.stderr)


def _packages_hint(model: str, exc: BaseException) -> str:
    """What to install when a backend's import fails."""
    return (f"{exc}. The {model} backend needs its packages: pip install "
            f"-r requirements-{model}.txt, in a virtual environment of its "
            "own")


def _first_run_checks(args: argparse.Namespace,
                      theta: Dict[str, Any]) -> None:
    """Say what a real-backend run is missing before a model is loaded.

    Each of these would otherwise surface after the weights have loaded,
    or as an exception from inside torch.
    """
    if not args.fasta:
        raise ConfigError("no reference genome. Pass --fasta <hg38.fa>")
    if not os.path.exists(args.fasta):
        raise ConfigError(f"reference FASTA not found: {args.fasta}")
    try:
        import torch
    except ImportError as exc:
        raise ConfigError(_packages_hint(theta["model"], exc)) from None
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise ConfigError(f"--device {args.device}: no CUDA GPU is "
                          "available. Pass --device cpu to run on the CPU")


def _write_table(rows: List[Dict[str, Any]], fields: Sequence[str],
                 path: Optional[str]) -> None:
    """Write the score table as TSV to `path`, or to stdout."""
    sink = open(path, "w", newline="") if path else sys.stdout
    try:
        wr = csv.DictWriter(sink, fieldnames=fields, delimiter="\t",
                            lineterminator="\n")
        wr.writeheader()
        wr.writerows(rows)
    finally:
        if path:
            sink.close()


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Parse the command line, score one arm, and write its table."""
    ap = _build_parser()
    args = ap.parse_args(argv)
    if args.free_mean_subtract:
        args.free_grid = True
    _apply_scope_flags(args, ap)

    if args.self_test:
        return self_test()
    if not args.config:
        ap.error("--config is required (or use --self-test)")

    theta = fill_defaults(
        apply_overrides(load_config(args.config), args.set))
    warns = validate(theta)
    if args.multi_channels:
        _configure_multi_channels(theta, args.free_mean_subtract)
    rho = resolve(theta)
    rho.warnings = warns

    if args.print_config:
        print(json.dumps(manifest(theta, rho, RunStats()), indent=2,
                         sort_keys=True))
        for w in warns:
            print(f"WARNING: {w}", file=sys.stderr)
        return 0

    # Build and check the backend before reading the cohort. Both steps
    # are cheap next to load_screen (Gasperini is 78k rows) and to the
    # run itself, so a backend that cannot serve theta["method"] says
    # so before either.
    if args.backend == "real":
        _first_run_checks(args, theta)
    try:
        backend = make_backend(theta, args.backend, device=args.device,
                               fasta=args.fasta)
    except ImportError as exc:
        raise ConfigError(_packages_hint(theta["model"], exc)) from exc
    check_backend(theta, backend)

    if args.backend == "synthetic":
        pairs = toy_cohort()
    else:
        set_reference(args.fasta)
        pairs = load_screen(theta, anchors_path=args.anchors)

    if args.free_grid:
        _configure_free_grid(theta, args.free_mean_subtract)

    rows, stats = run_experiment(pairs, theta, backend)
    fields = _output_fields(bool(stats.shard))
    if not stats.shard:
        # `_gene_rank` is a shard's merge key and nothing else, so an
        # unsharded table does not carry it.
        for r in rows:
            r.pop("_gene_rank", None)
    runtime = _runtime_record(args, backend)
    if args.ism_draws is not None:
        _write_ism_draws(args.ism_draws, theta)
    _write_table(rows, fields, args.out)

    # The manifest is an auxiliary artefact, so it is written after the
    # score table: a failure here cannot cost the run its primary
    # output.
    if args.manifest:
        with open(args.manifest, "w") as fh:
            json.dump(manifest(theta, rho, stats, runtime), fh,
                      indent=2, sort_keys=True)
    for w in warns:
        print(f"WARNING: {w}", file=sys.stderr)
    print(f"scored {stats.pairs_scored} pairs, dropped "
          f"{stats.pairs_dropped}, {stats.gpu_calls} model calls, "
          f"{stats.cache_hits} cache hits", file=sys.stderr)
    return 0


# ============================================================================
# 8. Screens and annotations
# ============================================================================

#: screen -> (pair table, the column its genes are keyed on).
#:
#: `gasperini2019-linder` is deliberately a separate identifier from
#: `gasperini2019` even though both come from GSE120861: the two papers
#: filter that source differently -- Linder keeps the rows with no
#: empirical p-value as negatives and applies no promoter filter, Avsec
#: does the reverse -- so one table cannot serve both without scoring
#: one paper's protocol on the other's cohort.
#:
#: `gschwind` keys on `gene_id` and not on `gene_symbol` because that is
#: the join the paper itself describes, and because it has to be: 465 of
#: the source table's 10,356 rows carry a symbol GENCODE has since renamed (H1FX,
#: C19orf43, the whole HIST1 and ATP5 families), so a symbol join would
#: quietly lose 88 genes where the id join loses two.
SCREEN_TABLES = {
    "fulco2019": ("inputs/fulco2019.hg38.tsv", "gene_symbol"),
    "gasperini2019": ("inputs/gasperini2019.hg38.tsv", "gene_id"),
    "flowfish-plus": ("inputs/flowfish-plus.hg38.tsv", "gene_symbol"),
    "gasperini2019-linder": ("inputs/gasperini2019-linder.hg38.tsv",
                             "gene_id"),
    # The Linder et al. (2025) authors' own pair tables, kept as separate
    # identifiers so the two reconstructions above stay exactly what the
    # paper's reproduction scored. Keyed on `gene_id`, resolved from the
    # supplied annotation by the authors' own symbol rule; meant to run
    # under `gencode: linder2025`, which is that annotation.
    "flowfish-plus-authors": ("inputs/flowfish-plus-authors.hg38.tsv",
                              "gene_id"),
    "gasperini2019-linder-authors": (
        "inputs/gasperini2019-linder-authors.hg38.tsv", "gene_id"),
    "gschwind": ("inputs/gschwind.hg38.tsv", "gene_id"),
}

#: GENCODE renamed these; the screens' tables carry the older symbols.
#: Without the alias the gene has no GENCODE record and drops out of a
#: 30- or 61-gene cohort. C19orf43 -> TRIR is a v32-era rename, while
#: H1FX -> H1-10 arrived with the histone-gene renaming and so bites
#: only the v41 cohorts.
SYMBOL_ALIASES = {"C19orf43": "TRIR", "H1FX": "H1-10"}

#: Retired-to-current gene-ID map, for screens keyed on `gene_id`.
#:
#: GENCODE occasionally retires a gene's stable ID and reissues the same
#: locus under a new one, which breaks an ID join even though the gene
#: is still annotated. `SYMBOL_ALIASES` cannot help there, because an
#: ID-keyed screen is a direct `by_id` lookup with no symbol involved.
#:
#: Each entry is a claim that two IDs name the same gene, so each must
#: be justified by coordinates rather than by symbol agreement alone.
GENE_ID_ALIASES = {
    # PAXX (a.k.a. C9orf142), v46 ENSG00000148362 -> v50
    # ENSG00000310560. Same locus: the screen's TSS for the old ID is
    # chr9:136,992,416 and v50's gene spans
    # chr9:136,992,400-136,993,381 (+), 16 bp apart, same symbol, same
    # chromosome. An ID reissue, not a different gene.
    "ENSG00000148362": "ENSG00000310560",
    # LINC01033, v46 ENSG00000249069 -> v50 ENSG00000293676: a GENCODE
    # two-gene merge, evidenced at transcript level rather than on the
    # coordinate match and shared symbol that carry PAXX above. The
    # merged locus loses the symbol, which is why a symbol lookup finds
    # nothing, but v50's span chr5:54,310,711-54,415,238 is the exact
    # union of the two v46 genes, 19 of the 20 v46 transcripts survive
    # with not one start coordinate moved, and 39 of 41 v46 exon
    # intervals are present exactly.
    #
    # Judge a merge on the transcript TSS set, not on the gene-level 5'
    # end: here that 5' end is the merge partner's promoter ~10 kb
    # away, while `anchor=each-tss` and `tss_major` read `all_tss`,
    # whose nearest member is 28 bp from the screen's TSS in both
    # releases.
    #
    # The alias restores 2 pairs, both positives, so the v50 cohort is
    # 10,353 / 471, matching the denominator of every published bar it
    # is compared against. It moves no scored number under
    # `anchor: single-tss`, `tss_major: false`, `region: tss`, where
    # every affected quantity is inert; the `gene-midpoint` and
    # `region: exons` arms do see the merged locus, at TSS count
    # 16 -> 23, exonic bp 3,566 -> 5,938 (+67%) and a centroid
    # 26,728 bp away.
    "ENSG00000249069": "ENSG00000293676",
}

#: Genes a release genuinely drops, with the reason. Empty, and
#: deliberately kept: a release can retire a gene with no successor at
#: all, and then dropping it explicitly, with the reason printed, beats
#: both failing the run and silently shrinking the cohort. `load_screen`
#: reports every entry it applies, so this is not a silent filter.
#:
#: The bar for an entry is that no successor can be identified honestly,
#: and a merely nearby gene is not a successor. Aliasing to one would
#: leave `provided_tss` alone, since the backbone reads the screen's own
#: column, while substituting a different gene's TSS set and exon set,
#: which `anchor=each-tss` and `region=exons` do read. Dropping a pair
#: is honest; re-anchoring an arm to a merged locus is not.
GENE_ID_RETIRED: Dict[str, str] = {}


def _open_data(path: str):
    """A text handle on `path`, or on `path + ".gz"` when only the gzipped
    copy exists (the repository ships the annotations gzipped); None when
    neither exists."""
    if os.path.exists(path):
        return gzip.open(path, "rt") if path.endswith(".gz") else open(path)
    if not path.endswith(".gz") and os.path.exists(path + ".gz"):
        return gzip.open(path + ".gz", "rt")
    return None


def _load_gencode_defs(version: str = "v32") -> Dict[str, Any]:
    """Transcript-5'-end definitions for one GENCODE release.

    Parameterised by release because Avsec 2021 names v32 and Linder
    2025 names v41, and 7.2% of the genes they share moved their 5' TSS
    between the two. Serving v32 annotations to a v41 config would
    re-anchor a twentieth of every cohort, so `Gene.gencode` carries the
    release forward and `build_pair_table` refuses a mismatch rather
    than scoring it.
    """
    path = os.path.join(_HERE, "data", f"gencode_{version}_tssdefs.json")
    fh = _open_data(path)
    if fh is None:
        raise ConfigError(f"missing {path} (and {path}.gz)")
    with fh:
        return json.load(fh)


def _load_exon_spans(version: str,
                     source: str = "all-transcripts") -> Dict[str, Any]:
    """gene_id (stripped) -> merged half-open exonic intervals.

    Only two settings read `Gene.exons`: `region = exons`, the log-sum
    readout across a gene's exons, and the
    `pair_filters.require_gene_in_annotation` filter. Both are Avsec
    2026's, and the Linder et al. (2025) configs read the spans of the
    annotation the authors supplied. Returns {} when no spans ship for
    an annotation, which is the right answer for v32: every Avsec 2021
    config is `region = tss` with no annotation filter. `source` is the
    exception, since it exists to distinguish two transcript sets.
    """
    stem = f"gencode_{version}" + ("_basic_nort"
                                   if source == "basic-nort" else "")
    path = os.path.join(_HERE, "data", f"{stem}_exon_spans.json")
    fh = _open_data(path)
    if fh is None:
        if source == "basic-nort":
            # Refuse rather than fall back to a different transcript
            # set: the point of the parameter is that the two give
            # different scalars.
            raise ConfigError(
                f"exon_source=basic-nort needs {path} (or {path}.gz), "
                "which is absent. Only the v41 basic-nort spans ship")
        return {}
    with fh:
        return json.load(fh)


def _load_anchor_table(
        path: Optional[str]) -> Tuple[Dict[str, int], Dict[str, str]]:
    """gene key -> precomputed MAJOR-TSS, plus the `#!` provenance line.

    Returning the provenance is the point. MAJOR-TSS depends only on
    (model, gencode, tss_major_assay, cell_type), which is what makes it
    cacheable across experiments: the Gasperini cohort alone is 82,979
    forward passes. But anchors are keyed by GENE, so a file from the
    wrong model joins perfectly and then scores silently against another
    model's argmax, which is why the header is parsed out for
    `validate_anchors` instead of being skipped with the comments.
    """
    if not path:
        return {}, {}
    fh = _open_data(path)
    if fh is None:
        raise ConfigError(f"anchor table not found: {path}")
    out: Dict[str, int] = {}
    prov: Dict[str, str] = {}
    with fh:
        for line in fh:
            if line.startswith("#!"):
                for tok in line[2:].split():
                    if "=" in tok:
                        k, v = tok.split("=", 1)
                        prov[k.strip()] = v.strip()
                continue
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 3 or f[0] == "gene_key":
                continue
            out[f[0]] = int(f[2])
    return out, prov


#: Anchor-compatibility warnings from the most recent `load_screen`, on
#: their way to the manifest: a guard that only prints leaves no durable
#: record of having fired. Module-level rather than a return value,
#: because `load_screen` runs before `run_experiment` and widening
#: either signature would touch every caller for what is bookkeeping.
#: Cleared per load, so a second load in one process cannot inherit the
#: first one's warnings.
ANCHOR_WARNINGS: List[str] = []


def validate_anchors(prov: Dict[str, str], theta: Dict[str, Any],
                     path: str) -> List[str]:
    """Refuse an anchor file that was not computed for this theta.

    Returns one warning per compatibility question the file's provenance
    leaves open.
    """
    warns: List[str] = []
    name = os.path.basename(path)
    if not prov:
        warns.append(
            f"anchor table {name} has no `#!` provenance line, so "
            f"compatibility with model={theta['model']} / "
            f"gencode={theta['gencode']} / "
            f"{theta['tss_major_assay']}:{theta['cell_type']} is "
            "UNVERIFIED")
        return warns
    # MAJOR-TSS's four dependencies. `model` is a checkpoint string, so
    # match by substring.
    if theta["model"] not in prov.get("model", "").lower():
        raise ConfigError(
            f"anchor/config model mismatch: {name} was computed with "
            f"model={prov.get('model')!r} but θ.model="
            f"{theta['model']!r}. MAJOR-TSS is model-specific; these "
            "anchors would join by gene and score silently wrong.")
    # The anchor's `cell_type` names the cell type the major TSS was
    # argmaxed over, which is not always the one being read out: a
    # cell-type-agnostic run reads out `all`, while the anchors it
    # should use -- the ones the published agnostic bars were computed
    # with -- are the k562 ones. So compare against the anchor's own
    # concept rather than against theta["cell_type"].
    expect = {"gencode": theta["gencode"],
              "tss_major_assay": theta["tss_major_assay"],
              "cell_type": major_tss_cell_type(theta)}
    for key, want_v in expect.items():
        want, got = str(want_v).lower(), prov.get(key, "").lower()
        if not got:
            # An absent field is unknown, which is not the same as
            # matching, so it warns rather than passing.
            warns.append(f"anchor table {name} records no {key}; "
                         f"compatibility with θ.{key}={want!r} is "
                         "UNVERIFIED")
            continue
        if got != want:
            raise ConfigError(f"anchor/config {key} mismatch: {name} "
                              f"has {key}={got!r}, θ has {want!r}")
    scr = prov.get("screen", "")
    if scr and not str(theta["screen"]).startswith(scr):
        warns.append(f"anchor table was exported for screen={scr!r} "
                     f"but θ.screen={theta['screen']!r}; gene keys may "
                     "not join")
    # Three states, not two: recorded and matching, recorded and
    # different, and not recorded at all. An anchor set whose
    # provenance lacks the field says `unrecorded`, which means
    # unknown rather than wrong, so it earns a warning instead of an
    # assumption either way.
    #
    # Compared against what this run will actually use rather than
    # against a hard-coded `true`: an anchor set computed with
    # use_tf_gamma=false is legitimate, and a guard insisting on `true`
    # would fire on exactly the run that is correct.
    tfg = prov.get("use_tf_gamma", "").lower()
    want = str(FROZEN_USE_TF_GAMMA).lower()
    if tfg not in ("true", "false"):
        warns.append(
            f"anchor table {name} records "
            f"use_tf_gamma={tfg or 'nothing'!r}, so whether it matches "
            "the scoring model is UNKNOWN. On Gasperini the flag moves "
            "17.3% of argmaxes (median 34 bp, 3.7% by a whole output "
            "bin); re-export the anchors before trusting the bars")
    elif tfg != want:
        warns.append(
            f"anchors were computed with use_tf_gamma={tfg} while this "
            f"run scores with {want} -- anchor and score come from "
            "slightly different models")
    return warns


def _screen_table_path(screen: str) -> Tuple[str, str]:
    """The pair table for one screen, and its gene-key column."""
    if screen not in SCREEN_TABLES:
        raise ConfigError(
            f"no loader for screen={screen!r}. Known: "
            f"{sorted(SCREEN_TABLES)}. Use --backend synthetic to "
            "exercise the algorithm without data.")
    rel, gene_key = SCREEN_TABLES[screen]
    path = os.path.join(_HERE, rel)
    # Only the compressed table is checked in, so a fresh clone has
    # `<screen>.hg38.tsv.gz` and no `<screen>.hg38.tsv`. Falling back to
    # the compressed copy is what makes a fresh checkout runnable with
    # no preparation step; gzip is lossless, so the reader below sees
    # the same bytes either way.
    if not os.path.exists(path) and os.path.exists(path + ".gz"):
        path += ".gz"
    if not os.path.exists(path):
        raise ConfigError(f"missing {path} (and {path}.gz); the screen "
                          "tables ship in inputs/")
    return path, gene_key


def _annotation_index(gencode: str, exon_source: str) -> Tuple[
        Dict[str, Any], Dict[str, List[Any]], Dict[str, Any]]:
    """One GENCODE release, indexed by id and by symbol, plus exons."""
    defs = _load_gencode_defs(gencode)
    by_symbol: Dict[str, List[Any]] = {}
    for gid, d in defs.items():
        # Stamp each record with its own key so a symbol hit can still
        # find the gene's exons, which are indexed by gene id. `defs` is
        # re-read from JSON per call, so the mutation stays local.
        d["_gid"] = gid.split(".")[0]
        by_symbol.setdefault(d["name"], []).append(d)
    by_id = {gid.split(".")[0]: d for gid, d in defs.items()}
    return by_id, by_symbol, _load_exon_spans(gencode, exon_source)


def _record_by_id(stem: str, by_id: Dict[str, Any]) -> Tuple[Any, str]:
    """A gene record by stable ID, following `GENE_ID_ALIASES`.

    Returns the record and the reissued ID it was found under, or ""
    when the screen's own ID was still current. That ID is tried first,
    so where it is current the alias table is never consulted and stays
    correct across releases.
    """
    d = by_id.get(stem)
    if d is not None:
        return d, ""
    new = GENE_ID_ALIASES.get(stem, "")
    d = by_id.get(new) if new else None
    return d, (new if d is not None else "")


def _record_by_symbol(sym: str, chrom: str,
                      by_symbol: Dict[str, List[Any]]) -> Any:
    """A gene record by symbol, following `SYMBOL_ALIASES`.

    The screen's own symbol is tried first and the alias only as a
    fallback, rather than letting the alias override. An alias is
    release-specific: H1FX -> H1-10 is correct for v41 but wrong for
    v32, where the gene is still H1FX, so an unconditional override
    would break every v32 cohort containing it. Fallback ordering
    handles both releases without a version-keyed table.

    Among the candidates, prefer the right chromosome, then a
    protein-coding record, then the one annotating the most TSSs.
    """
    cands = (by_symbol.get(sym, [])
             or by_symbol.get(SYMBOL_ALIASES.get(sym, sym), []))
    cands = [c for c in cands if c["chrom"] == chrom] or cands
    pc = [c for c in cands if c.get("type") == "protein_coding"] or cands
    if not pc:
        return None
    return max(pc, key=lambda c: len(c.get("all_tss") or ()))


def _gene_from_row(key: str, row: Dict[str, str], rec: Dict[str, Any],
                   spans: Dict[str, Any], anchor: Optional[int],
                   gencode: str) -> Gene:
    """Assemble one cohort gene from its screen row and GENCODE record.

    `provided_tss` comes from the screen's own `tss` column, so
    distances stay exactly those the table was verified against; the
    model's anchor rides separately in `anchor_tss`. `exons` stays empty
    unless the release's exon-span artefact exists, which is the right
    answer for the v32 and v41 screens: every Avsec 2021 and Linder 2025
    arm is region=tss and never reads it, while Avsec 2026 is
    region=exons and needs the real intervals.
    """
    ex = spans.get(rec.get("_gid") or key.split(".")[0]) or ()
    return Gene(gid=key, chrom=row["chrom"],
                strand=rec.get("strand", "+"),
                provided_tss=int(row["tss"]),
                gencode_tss=tuple(sorted(rec.get("all_tss")
                                         or (int(row["tss"]),))),
                exons=tuple((int(b), int(e)) for b, e in ex),
                anchor_tss=anchor, gencode=gencode)


def _note_cohort_changes(screen: str, gencode: str,
                         reissued: Dict[str, str],
                         retired: Dict[str, str]) -> None:
    """Announce the gene-ID aliases and drops one load applied.

    An applied alias or an applied drop changes the cohort, so neither
    may be silent. Printed to stderr because stdout can be the score
    table.
    """
    for old, new in sorted(reissued.items()):
        print(f"note: {screen}/GENCODE {gencode}: gene id {old} is "
              f"retired; using its reissued id {new} (GENE_ID_ALIASES)",
              file=sys.stderr)
    for old, why in sorted(retired.items()):
        print(f"note: {screen}/GENCODE {gencode}: DROPPED gene {old} "
              f"-- {why} (GENE_ID_RETIRED)", file=sys.stderr)


def load_screen(theta: Dict[str, Any], anchors_path: Optional[str] = None
                ) -> List[Tuple[Enhancer, Gene]]:
    """Read this theta's screen table into (Enhancer, Gene) pairs.

    Fulco's `gene_id` column is empty in all 4,777 rows, so its genes
    have to be keyed by `gene_symbol` (30 genes); Gasperini is the
    reverse, with `gene_id` filled throughout. `SCREEN_TABLES` records
    which column each screen joins on.
    """
    screen, gencode = theta["screen"], theta["gencode"]
    path, gene_key = _screen_table_path(screen)
    by_id, by_symbol, spans = _annotation_index(gencode,
                                                theta["exon_source"])
    anchors, anchor_prov = _load_anchor_table(anchors_path)
    ANCHOR_WARNINGS.clear()
    if anchors_path:
        # Refuse anchors computed for a different (model, gencode,
        # assay, cell_type). Those four are MAJOR-TSS's only
        # dependencies, which is both what makes the computation
        # reusable across experiments and what has to be checked before
        # reusing it.
        for w in validate_anchors(anchor_prov, theta, anchors_path):
            print(f"  WARN {w}")
            ANCHOR_WARNINGS.append(w)

    genes: Dict[str, Gene] = {}
    pairs: List[Tuple[Enhancer, Gene]] = []
    missing_gencode: set = set()
    missing_anchor: set = set()
    reissued: Dict[str, str] = {}      # old id -> current id used
    retired_hits: Dict[str, str] = {}  # old id -> why it is dropped

    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            key = r[gene_key]
            if not key:
                raise ConfigError(
                    f"{screen}: empty {gene_key!r} in {path}. Fulco "
                    "keys on gene_symbol, Gasperini on gene_id; "
                    "SCREEN_TABLES picks per screen.")
            if key not in genes:
                if gene_key == "gene_id":
                    stem = key.split(".")[0]
                    d, new = _record_by_id(stem, by_id)
                    if new:
                        reissued[stem] = new
                    if d is None and stem in GENE_ID_RETIRED:
                        retired_hits[stem] = GENE_ID_RETIRED[stem]
                        continue
                else:
                    d = _record_by_symbol(key, r["chrom"], by_symbol)
                if d is None:
                    missing_gencode.add(key)
                    continue
                a = anchors.get(key)
                if anchors and a is None:
                    missing_anchor.add(key)
                genes[key] = _gene_from_row(key, r, d, spans, a, gencode)
            pairs.append((Enhancer(chrom=r["chrom"],
                                   start=int(r["enhancer_start"]),
                                   end=int(r["enhancer_end"]),
                                   eid=r["enhancer_id"]),
                          genes[key]))

    _note_cohort_changes(screen, gencode, reissued, retired_hits)
    if missing_gencode:
        remedy = ("Add the retired->current id to GENE_ID_ALIASES "
                  "(verify the two ids name the same LOCUS, not merely "
                  "the same symbol), or, if the gene has no honest "
                  "successor in this release, record it in "
                  "GENE_ID_RETIRED with the reason"
                  if gene_key == "gene_id" else
                  "Add an alias to SYMBOL_ALIASES")
        raise ConfigError(
            f"{screen}: {len(missing_gencode)} genes absent from "
            f"GENCODE {gencode} (e.g. {sorted(missing_gencode)[:5]}). "
            f"{remedy} -- rather than letting them drop, which would "
            "silently shrink the cohort.")
    if missing_anchor:
        raise ConfigError(
            f"{screen}: --anchors supplied but {len(missing_anchor)} of "
            f"{len(genes)} genes have no precomputed MAJOR-TSS (e.g. "
            f"{sorted(missing_anchor)[:5]}). Falling back to the "
            "annotation TSS for those would score part of the cohort "
            "under a different protocol; extend the anchor table or "
            "drop those genes explicitly.")
    return pairs


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ConfigError as exc:
        # a wrong setting or a missing input is a message, not a traceback
        raise SystemExit(f"error: {exc}") from None
