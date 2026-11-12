"""
backend_alphagenome.py — the AlphaGenome backend for `score_eg.py`.

Companion to `backend_enformer.py` and `backend_borzoi.py`, with the
same division of labour: everything here is torch plumbing, the
specification stays in `score_eg.py`. Runs the `alphagenome_pytorch`
port, pinned in `requirements-alphagenome.txt`, which also says how to
produce the weights. Implements all three `Backend` methods —

    forward(x, replicate)              -> (1_048_576, n_tracks)
    scalar_gradient(x, replicate, fn)  -> (1_048_576, 4)
    attention(x, replicate, how)       -> (8_192, 8_192)

Attention is on the transformer's own 128 bp token grid, not on the 1 bp
output grid: an output position maps to token `pos // 128`. No reported
arm reads it, so the shape assertion in `attention` is the tripwire.

Five things here are easy to get silently wrong. Each is asserted or
derived rather than assumed.

**1. This file's track indices are ABSOLUTE, the metadata's are
POSITIONAL.** The RNA_SEQ head is 768 tracks at 1 bp over 1 Mb, 3.2 GB
for one forward pass, and ISM needs a pass per draw, so `forward`
returns only the requested columns and `ALPHAGENOME_META.tracks` indexes
*that* array. `ABSOLUTE_TRACKS` holds the head-space indices, and
`_check_track_contract` asserts the two agree at construction. A
mismatch would read the wrong track and raise nothing, because the gene
scalar does a bare `y[:, tracks]`.

**2. `scalar_gradient` returns the RAW 4-channel gradient, not
`grad * input`.** `score_eg.py` mean-subtracts across the four channels
*before* taking the reference base, so all four must be present and
unmultiplied; `grad * input` has already zeroed three of them.

**3. The gradient path never builds the full 1 bp head.** Running the
whole head at 1 Mb costs a (1, 1_048_576, 768) activation and an equal
cotangent, which is what does not fit. It is not needed: the head is
position-wise (`MultiOrganismConv1d` is a 1x1 conv, `heads.py:161-199`;
softplus and `predictions_scaling` are elementwise), so restricting it
to the readout POSITIONS and to the requested CHANNELS is exact rather
than approximate. `_gathered_readout` and `_sliced_head` do that, and
each states its own licence.

**4. The gene scalar exists twice, so it is gated on every call.**
`scalar_fn` is numpy; the same reduction is mirrored in torch so that
autograd can supply dU/dy in one pass. The two values are compared on
every gradient call, to `SCALAR_GATE_RTOL`.

**5. Readout arithmetic is fp32 whatever the trunk computed in.** Under
`bf16` the trunk runs in bfloat16, but summing ~12,000 bf16 values would
lose more than the model's own numerics do, so the gathered readout is
upcast before any reduction -- which is what the port's own `predict()`
does to its outputs.
"""

from __future__ import annotations

import os

# Must be set before the CUDA caching allocator initialises. The 1 Mb
# activations are very large and short-lived, so the allocator otherwise
# strands freed blocks and turns a fitting run into a fragmentation OOM.
# `backend_enformer.py` carries the same line for the same reason.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from typing import Any, Callable, Dict, FrozenSet, Optional, Tuple

import numpy as np

import score_eg as S

#: Head-space (absolute) track indices per (assay, cell_type). The
#: matching POSITIONAL blocks, the strands, and the provenance of every
#: index below all live in `score_eg.ALPHAGENOME_META`.
#:
#: Order is load-bearing: strands are matched to these entries
#: positionally, so `(+, -)` here and `{0: "+", 1: "-"}` there have to
#: move together, or a '-' gene reads the sense track and nothing raises.
ABSOLUTE_TRACKS: Dict[Tuple[str, str], Tuple[int, ...]] = {
    # 119 polyA+ (+)  120 total (+)  390 polyA+ (-)  391 total (-)
    # 571 polyA+ (.)
    ("rnaseq", "k562"): (119, 120, 390, 391, 571),
    ("cage", "k562"): (150, 423),      # the only two K562 rows of 640
    ("procap", "k562"): (2, 8),        # the only two K562 rows of 128
    # The CAGE cell-type sets. `_alphagenome_cage_tracks` fixes both the
    # expansion rule -- sample k is the pair (k, k + 273), a constant
    # offset rather than Borzoi's interleave -- and the order the strand
    # map is built against, so the two sides cannot drift apart.
    ("cage", "all"): S._alphagenome_cage_tracks(
        range(S.ALPHAGENOME_CAGE_N_SAMPLES)),
    ("cage", "not-k562"): S._alphagenome_cage_tracks(
        S.ALPHAGENOME_CAGE_NOT_K562_ORDINALS),
    **{("cage", name): S._alphagenome_cage_tracks([k])
       for name, k in S.ALPHAGENOME_CAGE_RAND1_ORDINALS.items()},
}

ASSAY_HEAD = {"rnaseq": "rna_seq", "cage": "cage", "procap": "procap"}


def _patched_mha_forward(self, x, attention_bias, compute_dtype=None):
    """`MHABlock.forward` plus one line that keeps the attention.

    Verbatim otherwise: rope, the 5.0 logit soft-cap and the
    bf16-matmul/fp32-accumulate precision dance are all load-bearing and
    are reproduced exactly. Mirrors `backend_enformer._patched_attention`
    and `backend_borzoi._patched_attention`; the signature differs
    because this port threads `attention_bias` and `compute_dtype`
    through.
    """
    import math
    import torch
    import torch.nn.functional as F

    B, n, _ = x.shape
    if compute_dtype is None:
        compute_dtype = x.dtype
    x = x.to(compute_dtype)
    h = self.norm(x)

    q = self.norm_q(self.q_proj(h).view(B, n, 8, 128))
    k = self.norm_k(self.k_proj(h).view(B, n, 1, 128))
    v = self.norm_v(self.v_proj(h).view(B, n, 1, 192))

    from alphagenome_pytorch.attention import apply_rope
    q = apply_rope(q, inplace=True)
    k = apply_rope(k, inplace=True)

    q_t = q.permute(0, 2, 1, 3)
    k_t = k.permute(0, 2, 1, 3)

    att = torch.matmul(q_t, k_t.transpose(-2, -1)).float()
    att = att / math.sqrt(128.0)
    if attention_bias is not None:
        att = att + attention_bias.float()
    logits_soft_cap = 5.0
    att = torch.tanh(att / logits_soft_cap) * logits_soft_cap

    attn_weights = F.softmax(att, dim=-1)

    # --- the one added line. Mean over heads HERE, so what is kept is
    # (n, n) per layer (268 MB fp32 at n = 8,192) rather than the
    # head-resolved (8, n, n) = 2.1 GB.
    if getattr(self, "_collect_attention", False):
        self._attn_head_mean = (
            attn_weights.detach().mean(dim=1)[0].float().cpu().numpy())

    v_t = v.permute(0, 2, 1, 3)
    y = torch.matmul(attn_weights.to(compute_dtype), v_t).float()
    y = y.to(compute_dtype)
    y = y.permute(0, 2, 1, 3).reshape(B, n, -1)
    y = self.linear_embedding(y)
    return self.final_norm(y)


#: Path to the converted AlphaGenome PyTorch checkpoint. There is no
#: default: set ALPHAGENOME_PT_WEIGHTS, or pass weights= explicitly. See
#: requirements-alphagenome.txt for how to produce the file.
DEFAULT_WEIGHTS = os.environ.get("ALPHAGENOME_PT_WEIGHTS")

#: Relative tolerance for the torch-vs-numpy gene-scalar gate. Not 1e-9:
#: the two sides reduce the same ~12,000 fp32 predictions in different
#: orders and accumulator widths, so the floor is fp32 round-off on a sum
#: of that length (~1e-7 relative). 1e-5 sits far above that and far
#: below every wiring error it exists to catch -- a wrong track set, a
#: dropped log, a mean-vs-sum swap and a missing strand mask each move
#: the scalar by >1%.
SCALAR_GATE_RTOL = 1e-5


class _SkipsOnHost(dict):
    """The encoder's U-Net skips, held in host RAM until popped.

    `SequenceDecoder.forward` does `intermediates.pop(f'bin_size_{n}')`
    and hands the result straight to an UpResBlock (`model.py:129-138`),
    so overriding `pop` is enough to make the transfer happen exactly
    when the tensor is needed and not before. Returning a CPU tensor
    would be a device mismatch inside the block; returning `.to(device)`
    is a differentiable op, so autograd sends the gradient back to the
    host tensor and on into the encoder.
    """

    def __init__(self, mapping, device):
        super().__init__(mapping)
        self._device = device

    def pop(self, key, *default):
        v = super().pop(key, *default)
        if hasattr(v, "to"):
            return v.to(self._device, non_blocking=True)
        return v


def _pin_fp32(torch) -> dict:
    """Turn TF32 OFF for convolutions and matmuls, and report what was found.

    Mirrors `backend_borzoi._pin_fp32`; read that docstring for the
    mechanism. torch ships `cudnn.allow_tf32 = True`, so on an Ampere or
    newer card every convolution would otherwise run with a 10-bit
    mantissa. AlphaGenome's measured exposure is small (about 1e-4 in the
    gene scalar, the attribution sums and the normaliser, with every
    ranking unchanged), much smaller than Enformer's, so the pin is for
    symmetry across the three trunks rather than for accuracy: Enformer
    and Borzoi are pinned, and an unpinned AlphaGenome would put a
    precision asymmetry into a three-way comparison.

    Two precision fields reach the manifest and they are different axes:
    `precision` is the dtype policy ('fp32' | 'bf16'); `precision_flags`
    is TF32, and it is this function's return value.
    """
    found = {"cudnn_allow_tf32_default": bool(torch.backends.cudnn.allow_tf32),
             "matmul_allow_tf32_default": bool(torch.backends.cuda.matmul.allow_tf32)}
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    found["tf32_pinned_off"] = True
    if torch.cuda.is_available():
        found["gpu_name"] = torch.cuda.get_device_name(0)
        found["gpu_capability"] = ".".join(str(v) for v in torch.cuda.get_device_capability(0))
        found["tf32_capable"] = torch.cuda.get_device_capability(0)[0] >= 8
    return found


class AlphaGenomeBackend(S.Backend):
    """AlphaGenome through the `alphagenome_pytorch` port."""

    IMPLEMENTS: FrozenSet[str] = frozenset(
        {"forward", "scalar_gradient", "attention"})

    capability_note = ("AlphaGenome attention is on a 128 bp token grid "
                       "(8,192 tokens), NOT the 1 bp output grid; map an "
                       "output position with pos // 128.")

    #: The transformer tower runs at 128 bp (`model.py:91`), so the token
    #: grid is `seq_len // 128` = 8,192 and an output position maps to
    #: `pos // 128`. The same grid Enformer and Borzoi attend on.
    ATTENTION_BIN = 128

    def __init__(self, meta, theta: Dict[str, Any], *,
                 weights: str = DEFAULT_WEIGHTS,
                 fasta: Optional[str] = None,
                 device: str = "cuda",
                 precision: Optional[str] = None):
        # torch is imported HERE, not at module scope, so `score_eg.py
        # --self-test` keeps running without torch and without a GPU.
        import torch
        from alphagenome_pytorch import AlphaGenome
        from alphagenome_pytorch.config import DtypePolicy

        self.meta = meta
        self._torch = torch

        self.key = (theta["assay"], theta["cell_type"])
        if self.key not in ABSOLUTE_TRACKS:
            raise S.ConfigError(
                "backend_alphagenome has no head-space indices for "
                f"{self.key}. Known: {sorted(ABSOLUTE_TRACKS)}.")
        self.head_name = ASSAY_HEAD[theta["assay"]]
        self.absolute_tracks = ABSOLUTE_TRACKS[self.key]
        self._check_track_contract()

        self.precision = (precision or os.environ.get("SCORE_EG_AG_PRECISION")
                          or "fp32")
        if self.precision not in ("bf16", "fp32"):
            raise S.ConfigError(
                f"precision must be bf16 or fp32, got {self.precision!r}")
        self._theta_scalar = self._freeze_scalar_theta(theta)

        self.weights = weights
        self.fasta = fasta
        self.device = device
        if not weights or not os.path.exists(weights):
            where = weights if weights else (
                "<unset>; set ALPHAGENOME_PT_WEIGHTS or pass weights=")
            raise S.ConfigError(
                f"no AlphaGenome PyTorch weights at {where}. Convert the "
                "JAX/Orbax checkpoint:\n"
                "  agt convert --input <alphagenome-all-folds> "
                "--output alphagenome.pth\n"
                "then set ALPHAGENOME_PT_WEIGHTS. Do NOT substitute a "
                "third-party mirror of the weights: the numerical "
                "agreement reported for this backend was established "
                "only against that conversion.")

        policy = (DtypePolicy.mixed_precision() if self.precision == "bf16"
                  else DtypePolicy.full_float32())
        # TF32 is a separate axis from `self.precision` (see _pin_fp32),
        # pinned at construction rather than import, as backend_borzoi
        # does, so a caller that imports early and constructs late is
        # still covered.
        self._precision = _pin_fp32(self._torch)
        # The resolved consequences of the dtype policy, named so that
        # neither `precision` key has to be decoded to read the manifest.
        self._dtype_policy = {
            "dtype_policy_name": type(policy).__name__,
            "dtype_policy_compute": str(getattr(policy, "compute_dtype", None)),
            "dtype_policy_autocast_enabled":
                str(getattr(policy, "compute_dtype", None)) != "torch.float32"}
        # Four ways to pay for the 1 Mb backward, set by
        # SCORE_EG_AG_OFFLOAD:
        #   skips  (default) gradient checkpointing on, and the seven
        #          U-Net skips parked in host RAM. ~7.4 GB of host.
        #   both   the same, plus save_on_cpu. Only the checkpoint block
        #          boundaries are staged, so host RAM and PCIe traffic
        #          both collapse and the card stops idling. Needs the
        #          determinism-check relaxation below.
        #   all    checkpointing off, save_on_cpu over every saved
        #          tensor. ~248 GB of host RAM per process, and slow.
        #   none   no offload of either kind. Does not fit at 1 Mb
        #          without the decoder crop below.
        #   auto   decided per gene from the decoder crop's length:
        #          `both` above SCORE_EG_AG_AUTO_MAX_CROP_BP, nothing
        #          below it. With the decoder cropped to the readout
        #          span, a typical gene's backward fits on a 48 GB card
        #          with no host offload, and offload is what makes
        #          `both` cost ~43 s per gene (save_on_cpu stages every
        #          checkpoint boundary through pageable host RAM). Only a
        #          gene whose readout spans most of the 1 Mb window
        #          still needs it.
        mode = os.environ.get("SCORE_EG_AG_OFFLOAD", "skips").lower()
        self.offload_auto = False
        if mode in ("1", "all"):
            self.offload, self.offload_skips = True, False
        elif mode in ("0", "none"):
            self.offload, self.offload_skips = False, False
        elif mode == "both":
            self.offload, self.offload_skips = True, True
        elif mode == "auto":
            self.offload, self.offload_skips = False, False
            self.offload_auto = True
        else:
            self.offload, self.offload_skips = False, True
        self.offload_mode = mode
        self.auto_max_crop_bp = int(
            os.environ.get("SCORE_EG_AG_AUTO_MAX_CROP_BP", "393216"))
        # DECODER CROP. `_gathered_readout` runs the 1 bp decoder only over
        # [min(readout) - halo, max(readout) + halo), 128 bp-aligned,
        # instead of the whole 1,048,576 bp. EXACT away from the crop
        # edges: the sequence decoder is seven up-blocks of 'same'-padded,
        # stride-1, undilated k=5 convolutions, a 2x repeat_interleave and
        # a pointwise skip projection, with inference-mode RMS batch norm
        # (per-channel constants) and an elementwise GELU. Its receptive
        # field is 2*(128+64+...+2) + 2*(64+...+1) = 762 bp per side, so a
        # 2,048 bp halo leaves every readout position identical to the
        # full decoder's up to the kernel-choice round-off cuDNN already
        # shows between sequence lengths (measured on real genes: 4e-7
        # relative). The encoder and transformer tower still run over the
        # whole window, since the readout's context is the point.
        # SCORE_EG_AG_DECODER_CROP=0 restores the full decoder.
        self.decoder_crop = os.environ.get("SCORE_EG_AG_DECODER_CROP",
                                           "1") != "0"
        self.crop_halo_bp = int(os.environ.get("SCORE_EG_AG_CROP_HALO_BP",
                                               "2048"))
        self._crop_stats = {"n": 0, "n_cropped": 0, "n_offloaded": 0,
                            "crop_bp_sum": 0}
        self._last_crop = None
        self.reclaim_before_backward = (
            os.environ.get("SCORE_EG_AG_RECLAIM", "1") == "1")
        # save_on_cpu and the port's non-reentrant checkpointing collide
        # out of the box, but the collision is narrow. torch compares the
        # metadata of tensors saved during the forward against those
        # saved during the recompute, position by position; save_on_cpu's
        # hook fires in both passes and reorders two *parameter* tensors
        # relative to each other, so the check trips on ORDER, not on
        # values. Relaxing that one check lets the two compose, which is
        # what takes the card from ~25% utilisation to compute-bound.
        #
        # This is a safety check being switched off, so `provenance()`
        # reports it, and `both` is only usable on a build whose
        # gradients have been checked against `all`.
        self.grad_checkpointing = (not self.offload) or self.offload_skips
        if (self.offload or self.offload_auto) and self.grad_checkpointing:
            import functools
            from torch.utils import checkpoint as ckpt
            import alphagenome_pytorch.model as agm
            if not getattr(agm, "_score_eg_ckpt_relaxed", False):
                agm.checkpoint = functools.partial(
                    ckpt.checkpoint, determinism_check="none")
                agm._score_eg_ckpt_relaxed = True
        self.model = AlphaGenome.from_pretrained(
            weights, dtype_policy=policy, device=device,
            gradient_checkpointing=self.grad_checkpointing)
        self.model.eval()
        # GRADIENTS W.R.T. THE INPUT ONLY. `.eval()` does not touch
        # requires_grad, so without this every backward also allocates a
        # .grad for all 450.5 M parameters -- 1.8 GB of buffers that are
        # computed, written and never read, on every gene. The
        # attribution score is d(gene scalar)/d(one-hot input); the
        # weights are frozen data.
        for p in self.model.parameters():
            p.requires_grad_(False)
        self._compute_dtype = self.model.dtype_policy.compute_dtype
        self._org = torch.zeros(1, dtype=torch.long, device=device)
        self._cols = torch.as_tensor(self.absolute_tracks,
                                     dtype=torch.long, device=device)

        self.gate_worst = 0.0
        self.n_forward = 0
        self.n_gradient = 0

    # -------------------------------------------------------------- setup
    def _check_track_contract(self) -> None:
        positional = self.meta.tracks[self.key]
        if tuple(positional) != tuple(range(len(self.absolute_tracks))):
            raise S.ConfigError(
                f"ALPHAGENOME_META.tracks[{self.key}] = {positional}, but "
                f"this backend returns {len(self.absolute_tracks)} columns "
                "positionally, so the metadata must be "
                f"{tuple(range(len(self.absolute_tracks)))}. Those indices "
                "are positions in the array `forward` returns, NOT "
                f"absolute head indices ({self.absolute_tracks}).")

    def _freeze_scalar_theta(self, theta: Dict[str, Any]) -> Dict[str, Any]:
        if theta["inverse_transform"]:
            raise S.ConfigError(
                "inverse_transform=true is not implementable on "
                "AlphaGenome: the head already returns experimental-data "
                "scale (predictions_scaling), which is why "
                "ALPHAGENOME_META.scalar_is_squashed is False.")
        for name, op in (("track_agg", theta["track_agg"]),):
            if op not in ("sum", "mean"):
                raise S.ConfigError(
                    f"{name}={op!r} is not expressible in this backend's "
                    "torch mirror. `sum` and `mean` are linear so they "
                    "become a weight vector; `max` would need its own "
                    "handling and no AlphaGenome arm uses it.")
        return {"track_agg": theta["track_agg"], "bin_agg": S.FROZEN_BIN_AGG,
                "log_scalar": bool(theta["log_scalar"]),
                "pseudocount": float(S.FROZEN_PSEUDOCOUNT),
                "log_base": self.meta.log_base}

    def _offload(self, force: Optional[bool] = None):
        """save_on_cpu for saved tensors, or a no-op when disabled.

        `force` overrides the constructed setting for one call, which is
        how the `auto` mode applies its per-gene choice.
        """
        import contextlib
        import torch
        if not (self.offload if force is None else force):
            return contextlib.nullcontext()
        # pin_memory=False deliberately. In `all` mode this stages every
        # saved activation in host RAM, ~248 GB for one 1 Mb backward.
        # Pinning that is page-locked memory the kernel cannot reclaim,
        # and the host, not the GPU, is the binding resource for this
        # arm: pageable staging is slower per transfer and survives.
        return torch.autograd.graph.save_on_cpu(pin_memory=False)

    def _autocast(self):
        return self._torch.autocast(
            "cuda" if self.device.startswith("cuda") else "cpu",
            dtype=self._compute_dtype,
            enabled=self._compute_dtype != self._torch.float32)

    def _put(self, x: np.ndarray):
        """(L, 4) float64 from `one_hot` -> (1, L, 4) float32 on device."""
        t = self._torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
        return t.unsqueeze(0).to(self.device)

    # ------------------------------------------------------------ forward
    def forward(self, x: np.ndarray, replicate: Any) -> np.ndarray:
        """x: (1_048_576, 4) one-hot -> (1_048_576, n_tracks)."""
        self._check_replicate(replicate)
        torch = self._torch
        with torch.no_grad():
            out = self.model.predict(self._put(x), organism_index=0,
                                     heads=[self.head_name], resolutions=[1])
            y = (out[self.head_name][1][0]
                 .index_select(1, self._cols).float().cpu().numpy())
        del out
        self.n_forward += 1
        n_tracks = len(self.meta.tracks[self.key])
        assert y.shape == (self.meta.n_bins, n_tracks), \
            (f"forward must return ({self.meta.n_bins}, {n_tracks}) -- the "
             f"columns ALPHAGENOME_META names, positionally -- got {y.shape}")
        return y

    # -------------------------------------------------------- sliced head
    def _gathered_readout(self, x_t, bins, offload_skips: Optional[bool] = None):
        """The forward pass, gathered to `bins` before the 1 bp stage.

        With `decoder_crop` the 1 bp decoder also never runs wider than
        the readout span plus a halo (see __init__). `offload_skips`
        overrides the constructed setting for one call (the `auto`
        offload mode).

        Replicates `AlphaGenome._compute_embeddings_ncl`
        (`model.py:618-666`) up to the decoder, then GATHERS the readout
        positions before `embedder_1bp` instead of after it. That is what
        makes the 1 Mb backward fit: `embeddings_1bp` is
        (1, 1536, 1_048_576) fp32 = 6.44 GB and its cotangent is another,
        so removing the pair is ~12.9 GB.

        EXACT, not an approximation, and the reason is narrow enough to
        state. `OutputEmbedder` (`embeddings.py:35-60`) is project_in ->
        (+ project_skip, upsampled) -> norm -> + organism -> gelu. Every
        step is position-local, and the one that looks as though it could
        not be -- `norm` -- is an inference-mode BatchNorm whose scale is
        `weight * rsqrt(running_var + eps)`, a per-CHANNEL constant read
        from a buffer (`layers.py:73-80`). No batch statistic, no
        sequence statistic. The only position MIXING is the skip's
        repeat_interleave, which is an index map: output position p reads
        128 bp bin p // 128.

        So the skip is gathered at `bins // 128` and handed over already
        aligned, one row per output position. `repeat_factor` inside
        OutputEmbedder is then n // n = 1 and its upsample branch is
        skipped, which is what lets the port's own module be used
        unmodified. Get that index map wrong and the skip contribution
        shifts by up to 127 bp while raising nothing.
        """
        torch = self._torch
        m = self.model
        oi = self._org
        if offload_skips is None:
            offload_skips = self.offload_skips
        dna = m.dtype_policy.cast_to_compute(x_t)
        trunk, intermediates = m.encoder(dna)
        L = x_t.shape[1]
        crop = None
        if self.decoder_crop and bins is not None and bins.numel() > 0:
            b_lo, b_hi = int(bins.min()), int(bins.max()) + 1
            lo_bp = max(0, b_lo - self.crop_halo_bp) // 128 * 128
            hi_bp = min(L, -(-(b_hi + self.crop_halo_bp) // 128) * 128)
            if hi_bp - lo_bp < L:
                crop = (lo_bp, hi_bp)
        if crop is not None:
            # Every skip is sliced to the crop on the 128 bp-aligned
            # bounds, so each resolution cuts on a bin boundary. `.clone()`
            # gives the decoder small contiguous inputs.
            lo_bp, hi_bp = crop
            sliced = {}
            for k, v in intermediates.items():
                b = int(k.rsplit("_", 1)[1])
                sliced[k] = v[:, :, lo_bp // b: hi_bp // b].clone()
            intermediates = sliced
            del sliced
        if offload_skips:
            # The seven U-Net skips are ~7.4 GB fp32 and stay live from
            # the encoder all the way through the decoder. Park them in
            # host RAM and fetch each one back at the moment its
            # UpResBlock asks for it.
            #
            # This is the CHEAP offload: 7.4 GB moved, against
            # save_on_cpu's ~248 GB of every saved activation. It also
            # composes with gradient checkpointing, which save_on_cpu
            # does not -- nothing here touches autograd's saved-tensor
            # hooks, it is two ordinary device transfers, and `.to()` is
            # differentiable so the gradient flows back through them to
            # the encoder unchanged.
            intermediates = _SkipsOnHost(
                {k: v.to("cpu", non_blocking=True)
                 for k, v in intermediates.items()},
                self.device)
        trunk = trunk.transpose(1, 2)
        trunk = trunk + m.organism_embed(oi).unsqueeze(1)
        trunk, _pair = m.tower(trunk,
                               compute_dtype=m.dtype_policy.compute_dtype)
        trunk_ncl = trunk.transpose(1, 2)
        emb128 = m.embedder_128bp(trunk_ncl, oi, channels_last=False)
        if crop is not None:
            decoded = m.decoder(
                trunk_ncl[:, :, lo_bp // 128: hi_bp // 128].clone(),
                intermediates)
            z = decoded.index_select(2, bins - lo_bp)   # (B, 768, n)
        else:
            decoded = m.decoder(trunk_ncl, intermediates)   # (B, 768, S)
            z = decoded.index_select(2, bins)               # (B, 768, n)
        del decoded
        self._last_crop = crop
        s = emb128.index_select(
            2, torch.div(bins, 128, rounding_mode="floor"))
        emb1 = m.embedder_1bp(z, oi, skip_x=s,
                              channels_last=False)     # (B, 1536, n)
        return self._sliced_head(emb1, None)            # (n, kept)

    def _sliced_head(self, emb_ncl, bins):
        """The head, restricted to `bins` positions and the kept tracks.

        EXACT, not an approximation. `GenomeTracksHead` at resolution 1
        is conv1x1 -> softplus(.) * softplus(residual_scale) ->
        predictions_scaling. Every step is position-wise, so gathering
        positions commutes with it, and every step is per-output-channel,
        so slicing channels commutes too. Running the head as shipped
        would cost a (1, 1_048_576, 768) activation plus an equal
        cotangent, which is what does not fit.

        emb_ncl: (1, C, S) 1 bp embeddings, NCL.  bins: (n,) long, or
        None when `_gathered_readout` has already gathered them.
        Returns (n, kept) fp32 on experimental scale -- the quantity
        `forward` returns, at `bins`.
        """
        torch = self._torch
        from alphagenome_pytorch.heads import predictions_scaling
        head = self.model.heads[self.head_name]
        conv = head.convs["1"]
        o = 0                                           # human
        z = emb_ncl if bins is None else emb_ncl.index_select(2, bins)
        w = conv.weight[o].index_select(0, self._cols)  # (n_kept, C)
        b = conv.bias[o].index_select(0, self._cols)    # (n_kept,)
        pred = (torch.einsum("bcs,oc->bos", z.float(), w.float())
                + b.float().view(1, -1, 1))
        scale = head.residual_scales["1"][o].index_select(
            0, self._cols).float()
        pred = (torch.nn.functional.softplus(pred)
                * torch.nn.functional.softplus(scale.view(1, -1, 1)))
        tm = (head.track_means[o].index_select(0, self._cols)
              .float().view(1, -1))
        pred = predictions_scaling(pred, track_means=tm, resolution=1,
                                   apply_squashing=head.apply_squashing,
                                   channels_last=False)
        return pred[0].transpose(0, 1)                  # (n, kept)

    # ----------------------------------------------------------- gradient
    def scalar_gradient(
            self, x: np.ndarray, replicate: Any,
            scalar_fn: Callable[[np.ndarray], float]) -> np.ndarray:
        """d scalar_fn(forward(x))/dx as (L, 4), RAW -- not grad*input."""
        self._check_replicate(replicate)
        torch = self._torch
        bins = list(getattr(scalar_fn, "bins", ()))
        kept = list(getattr(scalar_fn, "tracks", ()))
        if not bins:
            raise S.ConfigError("gene_scalar_fn advertised no readout bins")
        if not kept:
            raise S.ConfigError("the strand mask removed every track")
        tt = self._theta_scalar
        n_tracks = len(self.meta.tracks[self.key])

        bt = torch.as_tensor(np.asarray(bins, dtype=np.int64),
                             device=self.device)
        # The span the decoder will run over, and from it the per-gene
        # offload choice under the `auto` mode.
        crop_bp = self.meta.seq_len
        if self.decoder_crop:
            b_lo, b_hi = int(min(bins)), int(max(bins)) + 1
            crop_bp = (min(self.meta.seq_len,
                           -(-(b_hi + self.crop_halo_bp) // 128) * 128)
                       - max(0, b_lo - self.crop_halo_bp) // 128 * 128)
        if self.offload_auto:
            heavy = crop_bp > self.auto_max_crop_bp
            use_offload, use_skips = heavy, heavy
        else:
            use_offload, use_skips = self.offload, self.offload_skips
        self._crop_stats["n"] += 1
        self._crop_stats["crop_bp_sum"] += int(crop_bp)
        self._crop_stats["n_cropped"] += int(
            self.decoder_crop and crop_bp < self.meta.seq_len)
        self._crop_stats["n_offloaded"] += int(use_offload)
        # The strand mask AND track_agg as one weight vector -- `sum` and
        # `mean` are both linear.
        wv = np.zeros(n_tracks, dtype=np.float32)
        wv[kept] = (1.0 / len(kept)) if tt["track_agg"] == "mean" else 1.0
        wt = torch.as_tensor(wv, device=self.device)

        xg = self._put(x).requires_grad_(True)
        with self._offload(force=use_offload), self._autocast():
            y = self._gathered_readout(xg, bt, offload_skips=use_skips)
        per_bin = (y * wt.unsqueeze(0)).sum(dim=1)      # A.2 8
        total = (per_bin.sum() if tt["bin_agg"] == "sum"
                 else per_bin.mean())                   # A.2 12
        if tt["log_scalar"]:                            # A.2 14-15
            arg = total + tt["pseudocount"] + S.LOG_EPS
            u = torch.log2(arg) if tt["log_base"] == "2" else torch.log(arg)
        else:
            u = total
        if self.reclaim_before_backward:
            # The backward's first act is a 3.00 GiB allocation, the
            # gradient w.r.t. the decoder's (1, 768, 1_048_576) output.
            # Returning the forward's cached-but-unallocated blocks to
            # the driver here is the difference between fitting and not,
            # and costs one sync per gene against ~10 s of compute.
            torch.cuda.empty_cache()
        u.backward()

        g = xg.grad.detach()[0].float().cpu().numpy()
        y_np = y.detach().float().cpu().numpy()
        u_t = float(u.detach())
        del y, per_bin, total, u, xg

        if not np.isfinite(g).all():
            raise RuntimeError(
                f"non-finite gradient: {(~np.isfinite(g)).sum():,} of "
                f"{g.size:,} entries (precision={self.precision}).")

        # The scalar mirror, on every call. Rebuild the full-length array
        # the numpy gene scalar expects; the zeros elsewhere are never
        # read, because `scalar_fn` touches only y[bins][:, tracks].
        y_probe = np.zeros((self.meta.n_bins, n_tracks), dtype=np.float64)
        y_probe[np.asarray(bins, dtype=np.int64)] = y_np
        u_np = float(scalar_fn(y_probe))
        # The gene readout of this pass, as differentiated and un-logged;
        # the frame agreement gate reads T.
        self._last_u = u_np
        lin = getattr(scalar_fn, "linear", None)
        self._last_T = float(lin(y_probe)) if lin is not None else float("nan")
        rel = abs(u_t - u_np) / max(1.0, abs(u_np))
        self.gate_worst = max(self.gate_worst, rel)
        if not (rel <= SCALAR_GATE_RTOL):
            raise RuntimeError(
                "the torch gene scalar and GENE-SCALAR disagree: "
                f"torch={u_t!r} numpy={u_np!r} rel={rel:.3e} > "
                f"{SCALAR_GATE_RTOL:.0e}, on {len(bins):,} bins x "
                f"{len(kept)} tracks. A difference this large is a wiring "
                "error (track set, log, reduction order), not round-off.")
        if not np.isfinite(u_np):
            raise ValueError(
                "the gene scalar is not finite on the reference sequence")

        self.n_gradient += 1
        assert g.shape == (self.meta.seq_len, 4), \
            f"gradient must be ({self.meta.seq_len}, 4), got {g.shape}"
        return g

    # ---------------------------------------------------------- attention
    def attention(self, x: np.ndarray, replicate: Any,
                  how: str) -> np.ndarray:
        """Head- and layer-averaged attention on the 128 bp token grid.

        Avsec 2021's `mean-heads-layers`, returned as (8,192 x 8,192).
        No reported arm reads this path, so the shape assertion below is
        the tripwire: if the port's layout differs, this raises rather
        than returning a matrix on the wrong grid.

        AlphaGenome is a U-Net, so an attention map is a PARTIAL account
        of enhancer-to-gene information flow: the encoder skips re-join
        the decoder having never been attended over. Real and readable,
        but not the whole path.

        MEMORY: `att` is already (1, 8, 8192, 8192) fp32 = 2.1 GB inside
        the forward pass; the head-mean kept here is one 268 MB matrix
        per layer, and the average over layers accumulates in place so
        only two are ever resident.
        """
        if how != "mean-heads-layers":
            raise ValueError(
                f"attention_aggregation={how!r} is not implementable "
                "through this interface: `Backend.attention` returns one "
                "matrix, so a per-layer or per-head choice would need a "
                "layer/head index parameter that PARAMS does not define.")
        self._check_replicate(replicate)
        mods = self._attn_modules()
        for m in mods:
            m._collect_attention = True
            m._attn_head_mean = None
        try:
            with self._torch.no_grad(), self._autocast():
                # The same call `forward` makes. Wasteful by one head
                # evaluation, but it is the path that is measured; a
                # private trunk entry point would be a second forward
                # implementation to keep in step.
                out = self.model.predict(self._put(x), organism_index=0,
                                         heads=[self.head_name],
                                         resolutions=[1])
                del out
            mats = [m._attn_head_mean for m in mods]
            if any(a is None for a in mats):
                raise RuntimeError(
                    "the attention patch did not fire on every MHABlock; "
                    "the port's layout changed and this backend must be "
                    "re-read against it")
            A = mats[0].astype("float64")
            for a in mats[1:]:   # incremental mean: two resident
                A += a
            A /= len(mats)
            A = A.astype("float32")
        finally:
            for m in mods:
                m._collect_attention = False
                m._attn_head_mean = None
        n_tok = self.meta.seq_len // self.ATTENTION_BIN
        assert A.shape == (n_tok, n_tok), \
            (f"attention must be on the 128 bp token grid ({n_tok}); got "
             f"{A.shape}. A different grid would shift every enhancer "
             "offset.")
        return A

    def _attn_modules(self):
        """The port's sequence-attention blocks, patched on first use.

        Patched lazily rather than in `__init__` so the gradient and ISM
        arms -- which never read attention -- pay nothing and cannot be
        perturbed by the patch.
        """
        if getattr(self, "_attn_patched", None) is not None:
            return self._attn_patched
        from alphagenome_pytorch.attention import MHABlock
        mods = [m for m in self.model.modules() if isinstance(m, MHABlock)]
        if not mods:
            raise RuntimeError(
                "no MHABlock found in the AlphaGenome port; attention "
                "cannot be read. RowAttentionBlock is the PAIR-embedding "
                "attention and is a different object.")
        import types
        for m in mods:
            m.forward = types.MethodType(_patched_mha_forward, m)
            m._collect_attention = False
        self._attn_patched = mods
        return mods

    # --------------------------------------------------------------- misc
    @staticmethod
    def _check_replicate(replicate: Any) -> None:
        if replicate not in (1, "1", "ENSEMBLE"):
            raise ValueError(
                "one AlphaGenome checkpoint is loaded; "
                f"replicate={replicate!r}. The all_folds checkpoint holds "
                "one parameter tree, so replicates > 1 would re-score "
                "identical weights and misreport it as an ensemble.")

    def provenance(self) -> dict:
        """What actually ran, for the run manifest."""
        import torch
        return {"backend": "alphagenome-pytorch", "weights": str(self.weights),
                "weights_source": ("agt convert of "
                                   "alphagenome-jax-all_folds-v1 "
                                   "(670/670 params)"),
                "precision": self.precision, "device": self.device,
                "precision_flags": self._precision, **self._dtype_policy,
                "torch": torch.__version__,
                "absolute_tracks": list(self.absolute_tracks),
                "positional_tracks": list(self.meta.tracks[self.key]),
                "head": self.head_name,
                # The memory strategy as it RESOLVED, not as it was
                # requested: a manifest asserting something that never
                # reached the computation is a provenance error.
                "offload_mode": self.offload_mode,
                "offload_save_on_cpu": self.offload,
                "offload_skips_to_host": self.offload_skips,
                "gradient_checkpointing": self.grad_checkpointing,
                "reclaim_cache_before_backward": self.reclaim_before_backward,
                # Loud on purpose: `both` relaxes torch's checkpoint
                # determinism check, which is a safety check being
                # switched off, and anyone reading a score produced that
                # way is entitled to know.
                "checkpoint_determinism_check_relaxed": bool(
                    (self.offload or self.offload_auto)
                    and self.grad_checkpointing),
                # the decoder crop and the per-gene offload decisions
                "decoder_crop": self.decoder_crop,
                "crop_halo_bp": self.crop_halo_bp,
                "offload_auto": self.offload_auto,
                "auto_max_crop_bp": self.auto_max_crop_bp,
                "crop_stats": dict(
                    self._crop_stats,
                    crop_bp_mean=(self._crop_stats["crop_bp_sum"]
                                  / max(1, self._crop_stats["n"]))),
                # The score is d(scalar)/d(input), so the weights
                # carry no gradient buffers.
                "model_params_frozen": True,
                "sliced_head": True,
                "gather_before_embedder_1bp": True,
                "scalar_gate_rtol": SCALAR_GATE_RTOL,
                "scalar_gate_worst_rel": self.gate_worst,
                "alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
                "n_forward": self.n_forward, "n_gradient": self.n_gradient,
                "fasta": self.fasta}
