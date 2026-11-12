"""
Borzoi backend for `score_eg.py`.

Companion to `backend_enformer.py`, with the same division of labour:
torch plumbing here, specification in `score_eg.py`. Implements all
three `Backend` methods.

    forward(x, replicate)              -> (16352, 7611)  predictions
    scalar_gradient(x, replicate, fn)  -> (L, 4)  d fn(forward(x))/dx
    attention(x, replicate, how)       -> (4096, 4096)  128 bp tokens

Five things here are easy to get silently wrong. Each is asserted or
derived rather than assumed.

**1. The prediction crop is 16,352 bins, not 6,144.** Borzoi trains
with `Cropping1D(5120)`, which is the 6,144-bin output everyone quotes,
but the released CRISPRi run scripts pass `params_pred.json`, whose
trunk crops 16 bins: 16,384 - 2*16 = 16,352. So the benchmark reads
exon bins over 523,264 bp of its 524,288 bp input, not over the central
196,608. `borzoi-pytorch` defaults to `return_center_bins_only=True`
(6,144); this backend forces that off and then asserts
`model.crop.target_length == meta.n_bins`, because a silent 6,144 would
place every exon bin 5,104 bins from where the paper places it.

**2. `checkpoint_sequential` is unsafe on `res_tower`.**
`nn.Module.named_children()` deduplicates by object identity and Borzoi
reuses one `MaxPool1d` instance for all four pools of the tower, so
`len(res_tower)` is 9 while `len(list(res_tower.children()))` is 6.
`checkpoint_sequential(res_tower, 9, x)` therefore raises, and any
`segments <= 6` would instead have run, silently recomputing a tower
with three of its four pools missing. This file iterates `res_tower`
directly -- `Sequential.__iter__` walks `_modules.values()` and does
not deduplicate -- and groups each (ConvBlock, MaxPool) pair into one
segment, which also halves the stored boundaries.

**3. The transformer needs the reentrant checkpoint, and that forbids
`autograd.grad(..., t)`.** `fast_relative_shift` is a `torch.vmap` of
an `as_strided` trick, and `torch.func` transforms are incompatible
with the saved-tensor hooks that `use_reentrant=False` installs. The
reentrant implementation has no such hooks, but it refuses `.grad()`
and `backward(inputs=...)`. So the trunk is differentiated with
`torch.autograd.backward(y_sub, dU/dy)` and the answer read off
`t.grad`, never `torch.autograd.grad(U, t)`.

**4. The gene scalar is mirrored in torch, and the mirror is gated on
every call.** `backend_enformer.py` forms dU/dy in float64 from the
gene scalar itself (in closed form where the scalar is linear-then-log),
which costs little there because Enformer's readout is a few bins of a
few tracks. Borzoi's is every
exon bin x 10 tracks, often 3,000 to 15,000 entries, so the same route
would need tens of thousands of numpy evaluations per gene. This file
re-expresses the gene scalar in torch (`_gene_scalar_torch`) and lets
autograd supply dU/dy in one pass. That is a second implementation of
the algorithm's most order-sensitive step, so it is checked rather than
trusted:

    GATE-VALUE  compares the torch scalar's value against `scalar_fn`,
                the specification, on the same fp64 numbers
    GATE-DERIV  central-differences a random handful of dU/dy entries
                through `scalar_fn` in fp64

Both run on every call and both raise. Any parameter combination
`_gene_scalar_torch` does not mirror raises `NotImplementedError`
naming it, rather than quietly computing something else.

**5. `forward` fills un-requested tracks with NaN, on purpose.** The
`Backend` contract is (n_bins, n_tracks) = 498 MB in fp32, and the head
that produces it is a 1x1 convolution, so materialising all 7,611
tracks costs real time on every ISM forward pass while the gene scalar
reads 10 of them. This backend is constructed with the exact track set
it will be asked for (`narrow_tracks`, which `make_backend` fills from
`resolve(theta).tracks`) and writes only those columns. Everything else
stays NaN, not zero: zero is a legal prediction and would propagate as
a plausible number, NaN propagates as NaN and fails the finite check.
"""

from __future__ import annotations

import os

# Set before the CUDA caching allocator initialises. The conv tower
# cycles through 0.5-0.7 GB short-lived buffers at 524,288 bp, so an
# out-of-memory failure here is fragmentation rather than capacity;
# expandable segments let the allocator grow one region instead of
# stranding freed blocks.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import types
from typing import Any, Callable, Dict, Optional, Sequence

import numpy as np

#: Bound by `BorzoiBackend.__init__`, the only place torch is imported.
#: The module must stay importable without a GPU so that
#: `--print-config` and the config validator work.
torch = None

#: The four replicates the released benchmark scores: data fold 3,
#: replicates 0-3 (`-f 3 -c 0,1,2,3`). These ports are 1:1 with those,
#: in order, so `replicate` in 1..4 indexes this tuple.
CHECKPOINTS = ("johahi/borzoi-replicate-0", "johahi/borzoi-replicate-1",
               "johahi/borzoi-replicate-2", "johahi/borzoi-replicate-3")


def _pin_fp32(torch) -> dict:
    """Turn TF32 off for convolutions and matmuls, and report the state.

    This is a reproducibility control, not a performance knob, and it is
    silent when wrong. `precision_policy: port` and
    `map_precision: float32` declare that this arm computes in fp32.
    TF32 is not fp32: it keeps fp32's exponent but truncates the
    mantissa to 10 bits, so a convolution tower run under it agrees with
    fp32 to about 1e-3 relative, not 1e-7.

    The flag's value is the same everywhere while its effect is not.
    torch ships `cudnn.allow_tf32 = True` by default, and TF32 exists
    only from compute capability 8.0, so the identical source computes
    something different on either side of that line with nothing in the
    manifest to say so. Borzoi's trunk is convolutional, so `cudnn` is
    the operative flag rather than `matmul`.

    Pinned rather than merely recorded, because the ISM arm has to stay
    numerically comparable to the gradient arm it is plotted beside. The
    returned dict goes into `provenance()`, so a run on different
    hardware can prove which it was.
    """
    found = {"cudnn_allow_tf32_default": bool(torch.backends.cudnn.allow_tf32),
             "matmul_allow_tf32_default":
                 bool(torch.backends.cuda.matmul.allow_tf32)}
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    found["tf32_pinned_off"] = True
    if torch.cuda.is_available():
        found["gpu_name"] = torch.cuda.get_device_name(0)
        found["gpu_capability"] = ".".join(
            str(v) for v in torch.cuda.get_device_capability(0))
        # 8.0 is where the default would otherwise have bitten.
        found["tf32_capable"] = torch.cuda.get_device_capability(0)[0] >= 8
    return found


# ---- the torch mirror of the gene scalar -----------------------------

def _unsquash_torch(z, scale: float, exponent: float,
                    clip_soft: Optional[float], generation: str):
    """Mirror of `score_eg.unsquash`, a five-line elementwise function.

    Kept next to the reduction chain it feeds rather than imported,
    because `score_eg.unsquash` takes a `Resolved` and operates on
    numpy. The two are compared on every call by GATE-VALUE.
    """
    if generation == "old":              # y/scale, unclip, ^(1/exponent)
        z = z / scale
        if clip_soft is not None:
            z = torch.where(z > clip_soft, (z - clip_soft) ** 2 + clip_soft, z)
        return torch.clamp(z, min=0.0) ** (1.0 / exponent)
    if generation == "new":
        if clip_soft is not None:
            z = torch.where(z > clip_soft,
                            (z - clip_soft + 1) ** 2 + clip_soft - 1, z)
        z = -1.0 + torch.clamp(z + 1.0, min=0.0) ** (1.0 / exponent)
        return z / scale
    raise NotImplementedError(f"transform_generation={generation!r}")


def _reduce_torch(a, dim: int, op: str):
    """Mirror of `score_eg._reduce_axis`."""
    if op == "sum":
        return a.sum(dim=dim)
    if op == "mean":
        return a.mean(dim=dim)
    if op == "max":
        return a.max(dim=dim).values
    raise NotImplementedError(
        f"reduction {op!r} is not mirrored in backend_borzoi")


def _patched_attention(self, x):
    """Upstream `Attention.forward`, plus one line that keeps `attn`.

    Verbatim otherwise: the module is pure einsum plus softmax with no
    fused SDPA kernel, so nothing is hidden and nothing needs
    re-deriving. Mirrors `backend_enformer._patched_attention`, and the
    two differ only where the packages do -- Borzoi caches
    `self.positions` and uses `fast_relative_shift`.
    """
    from einops import rearrange
    from torch import einsum
    from borzoi_pytorch.pytorch_borzoi_transformer import (
        get_positional_embed, fast_relative_shift)

    n, h, device = x.shape[-2], self.heads, x.device
    if self.positions is None:
        self.positions = get_positional_embed(
            n, 4096, self.num_rel_pos_features, self.to_v.weight.device)
    q, k, v = self.to_q(x), self.to_k(x), self.to_v(x)
    q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h),
                  (q, k, v))
    q = q * self.scale
    content_logits = einsum("b h i d, b h j d -> b h i j",
                            q + self.rel_content_bias, k)
    positions = self.pos_dropout(self.positions)
    rel_k = self.to_rel_k(positions)
    rel_k = rearrange(rel_k, "n (h d) -> h n d", h=h)
    rel_logits = fast_relative_shift(q + self.rel_pos_bias, rel_k)
    logits = content_logits + rel_logits
    attn = logits.softmax(dim=-1)

    # The one added line. Mean over heads here, so each layer holds
    # (n, n) -- 67 MB fp32 at n = 4,096 -- instead of the head-resolved
    # (8, n, n) = 537 MB, and 4.3 GB across the 8 layers.
    if getattr(self, "_collect_attention", False):
        self._attn_head_mean = (
            attn.detach().mean(dim=1)[0].float().cpu().numpy())

    attn = self.attn_dropout(attn)
    out = einsum("b h i j, b h j d -> b h i d", attn, v)
    out = rearrange(out, "b h n d -> b n (h d)")
    return self.to_out(out)


class BorzoiBackend:
    """Borzoi via `borzoi-pytorch`; `replicate` selects one of four.

    One model lives on the device; the other three replicates are held
    as state dicts and swapped in with `load_state_dict`.
    `run_experiment`'s loop order is gene -> window -> replicate ->
    augmentation, so a replicate changes once per gene per replicate:
    four 0.69 GB host-to-device copies per gene (about 0.1 s each)
    against four 710 MB safetensors reads (about 1.5 s each) if the
    model were rebuilt, or 2.8 GB of resident device memory if all four
    were kept there, which does not fit beside a 6.4 GB backward pass on
    an 8 GB device.
    """

    def __init__(self, meta, theta: Dict[str, Any], rho, device: str = "cuda",
                 checkpoints: Sequence[str] = CHECKPOINTS,
                 checkpoint_trunk: bool = True,
                 narrow_tracks: Optional[Sequence[int]] = None,
                 gate_deriv_samples: int = 4):
        global torch
        import torch as _torch
        torch = _torch
        from borzoi_pytorch import Borzoi
        import score_eg      # for LOG_EPS; never for the scalar itself

        self._precision = _pin_fp32(torch)
        self._log_eps = score_eg.LOG_EPS
        self.meta = meta
        self.theta = theta
        self.rho = rho
        self.device = device
        self.checkpoints = tuple(checkpoints)
        self.checkpoint_trunk = checkpoint_trunk
        self.gate_deriv_samples = int(gate_deriv_samples)
        self._torch = torch
        self._gate_rng = np.random.RandomState(0)
        #: Entries the derivative gate declined to audit because no
        #: valid finite-difference step exists for them. Counted, never
        #: silent; see `_run_gates`.
        self._gate_deriv_skipped = 0
        self._gate_worst_value = 0.0
        self._gate_worst_deriv = 0.0
        self._gate_checks = 0

        if len(self.checkpoints) < theta["replicates"]:
            raise ValueError(f"replicates={theta['replicates']} but only "
                             f"{len(self.checkpoints)} checkpoints are "
                             f"configured")

        # `return_center_bins_only=False` selects
        # `TargetLengthCrop(16384 - 32)`; see trap 1.
        model = Borzoi.from_pretrained(self.checkpoints[0],
                                       return_center_bins_only=False)
        got = int(model.crop.target_length)
        if got != meta.n_bins:
            raise RuntimeError(
                f"Borzoi's crop returns {got} bins but ModelMeta declares "
                f"{meta.n_bins}. params_pred.json crops 16 bins (=> 16,352); "
                "params.json crops 5,120 (=> 6,144). The CRISPRi runs used "
                "params_pred.json, so 16,352 is the reproduction's value.")
        # .eval() is load-bearing and silent if forgotten: dropout_rate
        # 0.2, attn_dropout 0.05, pos_dropout 0.01, and BatchNorm1d
        # throughout the tower.
        model = model.eval()
        # Freezing the weights is a memory decision, not only a
        # correctness one: with requires_grad=False autograd never needs
        # to save a convolution's input (dL/dinput needs the weights,
        # dL/dweight needs the input), which is most of the tower's
        # activations.
        for p in model.parameters():
            p.requires_grad_(False)
        self.model = model.to(device)

        # Attention capture. Borzoi's transformer runs on 4,096 tokens
        # of 128 bp -- the same grid Enformer attends on -- and its
        # `Attention.forward` materialises the softmax explicitly,
        # so the matrix is readable by patching one line.
        # `FlashAttention` (selected by `config.flashed`) never
        # materialises it, so that path is refused rather than silently
        # returning something else.
        from borzoi_pytorch.pytorch_borzoi_transformer import (
            Attention, FlashAttention)
        if any(isinstance(m, FlashAttention) for m in self.model.modules()):
            self._attn_modules = []
            self._attn_note = (
                "this Borzoi was built with flashed=True; FlashAttention "
                "never materialises the attention matrix, so it cannot be "
                "read. Rebuild with flashed=False to score method=attention.")
        else:
            self._attn_modules = [m for m in self.model.modules()
                                  if isinstance(m, Attention)]
            self._attn_note = ""
            if not self._attn_modules:
                raise RuntimeError(
                    "no Attention modules found; the package layout changed")
            for m in self._attn_modules:
                m.forward = types.MethodType(_patched_attention, m)
                m._collect_attention = False

        if checkpoint_trunk:
            self._install_checkpointing()

        # The other replicates, as state dicts. Loaded eagerly so a
        # missing checkpoint fails now rather than at gene 40 of a long
        # run.
        #
        # They live on the host by default, and `_select` then
        # copies 0.69 GB host-to-device per swap, which dominates
        # `forward` time on an ISM arm. `EG_GPU_REPLICATE_CACHE=1` pins
        # the cache to the device instead, so `load_state_dict` copies
        # device-to-device: same tensors, same values, same `copy_`,
        # about 2 GB more device memory. It is a residency choice, not
        # a numerical one, and it does not suit an 8 GB device.
        _gpu_cache = os.environ.get("EG_GPU_REPLICATE_CACHE", "") == "1"
        _cache_dev = device if _gpu_cache else "cpu"
        self._replicate_cache_device = str(_cache_dev)
        self._state: Dict[int, Any] = {
            1: {k: v.detach().clone().to(_cache_dev)
                for k, v in self.model.state_dict().items()}}
        for i in range(2, theta["replicates"] + 1):
            m = Borzoi.from_pretrained(self.checkpoints[i - 1],
                                       return_center_bins_only=False)
            self._state[i] = {k: v.detach().to(_cache_dev)
                              for k, v in m.state_dict().items()}
            del m
        self._loaded = 1

        # Read the track count off the model, never off a paper sentence
        # or a metadata field.
        self.n_tracks_total = int(self.model.human_head.out_channels)
        # Which output columns `forward` will actually compute.
        # Everything else stays NaN.
        self._narrow = (tuple(sorted(set(int(t) for t in narrow_tracks)))
                        if narrow_tracks is not None
                        else tuple(range(self.n_tracks_total)))
        if self._narrow and self._narrow[-1] >= self.n_tracks_total:
            raise ValueError(f"track index {self._narrow[-1]} is outside "
                             f"Borzoi's {self.n_tracks_total}-track human "
                             f"head")
        self._narrow_t = torch.as_tensor(self._narrow, dtype=torch.long,
                                         device=device)
        # One reusable (n_tracks, n_bins) fp64 buffer, returned
        # transposed, so that the gene scalar's `y[:, tracks]` gathers
        # whole contiguous rows instead of striding a 996 MB array by
        # 60 KB per element.
        self._buf = np.full((self.n_tracks_total, meta.n_bins), np.nan,
                            dtype=np.float64)

    # ---- setup -------------------------------------------------------
    def _install_checkpointing(self):
        """Recompute the trunk instead of storing its activations.

        Grouping matters: checkpointing every module in `res_tower`
        separately stores the post-convolution tensors too, and
        608 x 262,144 fp32 is 638 MB on its own. Grouping each
        convolution with the pool that follows it drops the tower's
        stored boundaries from about 2.7 GB to 1.3 GB. Measured peak for
        a full forward and backward at 524,288 bp is 6.4 GB.

        `use_reentrant=True` for the transformer only, because of
        trap 3.
        """
        from torch.utils.checkpoint import checkpoint

        def ck(fn, *a):
            return checkpoint(fn, *a, use_reentrant=False)

        def ckr(fn, *a):
            return checkpoint(fn, *a, use_reentrant=True)

        def segments(seq):
            """(ConvBlock, MaxPool) pairs, by `iter` not `children`."""
            mods = list(iter(seq))                 # NOT .children(): trap 2
            out, i = [], 0
            while i < len(mods):
                if (i + 1 < len(mods)
                        and type(mods[i + 1]).__name__ == "MaxPool1d"):
                    out.append((mods[i], mods[i + 1]))
                    i += 2
                else:
                    out.append((mods[i],))
                    i += 1
            return out

        res_segs = segments(self.model.res_tower)
        unet_segs = segments(self.model.unet1)

        def embs(self, x):
            def run(group):
                def f(t):
                    for m in group:
                        t = m(t)
                    return t
                return f

            x = ck(self.conv_dna, x)
            for g in res_segs:
                x = ck(run(g), x)
            x_unet0 = x
            for g in unet_segs:
                x = ck(run(g), x)
            x_unet1 = x
            y = self._max_pool(x_unet1)
            x_unet1 = ck(self.horizontal_conv1, x_unet1)
            x_unet0 = ck(self.horizontal_conv0, x_unet0)
            y = y.permute(0, 2, 1)
            for blk in self.transformer:
                y = ckr(blk, y)
            y = y.permute(0, 2, 1)
            y = ck(self.upsampling_unet1, y)
            # Not `+=`: an in-place add on a checkpointed output
            # confuses the recompute.
            y = y + x_unet1
            y = ck(self.separable1, y)
            y = ck(self.upsampling_unet0, y)
            y = y + x_unet0
            y = ck(self.separable0, y)
            y = self.crop(y.permute(0, 2, 1))
            return y.permute(0, 2, 1)

        self.model.checkpointed_embs = types.MethodType(embs, self.model)

    def _select(self, replicate: Any) -> None:
        r = self._check_replicate(replicate)
        if r == self._loaded:
            return
        self.model.load_state_dict(self._state[r])
        self._loaded = r

    def _check_replicate(self, replicate: Any) -> int:
        if replicate == "ENSEMBLE":
            raise ValueError(
                "replicate_aggregation=ensemble asks this backend for one "
                "averaged model. Linder 2025 averages the four replicates' "
                "FINAL SCORES (borzoi_bench_flowfish_folds.py:236 "
                "`np.array(site_scores).mean(axis=0)`), which is "
                "replicate_aggregation=score-mean.")
        try:
            r = int(replicate)
        except (TypeError, ValueError):
            raise ValueError(f"replicate must be an integer in "
                             f"1..{len(self.checkpoints)}, got {replicate!r}")
        if not (1 <= r <= len(self.checkpoints)):
            raise ValueError(
                f"replicate={r} is outside 1..{len(self.checkpoints)}")
        return r

    # ---- helpers -----------------------------------------------------
    def _tensor(self, x: np.ndarray):
        """(L, 4) one-hot -> (1, 4, L). Borzoi is channels-first."""
        if x.shape != (self.meta.seq_len, 4):
            raise ValueError(f"expected a ({self.meta.seq_len}, 4) one-hot, "
                             f"got {x.shape}")
        t = self._torch.from_numpy(np.ascontiguousarray(x.T, dtype=np.float32))
        return t[None, ...].to(self.device)

    def _embs(self, t):
        m = self.model
        return (m.checkpointed_embs(t)
                if self.checkpoint_trunk and t.requires_grad
                else m.get_embs_after_crop(t))

    def _head(self, e, tracks_t):
        """The 1x1 head, applied to whatever bins `e` holds.

        `final_joined_convs` is BatchNorm1d -> GELU -> Conv1d(k=1) and
        `human_head` is Conv1d(k=1), so every operation is pointwise in
        the bin axis and slicing bins before the head is exact rather
        than an approximation. That is what makes the gradient arm
        affordable: the head runs on a gene's few thousand exon bins,
        not on all 16,352.
        """
        torch = self._torch
        z = self.model.final_joined_convs(e)
        W = self.model.human_head.weight.index_select(0, tracks_t)
        b = self.model.human_head.bias.index_select(0, tracks_t)
        # Borzoi's own forward disables autocast for the head; keep
        # that, in fp32.
        with torch.amp.autocast("cuda", enabled=False):
            y = self.model.final_softplus(
                torch.nn.functional.conv1d(z.float(), W, b))
        return y

    #: The gene readout of the last gradient pass (see score_eg.Backend):
    #: u as differentiated, T un-logged. The frame agreement gate reads T.
    _last_T = float("nan")
    _last_u = float("nan")

    # ---- Backend interface -------------------------------------------
    def forward(self, x: np.ndarray, replicate: Any) -> np.ndarray:
        """Predictions as (n_bins, n_tracks) = (16352, 7611).

        Only the columns this backend was constructed for
        (`narrow_tracks`) are computed. Every other column is NaN, and
        reading one propagates NaN into the gene scalar rather than a
        plausible zero. `make_backend` sets `narrow_tracks` from
        `resolve(theta).tracks`, plus the MAJOR-TSS track set when
        `tss_major` is on, which is the complete set the gene scalar and
        MAJOR-TSS can read; so a NaN reaching a score means the
        resolution changed underneath the backend, which is a bug rather
        than a number.

        The returned array is a view of a buffer this backend reuses on
        the next call. Every caller in `score_eg.py` indexes it and
        drops it immediately, so this is safe and saves a 996 MB
        allocation per ISM draw. Copy it to keep it.
        """
        self._select(replicate)
        torch = self._torch
        with torch.no_grad():
            e = self._embs(self._tensor(x))
            y = self._head(e, self._narrow_t)    # (1, |narrow|, n_bins)
        y = y[0].detach().float().cpu().numpy()
        if not np.isfinite(y).all():
            raise RuntimeError("Borzoi produced a non-finite prediction on "
                               "the requested tracks")
        assert y.shape == (len(self._narrow), self.meta.n_bins), \
            f"head returned {y.shape}, expected " \
            f"{(len(self._narrow), self.meta.n_bins)}"
        self._buf[list(self._narrow), :] = y
        return self._buf.T

    def scalar_gradient(self, x: np.ndarray, replicate: Any,
                        scalar_fn: Callable[[np.ndarray], float]
                        ) -> np.ndarray:
        """d scalar_fn(forward(x)) / dx, as (L, 4).

        The vector-Jacobian route of `backend_enformer.py` does not
        scale here (trap 4), so dU/dy comes from a torch mirror of the
        gene scalar, gated against the numpy specification on every
        call.
        """
        self._select(replicate)
        torch = self._torch
        bins = tuple(int(b) for b in getattr(scalar_fn, "bins"))
        tracks = tuple(int(t) for t in getattr(scalar_fn, "tracks"))
        if not bins or not tracks:
            raise ValueError(
                "scalar_fn advertises no readout bins or no tracks")
        missing = sorted(set(tracks) - set(self._narrow))
        if missing:
            raise ValueError(
                f"scalar_fn reads tracks {missing[:5]} that this backend was "
                "not built for; narrow_tracks must cover every track "
                "resolve() produces")

        bins_t = torch.as_tensor(bins, dtype=torch.long, device=self.device)
        tracks_t = torch.as_tensor(tracks, dtype=torch.long,
                                   device=self.device)

        t = self._tensor(x).requires_grad_(True)
        e = self._embs(t)
        e = e.index_select(2, bins_t)            # (1, dim, |bins|)
        y = self._head(e, tracks_t)              # (1, |tracks|, |bins|)
        y_sub = y[0].transpose(0, 1).double()    # (|bins|, |tracks|)

        U = self._gene_scalar_torch(y_sub, getattr(scalar_fn, "xform", None))
        if not torch.isfinite(U):
            raise ValueError(
                "the gene scalar is not finite on the reference sequence")

        # dU/dy_sub. `autograd.grad` is legal here because the path from
        # U to y_sub does not cross a reentrant checkpoint; the trunk is
        # reached below via `backward`, which does.
        dUdy, = torch.autograd.grad(U, y_sub, retain_graph=True)

        self._run_gates(scalar_fn, bins, tracks, y_sub, U, dUdy)

        # Trap 3: never grad(..., inputs=t).
        torch.autograd.backward(y_sub, dUdy)
        g = t.grad
        if g is None:
            raise RuntimeError("no gradient reached the input tensor")
        g = g[0].transpose(0, 1).detach().float().cpu().numpy()   # (L, 4)
        if not np.isfinite(g).all():
            raise RuntimeError("Borzoi produced a non-finite input gradient")
        return g

    # ---- the mirrored scalar -----------------------------------------
    def _gene_scalar_torch(self, y_sub, xform=None):
        """GENE-SCALAR (A.2 4-15) on the sliced block of predictions.

        `gene_scalar_fn` has already applied the strand mask (it chose
        `tracks`) and READOUT-BINS (it chose `bins`), so what is left
        is the four steps that carry the numbers: the inverse
        transform, the cross-track reduction, the cross-bin reduction,
        and the logarithm.
        """
        theta, rho = self.theta, self.rho
        # `xform` overrides this backend's own constants, for the case
        # where the scalar being differentiated belongs to a different
        # track set than `self.rho`. It defaults to self.rho, so every
        # ordinary arm is unaffected.
        scale, exponent, clip_soft = xform if xform is not None else (
            rho.scale, rho.exponent, rho.clip_soft)
        z = y_sub
        if theta["inverse_transform"]:                                # A.2 5-7
            z = _unsquash_torch(z, scale, exponent, clip_soft,
                                rho.meta.transform_generation)
        per_bin = _reduce_torch(z, dim=1, op=theta["track_agg"])      # A.2 8
        # `bin_agg` is frozen at "sum" (score_eg.FROZEN_BIN_AGG), so the
        # reduction is spelled out here rather than imported.
        total = _reduce_torch(per_bin, dim=0, op="sum")               # A.2 12
        if not theta["log_scalar"]:                                   # A.2 15
            return total
        # `pseudocount` is frozen at 0.0 (score_eg.FROZEN_PSEUDOCOUNT);
        # adding it is a no-op, so the term is dropped rather than
        # imported across the boundary.
        arg = total + self._log_eps                                   # A.2 14
        if rho.meta.log_base == "2":
            return torch.log2(arg)
        if rho.meta.log_base == "e":
            return torch.log(arg)
        raise NotImplementedError(f"log_base={rho.meta.log_base!r}")

    def _run_gates(self, scalar_fn, bins, tracks, y_sub, U, dUdy) -> None:
        """GATE-VALUE and GATE-DERIV, the checks on the torch mirror.

        Both compare against `scalar_fn`, which is the specification, on
        the same fp64 numbers. A `Y` array of the model's full output
        shape is needed because `scalar_fn` indexes tracks absolutely;
        the entries it does not read stay NaN, which is harmless --
        they reduce across tracks into bins the scalar never touches --
        and is why this costs one buffer rather than one per
        evaluation.
        """
        yn = y_sub.detach().cpu().numpy()
        Y = self._buf                     # (n_tracks, n_bins), NaN-filled
        rows, cols = list(tracks), list(bins)
        Y[np.ix_(rows, cols)] = yn.T
        try:
            u_spec = float(scalar_fn(Y.T))
            self._last_u = u_spec
            lin = getattr(scalar_fn, "linear", None)
            self._last_T = float(lin(Y.T)) if lin is not None else float("nan")
            u_mine = float(U.detach().cpu())
            rel = abs(u_mine - u_spec) / max(1.0, abs(u_spec))
            self._gate_worst_value = max(self._gate_worst_value, rel)
            if not (rel <= 1e-9):
                raise AssertionError(
                    "GATE-VALUE failed: the torch mirror of GENE-SCALAR "
                    "disagrees with score_eg.gene_scalar_fn.  "
                    f"mine={u_mine!r} spec={u_spec!r} rel={rel:.3e}.\n"
                    "  backend_borzoi._gene_scalar_torch has drifted from "
                    "score_eg §A.2, or a parameter it does not mirror was "
                    "set.")

            k = min(self.gate_deriv_samples, yn.size)
            if k > 0:
                dmine = dUdy.detach().cpu().numpy()
                # Which entries to test, and why not just any of
                # them. The readout is `log(sum_b mean_t (y/0.3)^(4/3))`
                # over hundreds of exon bins, most of which the model
                # predicts as y ~ 1e-4, essentially no coverage. There,
                # the finite difference is hopeless in fp64 for two
                # reasons that pull opposite ways: the third derivative
                # of y^(4/3) diverges like y^(-5/3), so the step must be
                # small relative to y; while the entry contributes
                # ~1e-10 of U, so U(y+h) - U(y-h) sits on the round-off
                # floor and the step must be large. At y = 1.1e-5 the
                # best relative error achievable over h in [1e-4, 1e-7]
                # is 7e-4, and 8e-2 at h = 1e-4 -- all of it
                # conditioning, none of it the derivative.
                #
                # So the gate samples the entries that actually
                # determine U, ranked by their share `|y * dU/dy|`.
                # That is also where a wrong derivative would change a
                # score, which makes it a stronger test than a uniform
                # sample rather than a weaker one.
                share = np.abs(yn * dmine)
                smax = float(share.max()) if share.size else 0.0
                elig = (np.flatnonzero(share.ravel() >= 1e-3 * smax)
                        if smax > 0 else np.array([]))
                if elig.size:
                    flat = self._gate_rng.choice(
                        elig, size=min(k, elig.size), replace=False)
                    dfloor = 1e-6 * float(np.abs(dmine).max())
                    for f in flat:
                        i, j = int(f) // yn.shape[1], int(f) % yn.shape[1]
                        b, tr = cols[i], rows[j]
                        v = yn[i, j]
                        # A relative step, floored at 1e-6 absolute
                        # so that a near-zero y cannot make h vanish.
                        h = 1e-6 * max(abs(float(v)), 1.0)
                        # The floor makes h = 1e-6 regardless of y, so
                        # for |y| < 10h the step is more than 10% of y
                        # and for |y| < h it straddles zero. Such an
                        # entry is not auditable by central differencing
                        # at any step: small enough to be valid and the
                        # difference underflows, large enough to clear
                        # round-off and it is no longer a derivative.
                        # Declining to test it is honest; failing on it
                        # would assert a bug that is not there. The
                        # count is reported rather than silent.
                        if abs(float(v)) < 10.0 * h:
                            self._gate_deriv_skipped += 1
                            continue
                        Y[tr, b] = v + h
                        up = float(scalar_fn(Y.T))
                        Y[tr, b] = v - h
                        dn = float(scalar_fn(Y.T))
                        Y[tr, b] = v
                        num = (up - dn) / (2.0 * h)
                        ana = float(dmine[i, j])
                        scale = max(abs(num), abs(ana), 1e-12)
                        rel = abs(num - ana) / scale
                        self._gate_worst_deriv = max(
                            self._gate_worst_deriv, rel)
                        if not (abs(num - ana) <= 1e-3 * abs(ana) + dfloor):
                            raise AssertionError(
                                "GATE-DERIV failed: autograd's dU/dy "
                                "disagrees with a central difference of "
                                "score_eg.gene_scalar_fn at "
                                f"(bin={b}, track={tr}).\n"
                                f"  autograd={ana:.6e} numeric={num:.6e} "
                                f"rel={rel:.3e} y={float(v):.6e} "
                                f"share={float(share[i, j]) / smax:.3e} "
                                "of max\n"
                                "  Before loosening this: sweep h and check "
                                "whether the error CONVERGES. A converging "
                                "error is conditioning; a constant one is a "
                                "bug in backend_borzoi._gene_scalar_torch.")
            self._gate_checks += 1
        finally:
            # Leave the buffer as it was found.
            Y[np.ix_(rows, cols)] = np.nan

    #: Borzoi's transformer runs at the U-Net bottleneck: `res_tower` ->
    #: `unet1` -> one more max-pool -> transformer, then two
    #: `Upsample(scale_factor=2)` stages back to the 32 bp output grid.
    #: So the token width is `bin_size * 4 = 128 bp`, the same grid
    #: Enformer attends on, and an output bin maps to `bin // 4`.
    ATTENTION_UPSAMPLE_FACTOR = 4

    def attention(self, x: np.ndarray, replicate: Any, how: str) -> np.ndarray:
        """Attention averaged over all heads and all layers.

        Returns the 4,096 x 4,096 token grid at 128 bp, PRE-CROP:
        attention runs before the crop, so the query index is the
        uncropped token index and the crop must not be subtracted.
        `Backend.attention` carries the interpretation caveat: the
        U-Net skips re-join the decoder having never been attended
        over, so an attention map is a partial account of the path.
        """
        if not self._attn_modules:
            raise ValueError(
                f"attention is unavailable on this backend: {self._attn_note}")
        if how != "mean-heads-layers":
            raise ValueError(
                f"attention_aggregation={how!r} is not implementable through "
                "this interface: `Backend.attention` returns one matrix, so a "
                "per-layer or per-head choice would need a layer/head index "
                "parameter that PARAMS does not define.")
        self._select(replicate)
        for m in self._attn_modules:
            m._collect_attention = True
            m._attn_head_mean = None
        try:
            # `_embs` runs the trunk, which is what fires the
            # patched Attention modules. The embeddings themselves are
            # discarded; only the captured matrices are wanted.
            with self._torch.no_grad():
                self._embs(self._tensor(x))
            mats = [m._attn_head_mean for m in self._attn_modules]
            if any(a is None for a in mats):
                raise RuntimeError(
                    "the attention patch did not fire on every layer")
            A = np.mean(np.stack(mats, axis=0), axis=0)   # over layers
        finally:
            for m in self._attn_modules:
                m._collect_attention = False
                m._attn_head_mean = None
        n_tok = ((self.meta.seq_len // self.meta.bin_size)
                 // self.ATTENTION_UPSAMPLE_FACTOR)
        assert A.shape == (n_tok, n_tok), \
            (f"attention must be on the uncropped 128 bp token grid "
             f"({n_tok}); got {A.shape}. A different grid would shift every "
             f"enhancer offset.")
        return A

    # ---- provenance --------------------------------------------------
    def provenance(self) -> dict:
        """What ran, for the manifest."""
        return {"backend": "borzoi-pytorch",
                "checkpoints":
                    list(self.checkpoints[: self.theta["replicates"]]),
                "device": self.device,
                "checkpoint_trunk": self.checkpoint_trunk,
                "alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
                # What the device would have done by default, and what
                # was pinned instead. See `_pin_fp32`.
                "precision": self._precision,
                "n_bins": self.meta.n_bins,
                "crop_bins": self.meta.crop_bins,
                "crop_source":
                    "params_pred.json Cropping1D=16 (NOT params.json 5120)",
                "narrow_tracks": list(self._narrow),
                "gate_calls": self._gate_checks,
                "gate_worst_value_rel": self._gate_worst_value,
                "gate_worst_deriv_rel": self._gate_worst_deriv,
                # Entries the derivative gate declined to audit
                # because no valid finite-difference step exists for
                # them (|y| < 10h). Recorded so that reduced audit
                # coverage is provenance, not a silent weakening.
                "gate_deriv_skipped": self._gate_deriv_skipped,
                "gate_deriv_samples": self.gate_deriv_samples}
