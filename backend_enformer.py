"""
The Enformer backend for `score_eg.py`.

Kept out of `score_eg.py` so that file stays a readable statement of the
algorithm: everything here is torch plumbing. Implements the three
methods `Backend` declares.

    forward(x, replicate)              -> (896, 5313)   predictions
    attention(x, replicate, how)       -> (1536, 1536)  head+layer mean
    scalar_gradient(x, replicate, fn)  -> (L, 4)        d fn(y) / dx

Three things here are easy to get silently wrong, so each is asserted
rather than assumed.

**1. Attention lives on a different grid from predictions.** Attention
runs *before* the crop, so its matrix is 1,536 x 1,536 (the token grid),
while predictions are 896 bins (the cropped output). `attention_map`
expands a row with `np.repeat(A[q], 128)` to the full 196,608 bp input,
which only works at 1,536. Returning the 896-bin grid would shift every
enhancer offset by 40,960 bp and raise nothing.

**2. `enformer-pytorch` does not expose attention weights.** In
`Attention.forward`, `attn = logits.softmax(dim=-1)` is a local
variable, consumed two lines later and never stored or returned; there
is no `output_attentions` flag. A `register_forward_hook` sees only the
module's input and `to_out(out)`, so it cannot recover `attn` without
redoing the whole q/k/rel-pos computation. The only honest route is to
re-run the body with one added statement: `_patched_attention` below
replicates the upstream body verbatim and stashes the head-mean. It is
patched per instance via `types.MethodType`, never on the installed
package.

**3. `use_tf_gamma` changes the attention matrices.** The package's
*module-level* `from_pretrained` sets it True for
`EleutherAI/enformer-official-rough`, substituting precomputed
TensorFlow gamma values into the relative positional features; the
*class* method `Enformer.from_pretrained` silently leaves it False.
Those positional features feed `Attention.forward` directly, and True is
the closer match to the released weights. This backend uses the
module-level helper.
"""

from __future__ import annotations

import math
import os
import types
from typing import Any, Callable

import numpy as np

# The CUDA caching allocator reads this when it initialises, at the
# first `.to(device)`; setting it afterwards has no effect. Enformer's
# backward pass cycles through very large short-lived buffers in the
# conv tower, so a gradient can fail on a modest allocation while most
# of the card is free -- the shortfall is fragmentation, not capacity.
# Expandable segments let the allocator grow one region instead of
# stranding freed blocks.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

CHECKPOINT = "EleutherAI/enformer-official-rough"


def _patched_attention(self, x):
    """Upstream `Attention.forward` plus the one statement that keeps
    the head-averaged attention.

    Verbatim otherwise: `enformer_pytorch.modeling_enformer.Attention`
    is pure einsum + softmax with no fused SDPA kernel, so nothing is
    hidden and nothing needs re-deriving.
    """
    from einops import rearrange
    from torch import einsum
    from enformer_pytorch.modeling_enformer import (get_positional_embed,
                                                    relative_shift)

    n, h, device = x.shape[-2], self.heads, x.device
    q, k, v = self.to_q(x), self.to_k(x), self.to_v(x)
    q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h),
                  (q, k, v))
    q = q * self.scale
    content_logits = einsum("b h i d, b h j d -> b h i j",
                            q + self.rel_content_bias, k)
    positions = get_positional_embed(n, self.num_rel_pos_features, device,
                                     use_tf_gamma=self.use_tf_gamma,
                                     dtype=self.to_rel_k.weight.dtype)
    positions = self.pos_dropout(positions)
    rel_k = self.to_rel_k(positions)
    rel_k = rearrange(rel_k, "n (h d) -> h n d", h=h)
    rel_logits = einsum("b h i d, h j d -> b h i j",
                        q + self.rel_pos_bias, rel_k)
    rel_logits = relative_shift(rel_logits)
    logits = content_logits + rel_logits
    attn = logits.softmax(dim=-1)

    # --- the one addition. The mean over heads is taken HERE, so a
    # layer holds (n, n) -- 9.4 MB fp32 at n=1536 -- rather than the
    # head-resolved (h, n, n) = 75 MB, i.e. 830 MB over 11 layers.
    if getattr(self, "_collect_attention", False):
        self._attn_head_mean = (
            attn.detach().mean(dim=1)[0].float().cpu().numpy())

    attn = self.attn_dropout(attn)
    out = einsum("b h i j, b h j d -> b h i d", attn, v)
    out = rearrange(out, "b h n d -> b n (h d)")
    return self.to_out(out)


def _pin_fp32(torch) -> dict:
    """Turn TF32 OFF for both convolutions and matmuls, and report what
    was found.

    A REPRODUCIBILITY CONTROL, NOT A PERFORMANCE KNOB, and silent when
    wrong. It matches `backend_borzoi._pin_fp32` deliberately: the two
    backends must make the SAME precision choice, or the
    Enformer-vs-Borzoi comparison acquires a hardware term, and that is
    the one comparison the "model is conflated with protocol" claim
    rests on.

    `precision_policy: port` and `map_precision: float32` declare that
    this arm computes in fp32, and TF32 is not fp32: it keeps fp32's
    exponent but truncates the mantissa to 10 bits. Enformer's trunk is
    convolutional, so `cudnn.allow_tf32` is the operative flag, not
    `matmul` -- and torch ships `cudnn.allow_tf32 = True` by DEFAULT
    while leaving `matmul` False, so a check that reads only `matmul`
    reports "TF32 off" on a machine where TF32 is fully live.

    The effect is small but real. With TF32 on, the three-bin K562 CAGE
    readout moves by ~4e-04 median, and the full 896 x 5,313 output by
    ~2e-04 median: far too small to move a benchmark whose intervals
    are +/-0.02, large enough to matter when one trunk is compared
    against another.
    """
    found = {
        "cudnn_allow_tf32_default": bool(torch.backends.cudnn.allow_tf32),
        "matmul_allow_tf32_default": bool(
            torch.backends.cuda.matmul.allow_tf32),
    }
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    found["tf32_pinned_off"] = True
    if torch.cuda.is_available():
        found["gpu_name"] = torch.cuda.get_device_name(0)
        found["gpu_capability"] = ".".join(
            str(v) for v in torch.cuda.get_device_capability(0))
        found["tf32_capable"] = torch.cuda.get_device_capability(0)[0] >= 8
    return found


class EnformerBackend:
    """Enformer via `enformer-pytorch`.

    One checkpoint, so `replicate` must be 1.
    """

    def __init__(self, meta, device: str = "cuda", target_length: int = 896,
                 use_tf_gamma: bool = True, checkpoint: str = CHECKPOINT,
                 checkpoint_trunk: bool = True):
        import torch
        from enformer_pytorch import from_pretrained
        from enformer_pytorch.modeling_enformer import Attention

        self.meta = meta
        self.device = device
        self.checkpoint = checkpoint
        self.use_tf_gamma = use_tf_gamma
        self._torch = torch
        # Before the first .to(device), so no kernel has yet been
        # selected under the unpinned flags.
        self._precision = _pin_fp32(torch)

        model = from_pretrained(checkpoint, use_tf_gamma=use_tf_gamma,
                                target_length=target_length)
        # .eval() is load-bearing and silent if forgotten:
        # dropout_rate=0.4, attn_dropout=0.05, and the conv tower is
        # BatchNorm1d. Training mode would randomise every score.
        self.model = model.eval().to(device)
        self._attn_modules = [m for m in self.model.modules()
                              if isinstance(m, Attention)]
        if not self._attn_modules:
            raise RuntimeError(
                "no Attention modules found; the package layout changed")
        for m in self._attn_modules:
            m.forward = types.MethodType(_patched_attention, m)
            m._collect_attention = False

        for m in self._attn_modules:
            assert getattr(m, "use_tf_gamma", None) == use_tf_gamma, \
                "use_tf_gamma did not propagate to the Attention modules"

        self.checkpoint_trunk = checkpoint_trunk
        if checkpoint_trunk:
            self._install_deep_checkpointing()
        # dU/dy bookkeeping, reported by provenance(): how many passes took
        # the closed form and how many the float64 central differences, and
        # the worst gate disagreement seen.
        self._dudy_analytic = 0
        self._dudy_numeric = 0
        self._dudy_gate_worst = 0.0
        self._dudy_rng = np.random.RandomState(0)

    #: The gene readout of the last gradient pass (see score_eg.Backend).
    _last_T = float("nan")
    _last_u = float("nan")

    def _install_deep_checkpointing(self):
        """Recompute activations in the stem and conv tower, not just
        the transformer.

        The package ships `trunk_checkpointed`, but it wraps only
        `self.transformer`, which is the cheap third of the trunk: peak
        memory is the same with it as without. What dominates is the
        stem's first `Conv1d(4 -> 768, k=15)` over 196,608 bp and the
        six conv-tower blocks, so those are wrapped too. The trade is
        roughly 1.4x compute for a peak that fits on a small card.

        Forward-only paths are unaffected: `use_checkpointing` is
        consulted only inside the gradient path, and `torch.no_grad()`
        makes checkpointing a no-op anyway.
        """
        from torch.utils.checkpoint import checkpoint, checkpoint_sequential
        from einops import rearrange

        def deep_trunk(self, x):
            x = rearrange(x, "b n d -> b d n")
            x = checkpoint(lambda t: self.stem(t), x, use_reentrant=False)
            for blk in self.conv_tower:
                x = checkpoint(lambda t, b=blk: b(t), x, use_reentrant=False)
            x = rearrange(x, "b d n -> b n d")
            x = checkpoint_sequential(self.transformer,
                                      len(self.transformer), x,
                                      use_reentrant=False)
            x = self.crop_final(x)
            x = self.final_pointwise(x)
            return x

        self.model.trunk_checkpointed = types.MethodType(deep_trunk,
                                                        self.model)
        self.model.use_checkpointing = True

    # ---------------------------------------------------------- helpers
    def _tensor(self, x: np.ndarray):
        t = self._torch.from_numpy(np.ascontiguousarray(x)).float()
        # Upstream `Enformer.forward` does `x.to(self.device)` and
        # discards the result, so the caller must move the tensor. That
        # one is at least loud: a device mismatch raises.
        return t.to(self.device)

    @staticmethod
    def _check_replicate(replicate: Any) -> None:
        if replicate not in (1, "1", "ENSEMBLE"):
            raise ValueError(
                f"Enformer is a single checkpoint; replicate={replicate!r}")

    def _predict(self, t):
        out = self.model(t[None, ...])
        y = out["human"] if isinstance(out, dict) else out
        return y[0]

    # ------------------------------------------------ Backend interface
    def forward(self, x: np.ndarray, replicate: Any) -> np.ndarray:
        """One-hot (L, 4) -> (896, 5313) predictions."""
        self._check_replicate(replicate)
        with self._torch.no_grad():
            y = self._predict(self._tensor(x))
        y = y.float().cpu().numpy()
        assert y.shape[0] == self.meta.n_bins, \
            f"expected {self.meta.n_bins} output bins, got {y.shape[0]}"
        return y

    def attention(self, x: np.ndarray, replicate: Any, how: str) -> np.ndarray:
        """Attention averaged over every head and every layer, which is
        Avsec 2021's `mean-heads-layers`.

        Returns the PRE-CROP (1536, 1536) token grid; see the module
        docstring.
        """
        self._check_replicate(replicate)
        if how != "mean-heads-layers":
            raise ValueError(
                f"attention_aggregation={how!r} is not implementable "
                "through this interface: `Backend.attention` returns one "
                "matrix, so a per-layer or per-head choice would need a "
                "layer/head index parameter that PARAMS does not define. "
                "Avsec 2021 uses mean-heads-layers.")
        for m in self._attn_modules:
            m._collect_attention = True
            m._attn_head_mean = None
        try:
            with self._torch.no_grad():
                self._predict(self._tensor(x))
            mats = [m._attn_head_mean for m in self._attn_modules]
            if any(a is None for a in mats):
                raise RuntimeError(
                    "the attention patch did not fire on every layer")
            A = np.mean(np.stack(mats, axis=0), axis=0)   # mean over layers
        finally:
            for m in self._attn_modules:
                m._collect_attention = False
                m._attn_head_mean = None
        n_tokens = self.meta.seq_len // self.meta.bin_size
        assert A.shape == (n_tokens, n_tokens), \
            ("attention must be on the uncropped token grid "
             f"({n_tokens}); got {A.shape}. The 896-bin grid would shift "
             "every enhancer offset by the crop width.")
        return A

    def scalar_gradient(
            self, x: np.ndarray, replicate: Any,
            scalar_fn: Callable[[np.ndarray], float]) -> np.ndarray:
        """d scalar_fn(forward(x)) / dx, as (L, 4).

        `scalar_fn` is GENE-SCALAR, which is numpy (it calls `unsquash`,
        `math.log`, `_reduce_axis`) and so cannot be differentiated by
        autograd. Rather than maintain a torch mirror of it -- two
        implementations of the most order-sensitive step in the
        algorithm, guaranteed to drift -- take the vector-Jacobian
        route: form dU/dy on exactly the entries GENE-SCALAR reads (it
        advertises them as `.bins` and `.tracks`, here 3 x 2 = 6 of
        4.76 M), then let autograd supply dy/dx in one backward pass.
        Exact for any smooth reduction in the schema, and GENE-SCALAR
        stays the single source of truth. `_dUdy_exact` forms dU/dy.
        """
        self._check_replicate(replicate)
        torch = self._torch
        t = self._tensor(x).requires_grad_(True)
        y = self._predict(t)
        y_np = y.detach().float().cpu().numpy()
        # The gene readout, as differentiated and un-logged; the frame
        # agreement gate reads T.
        self._last_u = float(scalar_fn(y_np))
        lin = getattr(scalar_fn, "linear", None)
        self._last_T = float(lin(y_np)) if lin is not None else float("nan")
        if not np.isfinite(scalar_fn(y_np)):
            raise ValueError(
                "the gene scalar is not finite on the reference sequence")

        bins = list(getattr(scalar_fn, "bins", range(y_np.shape[0])))
        tracks = list(getattr(scalar_fn, "tracks", range(y_np.shape[1])))
        dUdy = self._dUdy_exact(y_np, bins, tracks, scalar_fn)
        g, = torch.autograd.grad(
            y, t,
            grad_outputs=torch.from_numpy(dUdy).to(y.device).to(y.dtype))
        return g.detach().float().cpu().numpy()

    def _dUdy_exact(self, y_np: np.ndarray, bins, tracks,
                    scalar_fn) -> np.ndarray:
        """dU/dy on the readout block, in float64, and in closed form
        where the scalar allows.

        WHY NOT SIMPLY float32 CENTRAL DIFFERENCES. GENE-SCALAR's per-bin
        track reduction is a float32 sum when the predictions are
        float32. A step of 1e-4 on one entry of a sum whose float32
        spacing is 3e-5 (two K562 CAGE tracks at ~150 each) is resolved
        only to +-1 ulp, so a derivative that should read exactly 0.5
        (track_agg = mean) comes out anywhere between 0.46 and 0.53, and
        between 0.31 and 0.61 at TSS values near 500. For the 638-track
        `cell_type = all` readout the per-bin sum is ~5e4 with a spacing
        of 4e-3, forty times the step: most of the 1,914 entries come out
        exactly zero and the rest up to fifty times too large, so the map
        would be the gradient of a few arbitrarily chosen tracks. The
        Borzoi and AlphaGenome backends form dU/dy in a float64 torch
        mirror and are unaffected.

        TWO LAYERS, both exact to float64 round-off:

        1. the central differences run on a float64 copy (spacing 1e-11
           at 5e4, so h = 1e-4 is resolved to ~1e-7 relative, and the
           O(h^2) truncation error of a log is ~1e-9);
        2. when `scalar_fn.ops` says the scalar is linear-then-log with a
           sum or mean over tracks and no un-squash (Enformer's
           transform_generation is "none"), dU/dy is written in closed
           form: d_total x w_t on every (bin, track) of the readout, with
           w_t = 1 (sum) or 1/n_tracks (mean) and d_total = 1, or
           1/((total + eps) ln base) under log_scalar. It is GATED on
           every call: GATE-VALUE compares the closed-form U with
           `scalar_fn(y)` to 1e-9 relative, and GATE-DERIV
           central-differences four random entries through `scalar_fn`
           in float64 and compares them to 1e-6 relative.

        Any other reduction (max, the multi-assay sum of logs, per-track)
        takes layer 1. `SCORE_EG_ENF_DUDY=numeric` forces layer 1 for
        every call, as a comparison switch.
        """
        ops = getattr(scalar_fn, "ops", None)
        mode = os.environ.get("SCORE_EG_ENF_DUDY", "analytic").lower()
        y64 = np.asarray(y_np, dtype=np.float64)
        dUdy = np.zeros(y_np.shape, dtype=np.float64)
        closed = (mode != "numeric" and ops is not None
                  and not ops.get("multi") and not ops.get("per_track")
                  and ops.get("track_agg") in ("sum", "mean")
                  and (not ops.get("inverse_transform")
                       or ops.get("transform_generation") == "none"))
        if not closed:
            self._dudy_numeric += 1
            h = 1e-4
            # ONE scratch buffer, perturbed in place and restored, rather
            # than two full copies per entry: at `cell_type = all` the loop
            # runs 3 x 638 = 1,914 times.
            scratch = y64.copy()
            for b in bins:
                for k in tracks:
                    orig = scratch[b, k]
                    scratch[b, k] = orig + h
                    up = scalar_fn(scratch)
                    scratch[b, k] = orig - h
                    dn = scalar_fn(scratch)
                    scratch[b, k] = orig
                    dUdy[b, k] = (up - dn) / (2.0 * h)
            return dUdy.astype(np.float32)
        # ---- closed form
        w_t = 1.0 if ops["track_agg"] == "sum" else 1.0 / len(tracks)
        total = float(y64[np.ix_(bins, tracks)].sum() * w_t)
        if ops["log_scalar"]:
            arg = total + float(ops["pseudocount"]) + float(ops["eps"])
            if arg <= 0.0:
                raise ValueError("the gene scalar's log argument is not "
                                 "positive on the reference sequence")
            if ops["log_base"] == "2":
                u_mine, d_total = math.log(arg, 2), 1.0 / (arg * math.log(2.0))
            else:
                u_mine, d_total = math.log(arg), 1.0 / arg
        else:
            u_mine, d_total = total, 1.0
        u_spec = float(scalar_fn(y64))
        rel = abs(u_mine - u_spec) / max(1.0, abs(u_spec))
        self._dudy_gate_worst = max(self._dudy_gate_worst, rel)
        if not rel <= 1e-9:
            raise AssertionError(
                "GATE-VALUE (backend_enformer dU/dy): the closed-form gene "
                "scalar disagrees with score_eg.gene_scalar_fn: "
                f"mine={u_mine!r} spec={u_spec!r} rel={rel:.3e}. The ops "
                "advertised on the closure do not describe the reduction "
                "it performs.")
        val = d_total * w_t
        dUdy[np.ix_(bins, tracks)] = val
        # GATE-DERIV: four random entries by float64 central differences
        # through the closure itself.
        n_ent = len(bins) * len(tracks)
        h = 1e-4
        scratch = y64.copy()
        for f in self._dudy_rng.choice(n_ent, size=min(4, n_ent),
                                       replace=False):
            b, k = bins[int(f) // len(tracks)], tracks[int(f) % len(tracks)]
            orig = scratch[b, k]
            scratch[b, k] = orig + h
            up = scalar_fn(scratch)
            scratch[b, k] = orig - h
            dn = scalar_fn(scratch)
            scratch[b, k] = orig
            num = (up - dn) / (2.0 * h)
            drel = abs(num - val) / max(abs(val), 1e-300)
            self._dudy_gate_worst = max(self._dudy_gate_worst, drel)
            if not drel <= 1e-6:
                raise AssertionError(
                    "GATE-DERIV (backend_enformer dU/dy): the closed form "
                    "disagrees with a float64 central difference of "
                    f"score_eg.gene_scalar_fn at (bin={b}, track={k}): "
                    f"analytic={val:.6e} numeric={num:.6e} rel={drel:.3e}.")
        self._dudy_analytic += 1
        return dUdy.astype(np.float32)

    # ------------------------------------------------------- provenance
    def provenance(self) -> dict:
        """What ran, for the run manifest."""
        return {"backend": "enformer-pytorch", "checkpoint": self.checkpoint,
                "use_tf_gamma": self.use_tf_gamma, "device": self.device,
                # dU/dy is float64 and, where the scalar is linear-then-log,
                # closed form and gated (`_dUdy_exact`)
                "dudy": {"mode": os.environ.get("SCORE_EG_ENF_DUDY",
                                                "analytic").lower(),
                         "closed_form_calls": self._dudy_analytic,
                         "numeric_calls": self._dudy_numeric,
                         "gate_worst_rel": self._dudy_gate_worst,
                         "precision": "float64"},
                "checkpoint_trunk": self.checkpoint_trunk,
                "alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
                "target_length": self.meta.n_bins,
                "attention_grid": self.meta.seq_len // self.meta.bin_size}
