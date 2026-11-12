"""The frame guard: no augmentation can leave the gene's readout behind, on any of the three models.

Run with:

    python3 tests/test_frame_guard.py

Needs only numpy and PyYAML, like `score_eg.py --self-test`: the probe runs on each model's real geometry
with invented weights, so no model, weights or genome are loaded.

Under a reverse-complement augmentation the model sees the reverse-complemented input, so its output bin b is
genomic bin (n_bins - 1 - b) and its '+' tracks carry the '-' strand. The readout must be read in that frame.
Reading it in the forward frame instead reads the gene's mirror image on the antisense tracks: severe for a
stranded readout (Borzoi RNA-seq at exons), mild for an unstranded 3-bin readout (Enformer CAGE at the TSS,
where it lands one 128 bp bin off). These tests pin the four layers of the guard on Enformer, Borzoi and
AlphaGenome and, the part that matters, show that each layer FAILS when that error, or one of its cousins,
is introduced:

  * the exact probe (layer 3) passes on every model x readout region x augmentation x method in use, and
    catches the forward-frame readout, an un-swapped strand, a one-bin offset, an un-reversed map and a
    misplaced ISM lesion;
  * a readout built without the pass's augmentation raises inside a model pass (layer 2);
  * the shift rule and the contract refuse what cannot be made exact (layer 1);
  * the real-model agreement gate (layer 4) stops a run whose rc readout departs from its forward readout;
  * every shipped config with a reverse-complement augmentation passes the contract and the probe.
"""

import csv
import glob
import os
import sys
from types import SimpleNamespace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import score_eg as S  # noqa: E402

CONFIGS = {   # one per model x readout region, all shipped configs
    ("enformer", "tss"): ("unified-t1-backbone-enformer.yaml", {"anchor": "single-tss"}),
    ("enformer", "all-tss"): ("unified-t1-backbone-enformer-alltss.yaml", {}),
    ("borzoi", "exons"): ("linder2025-borzoi-gradient-flowfish-plus-authors.yaml", {}),
    ("borzoi", "tss"): ("unified-t1-backbone-borzoi.yaml", {}),
    ("borzoi", "all-tss"): ("unified-t1-backbone-borzoi.yaml", {"region": "all-tss"}),
    ("alphagenome", "exons"): ("avsec2026-alphagenome-gradient-gschwind.yaml", {}),
    ("alphagenome", "tss"): ("unified-t1-backbone-alphagenome.yaml", {}),
    # both strands (strand_mask off) at the window centre, as the matched ISM protocol reads them: with
    # H['+'] + H['-'] the same for every base the probe would read a constant here and see nothing
    ("borzoi", "tss, both strands"): ("unified-t1-backbone-borzoi-ism.yaml", {}),
    ("alphagenome", "tss, both strands"): ("unified-t1-backbone-alphagenome-ism.yaml", {"strand_mask": "off"}),
}
TTAS = (("rc", None), ("rc+shift", 3), ("avsec8", 3))


def _theta(key, **over):
    cfg, base = CONFIGS[key]
    th = S.load_config(os.path.join(ROOT, "configs", cfg))
    th.update(base)
    th.update(over)
    return S.fill_defaults(th)


def _fresh():
    S._FRAME_SELFTEST_DONE.clear()


def _expect_frame_error(fn, what):
    try:
        fn()
    except S.FrameError:
        return
    raise AssertionError(f"the frame guard did not catch: {what}")


def _with_patched(name, repl, fn):
    orig = getattr(S, name)
    setattr(S, name, repl)
    try:
        return fn()
    finally:
        setattr(S, name, orig)


def test_probe_passes_every_model_readout_tta_and_method():
    n = 0
    for key in CONFIGS:
        for tta, k in TTAS:
            for method in ("gradient", "ism"):
                _fresh()
                th = _theta(key, tta=tta, tta_shift_bp=k, method=method)
                rep = S.frame_selftest(S.MODEL_META[key[0]], th)
                assert rep["status"] == "passed" and rep["checks"] > 0, (key, tta, method, rep)
                n += 1
    # attention: Enformer's token grid is the only one the probe materialises
    _fresh()
    th = _theta(("enformer", "tss"), tta="rc", method="attention")
    assert S.frame_selftest(S.MODEL_META["enformer"], th)["status"] == "passed"
    print(f"    probe passed on {n + 1} model x readout x tta x method combinations")


def test_probe_catches_the_forward_frame_readout_on_every_model():
    """The readout left in the forward frame on the rc pass."""
    for key in CONFIGS:
        _fresh()
        th = _theta(key, tta="rc")
        meta = S.MODEL_META[key[0]]
        forward_frame = lambda gene, aug, meta: S.ReadoutFrame(gene.strand, 0, False)   # noqa: E731
        _with_patched("frame_for", forward_frame, lambda: _expect_frame_error(
            lambda: S.frame_selftest(meta, th), f"{key}: forward-frame readout on the rc pass"))


def test_probe_catches_a_mirror_without_the_strand_swap():
    for key in CONFIGS:
        th = _theta(key, tta="rc")
        rho = S.resolve(th, S.MODEL_META[key[0]])
        if not any(s in ("+", "-") for s in rho.track_strands):
            continue                                                  # unstranded: nothing to swap
        _fresh()
        half = lambda gene, aug, meta: S.ReadoutFrame(gene.strand, 0, bool(aug and aug.rc))  # noqa: E731
        _with_patched("frame_for", half, lambda: _expect_frame_error(
            lambda: S.frame_selftest(S.MODEL_META[key[0]], th), f"{key}: mirrored bins, unswapped strand"))


def test_probe_catches_a_one_bin_offset():
    """The unstranded case of the forward-frame readout: a 3-bin readout one bin off."""
    orig = S.frame_for

    def off_by_one(gene, aug, meta):
        fr = orig(gene, aug, meta)
        return S.ReadoutFrame(fr.strand, fr.offset + (1 if aug is not None and aug.rc else 0), fr.mirror)
    for key in CONFIGS:
        _fresh()
        th = _theta(key, tta="rc")
        _with_patched("frame_for", off_by_one, lambda: _expect_frame_error(
            lambda: S.frame_selftest(S.MODEL_META[key[0]], th), f"{key}: one-bin offset"))


def test_probe_catches_an_unreversed_map():
    for key in (("enformer", "tss"), ("borzoi", "exons"), ("alphagenome", "tss")):
        _fresh()
        th = _theta(key, tta="rc", method="gradient")
        bad = lambda m, a: np.roll(m, -a.shift, axis=0)                 # noqa: E731  forgets the reversal
        _with_patched("unaugment", bad, lambda: _expect_frame_error(
            lambda: S.frame_selftest(S.MODEL_META[key[0]], th), f"{key}: map not un-reversed"))


def test_probe_catches_a_misplaced_ism_lesion():
    for key in (("enformer", "tss"), ("borzoi", "exons"), ("alphagenome", "tss")):
        _fresh()
        th = _theta(key, tta="rc", method="ism")
        bad = lambda i, j, a, length: (i, j)                            # noqa: E731  forgets the augmentation
        _with_patched("augment_interval", bad, lambda: _expect_frame_error(
            lambda: S.frame_selftest(S.MODEL_META[key[0]], th), f"{key}: ISM lesion in the forward frame"))


def test_a_forgotten_aug_raises_inside_a_model_pass():
    th = _theta(("borzoi", "exons"), tta="rc")
    meta = S.MODEL_META["borzoi"]
    rho = S.resolve(th, meta)
    w, genes = S._probe_genes(meta, meta.bin_size)
    g = genes[0]
    S.gene_scalar_fn(g, w, th, rho, aug=None)                          # outside a pass: the forward reference
    rc = S.Aug(0, True)
    with S.augmented_frame(rc):
        S.gene_scalar_fn(g, w, th, rho, aug=rc)                        # the pass's own frame: fine
        _expect_frame_error(lambda: S.gene_scalar_fn(g, w, th, rho), "aug omitted inside an rc pass")
        _expect_frame_error(lambda: S.gene_scalar_fn(g, w, th, rho, aug=S.Aug(3, True)),
                            "a different augmentation inside an rc pass")
    with S.augmented_frame(S.Aug(0, False)):
        S.gene_scalar_fn(g, w, th, rho, aug=None)                      # None is the identity


def test_shift_rule():
    enf, brz, ag = (S.MODEL_META[m] for m in ("enformer", "borzoi", "alphagenome"))
    assert S.shift_bins(3, enf) == 0 and S.shift_bins(43, enf) == 0 and S.shift_bins(-43, enf) == 0
    assert S.shift_bins(128, enf) == 1 and S.shift_bins(-256, enf) == -2
    assert S.shift_bins(3, brz) == 0 and S.shift_bins(32, brz) == 1
    assert S.shift_bins(3, ag) == 3 and S.shift_bins(-5, ag) == -5
    for sh, meta in ((64, enf), (100, enf), (16, brz), (43, brz)):
        _expect_frame_error(lambda: S.shift_bins(sh, meta), f"ambiguous {sh} bp shift on {meta.name}")


def test_contract_refusals():
    def stub(model):
        return SimpleNamespace(meta=S.MODEL_META[model])
    # a 43 bp shift on Borzoi's 32 bp bins
    _expect_frame_error(lambda: S.check_frame_contract(_theta(("borzoi", "exons"), tta="shift", tta_shift_bp=43),
                                                       stub("borzoi")), "43 bp shift on Borzoi")
    S.check_frame_contract(_theta(("enformer", "tss"), tta="rc+shift", tta_shift_bp=43), stub("enformer"))
    # per-track ISM on stranded tracks under rc
    th = _theta(("borzoi", "exons"), tta="rc", method="ism", track_abs_order="before")
    _expect_frame_error(lambda: S.check_frame_contract(th, stub("borzoi")), "per-track stranded ISM under rc")
    S.check_frame_contract(_theta(("enformer", "tss"), tta="rc", method="ism", track_abs_order="before"),
                           stub("enformer"))


def test_agreement_gate():
    th = _theta(("borzoi", "exons"), tta="rc")
    meta = S.MODEL_META["borzoi"]
    augs = (S.Aug(0, False), S.Aug(0, True))
    rng = np.random.RandomState(1)
    ok = S.FrameAgreementGate(th)
    for i in range(40):
        t = float(rng.lognormal(5, 1))
        ok.observe(f"g{i}", augs, {(0, 0, 0): t, (0, 0, 1): t * float(rng.lognormal(0, 0.05))}, meta)
    ok.check()
    assert ok.report()["status"] == "passed"
    bad = S.FrameAgreementGate(th)

    def feed():
        for i in range(40):
            t = float(rng.lognormal(5, 1))
            bad.observe(f"g{i}", augs, {(0, 0, 0): t, (0, 0, 1): t / 16.0}, meta)   # a 16-fold rc disagreement
    _expect_frame_error(feed, "an rc readout 16x off the forward readout")
    assert S.FrameAgreementGate(_theta(("borzoi", "exons"), tta="none")).report()["status"] == "inactive"


def test_aug_has_only_mapped_fields():
    assert tuple(f.name for f in S._dc_fields(S.Aug)) == S._FRAME_FIELDS == ("shift", "rc")


def test_every_shipped_rc_config_passes_contract_and_probe():
    """Every protocol that reverse-complements must be certified, not refused: each shipped config, and each
    ledger row whose overrides add a reverse-complement augmentation to a config."""
    def certify(th, what):
        meta = S.MODEL_META[th["model"]]
        S.check_frame_contract(th, SimpleNamespace(meta=meta))
        _fresh()
        assert S.frame_selftest(meta, th)["status"] == "passed", what

    def has_rc(th):
        return th.get("tta") not in (None, "none") and ("rc" in th["tta"] or th["tta"] == "avsec8")

    n = 0
    for f in sorted(glob.glob(os.path.join(ROOT, "configs", "*.yaml"))):
        th = S.fill_defaults(S.load_config(f))
        if has_rc(th):
            certify(th, f)
            n += 1
    assert n >= 6, n
    m = 0
    with open(os.path.join(ROOT, "reproducing.tsv")) as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            if not row["config"].endswith(".yaml") or "tta=" not in row["overrides"]:
                continue
            th = S.fill_defaults(S.apply_overrides(S.load_config(os.path.join(ROOT, row["config"])),
                                                   row["overrides"].split()))
            if has_rc(th):
                certify(th, row["result"])
                m += 1
    assert m >= 4, m
    print(f"    {n} shipped rc-augmented configs and {m} rc-augmented ledger rows certified")


def _main():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        print(f"{fn.__name__} ...", flush=True)
        fn()
    print(f"OK: {len(fns)} tests")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
