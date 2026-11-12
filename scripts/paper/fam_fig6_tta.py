"""fam_fig6_tta.py -- Fig. 6, test-time augmentation block (Section 4.6).

Family `fig6-tta`: seven Enformer contrasts, all on the pairs both runs reach:
    O4 - A1w   reverse complement                      in the text and a figure row
    Z1 - A1w   shift -43/0/+43 bp                      in the text and a figure row
    AV8 - A1w  Avsec et al.'s 8 augmentations          in the text and a figure row
    AV8 - O4   avsec8 against rc alone                 in the text
    V1 - A1w   shift -3/0/+3 bp                        figure row
    S1 - A1w   rc and shift -3/0/+3 bp                 figure row
    Z2 - A1w   rc and shift -43/0/+43 bp               figure row
Estimator: lib_c.fig6_pair (lib_c.subset_eval, tie-aware AP); see fam_fig6_anchor for the bootstrap.
"""
from __future__ import annotations

import lib_c as L

FAMILY = "fig6-tta"
C = L.CTRL
P = {k: v[1] for k, v in L.FIG6_POINT.items()}
ROWS = [("text:Enformer: reverse complement (rc) vs none", (C, P["O4"]), True),
        ("text:Enformer: shift -43/0/+43 bp vs none", (C, P["Z1"]), True),
        ("text:Enformer: avsec8 augmentation vs none", (C, P["AV8"]), True),
        ("text:Enformer: avsec8 augmentation vs rc alone", (P["O4"], P["AV8"]), True),
        ("fig6-tta:Enformer: shift -3/0/+3 bp vs none", (C, P["V1"]), False),
        ("fig6-tta:Enformer: rc & shift -3/0/+3 bp vs none", (C, P["S1"]), False),
        ("fig6-tta:Enformer: rc & shift -43/0/+43 bp vs none", (C, P["Z2"]), False)]


def compute() -> list:
    return [{k: t[k] for k in ("id", "delta", "lo", "hi", "p")} for t in L.point_tests(ROWS)]


def values() -> dict:
    return L.point_values(L.point_tests(ROWS))
