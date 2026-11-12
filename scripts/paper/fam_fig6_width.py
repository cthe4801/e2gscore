"""fam_fig6_width.py -- Fig. 6, gene readout width block (Section 4.6).

Family `fig6-width`: Enformer's readout over 1, 2 and 5 bins of 128 bp against the matched protocol's 3 bins
(N2, L1, N3, each minus A1w on the pairs both reach). All three are quoted in Section 4.6, so Supplementary
Table 2 lists them under `text:` ids. Estimator: lib_c.fig6_pair (lib_c.subset_eval, tie-aware AP); see
fam_fig6_anchor for the bootstrap.
"""
from __future__ import annotations

import lib_c as L

FAMILY = "fig6-width"
C = L.CTRL
ROWS = [("text:Enformer: 1-bin readout vs 3 bins", (C, L.FIG6_POINT["N2"][1]), True),
        ("text:Enformer: 2-bin readout vs 3 bins", (C, L.FIG6_POINT["L1"][1]), True),
        ("text:Enformer: 5-bin readout vs 3 bins", (C, L.FIG6_POINT["N3"][1]), True)]


def compute() -> list:
    return [{k: t[k] for k in ("id", "delta", "lo", "hi", "p")} for t in L.point_tests(ROWS)]


def values() -> dict:
    return L.point_values(L.point_tests(ROWS))
