"""fam_screen_Gasperini2019.py -- family screen-Gasperini2019: the Gasperini2019 column of the screens figure
(Supplementary Fig. 16, Appendix E), 49 tests. The Gschwind K562 benchmark stratified by the CRISPRi screen each pair
comes from; every comparison of Figs. 3, 4 (with the ISM rows of Supplementary Fig. 13) and 5 on its own figure's
support, the paired component bootstrap of the full cohort masked to support x screen. Computation:
lib_d.compute_all() (shared by the three screen families; see lib_d)."""
import lib_d

SCREEN = "Gasperini2019"


def compute():
    return lib_d.family(SCREEN)


def values():
    return lib_d.family_values(SCREEN)
