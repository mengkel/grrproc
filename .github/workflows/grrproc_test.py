import requests, io
import wnnet.net as wn
import grrproc as grp
import numpy as np
from numpy import linalg as LA


def test_grrproc():
    nuc_xpath = "[(a = 1) or (z >= 26 and z <= 40)]"

    r = grp.GrRproc(
        io.BytesIO(requests.get("https://osf.io/kyhbs/download").content),
        nuc_xpath=nuc_xpath,
    )

    t9 = 2.0
    rho = 1.0e5
    y_n = 6.5e-1
    d_t = 1.0e-10

    z_min, z_max = r.get_z_lims()
    n_min, n_max = r.get_n_lims(z_max)
    y0 = np.zeros([z_max + 1, n_max + 1])

    z = 26
    a = 56
    y0[z, a - z] = (1.0 - y_n) / a

    r.update_rates(t9, rho)

    y_g = r.compute_y(y0, y_n, d_t, method="graph")
    y_m = r.compute_y(y0, y_n, d_t, method="matrix")

    d_y = y_g - y_m

    assert LA.norm(d_y) < 1.0e-10

def test_net():
    nuc_xpath = "[(a = 1) or (z >= 26 and z <= 40)]"

    r = grp.GrRproc(
        io.BytesIO(requests.get("https://osf.io/kyhbs/download").content),
        nuc_xpath=nuc_xpath,
    )

    assert len(r.get_net().get_nuclides()) > 0

    assert len(r.get_net().get_reactions()) > 0

