import requests, io
import grrproc as grp
import numpy as np
from numpy import linalg as LA
import wnnet as wn


def test_grrproc():
    nuc_xpath = "[(a = 1) or (z >= 26 and z <= 40)]"

    net = wn.net.Net(
        io.BytesIO(requests.get("https://osf.io/kyhbs/download").content),
        nuc_xpath=nuc_xpath,
    )

    r = grp.GrRproc(net)

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

    net = wn.net.Net(
        io.BytesIO(requests.get("https://osf.io/kyhbs/download").content),
        nuc_xpath=nuc_xpath,
    )

    r = grp.GrRproc(net)

    assert len(r.get_net().get_nuclides()) > 0

    assert len(r.get_net().get_reactions()) > 0

def test_beta():
    nuc_xpath = "[(a = 1) or (z >= 26 and z <= 40)]"

    net = wn.net.Net(
        io.BytesIO(requests.get("https://osf.io/kyhbs/download").content),
        nuc_xpath=nuc_xpath,
    )

    r = grp.GrRproc(net)

    Lambda = r.compute_beta_matrix(30, 1.)

    assert np.any(Lambda)

def test_m():
    nuc_xpath = "[(a = 1) or (z = 30)]"

    net = wn.net.Net(
        io.BytesIO(requests.get("https://osf.io/kyhbs/download").content),
        nuc_xpath=nuc_xpath,
    )

    r = grp.GrRproc(net)

    r.update_rates(1., 1.e4)

    M = r.compute_m(30, 1.e-4, 1.e-2)

    my_sum = np.sum(M, axis=0)

    eps = 1.e-6
    for n in range(*r.get_n_lims(30)):
        assert 1 - eps < my_sum[n] < 1 + eps
