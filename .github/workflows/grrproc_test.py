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

def test_g_up():
    nuc_xpath = "[(a = 1) or (z >= 26)]"

    net = wn.net.Net(
        io.BytesIO(requests.get("https://osf.io/kyhbs/download").content),
        nuc_xpath=nuc_xpath,
    )

    r = grp.GrRproc(net)

    r.update_rates(1., 1.e4)

    G = r.compute_g_up(26, 1.e-4, 1.e-2)

    eps = 1.e-6
    for n in range(*r.get_n_lims(26)):
        my_sum = 0
        for z in G:
            my_sum += np.sum(G[z][:,n], axis=0)
        assert 1 - eps < my_sum < 1 + eps

def test_g_down():
    nuc_xpath = "[(a = 1) or (z >= 26)]"

    net = wn.net.Net(
        io.BytesIO(requests.get("https://osf.io/kyhbs/download").content),
        nuc_xpath=nuc_xpath,
    )

    r = grp.GrRproc(net)

    r.update_rates(1., 1.e4)

    z_l, z_u = r.get_z_lims()

    G = r.compute_g_down(z_u, 1.e-4, 1.e-2)

    for _z in G:
        assert np.all(G[_z] >= 0) and np.all(G[_z] <= 1)

def test_g_both():
    nuc_xpath = "[(a = 1) or (z >= 26)]"

    net = wn.net.Net(
        io.BytesIO(requests.get("https://osf.io/kyhbs/download").content),
        nuc_xpath=nuc_xpath,
    )

    z_c = 40

    r = grp.GrRproc(net)

    r.update_rates(1., 1.e4)

    G_down = r.compute_g_down(z_c, 1.e-4, 1.e-2, z_lower=z_c)
    G_up = r.compute_g_up(z_c, 1.e-4, 1.e-2, z_upper=z_c)

    for _z in G_down:
        assert np.array_equal(G_down[_z], G_up[_z])

    for _z in G_up:
        assert np.array_equal(G_up[_z], G_down[_z])
