"""A module for the graphical r process."""

import math
import numpy as np
import wnnet.net as wn


class GrRproc:
    """A class for handling graph-based r-process calculations.

    Args:
        ``xml_file``: An input xml file.

        ``nuc_xpath`` (:obj:`str`, optional): An XPath expression definining the
        r-process network.  Default is all species.

        ``n_bdn_max`` (:obj:`int`, optional): Maximum number of emitted
        beta-delayed neutrons.

    """

    def __init__(self, xml_file, nuc_xpath="", n_bdn_max=3):

        reac_xpath = "[ \
            (reactant = 'n' and product = 'gamma') or \
             (count(reactant) = 1 and \
             product = 'electron' and not(reactant = 'n'))]"

        self.net = wn.Net(xml_file, nuc_xpath=nuc_xpath, reac_xpath=reac_xpath)
        self.nucs = self.net.get_nuclides()

        assert "n" in self.nucs

        self.lims = self._set_limits(self.nucs)

        arr = [self.lims["z_max"] + 1, self.lims["n_max"] + 1]

        self.rates = {}
        self.rates["ncap"] = np.zeros(arr)
        self.rates["gamma"] = np.zeros(arr)
        self.rates["beta total"] = np.zeros(arr)

        arr.append(n_bdn_max + 1)
        self.rates["beta"] = np.zeros(arr)

        self.reactions = {}

        self.reactions["ncap"] = self.net.get_valid_reactions(
            nuc_xpath=nuc_xpath,
            reac_xpath="[reactant = 'n' and product = 'gamma']",
        )

        self.reaction_map = {}

        for key, value in self.reactions["ncap"].items():
            for reactant in value.nuclide_reactants:
                if reactant != "n" and reactant in self.nucs:
                    self.reaction_map[key] = (
                        "ncap",
                        self.nucs[reactant]["z"],
                        self.nucs[reactant]["n"],
                    )
                    self.reaction_map[key] = (
                        "gamma",
                        self.nucs[reactant]["z"],
                        self.nucs[reactant]["n"],
                    )

        self.reactions["beta"] = self.net.get_valid_reactions(
            nuc_xpath=nuc_xpath,
            reac_xpath="[count(reactant) = 1 and product = 'electron']",
        )

        for key, value in self.reactions["beta"].items():
            reactant = value.nuclide_reactants[0]
            if reactant in self.nucs:
                self.reaction_map[key] = (
                    "beta",
                    self.nucs[reactant]["z"],
                    self.nucs[reactant]["n"],
                )

    def _set_limits(self, nucs):
        lims = {}
        lims["min_n"] = {}
        lims["max_n"] = {}

        lims["z_min"] = math.inf
        lims["z_max"] = 0
        lims["n_max"] = 0

        for value in nucs.values():
            _z = value["z"]
            _n = value["n"]
            if 1 < _z < lims["z_min"]:
                lims["z_min"] = _z
            if _z > lims["z_max"]:
                lims["z_max"] = _z
            if _n > lims["n_max"]:
                lims["n_max"] = _n
            if _z in lims["min_n"]:
                if _n < lims["min_n"][_z]:
                    lims["min_n"][_z] = _n
            else:
                lims["min_n"][_z] = _n
            if _z in lims["max_n"]:
                if _n > lims["max_n"][_z]:
                    lims["max_n"][_z] = _n
            else:
                lims["max_n"][_z] = _n

        # Add Z = 1 to lower limit if multiple isotopes

        if lims["max_n"][1] and lims["max_n"][1] > 0:
            lims["z_min"] = 1

        return lims

    def get_net(self):
        """Method to return the network.

        Returns:
            A `wnnet <https://wnnet.readthedocs.io>`_ network object.

        """

        return self.net

    def get_z_lims(self):
        """Method to return the smallest and largest atomic numbers in the
        network.

        Returns:
            :obj:`tuple`: A tuple of two :obj:`int` objects.  The first element
            is the smallest atomic number present in the network (greater than
            one) and the second element is the largest atomic number.

        """

        return (self.lims["z_min"], self.lims["z_max"])

    def get_n_lims(self, z_c):
        """Method to return the smallest and largest neutron number in
        an isotopic chain in the network.

        Args:
            ``z_c`` (:obj:`int`): The atomic number giving the isotopic chain.

        Returns:
            :obj:`tuple`: A tuple whose first element is the smallest neutron
            number in the isotopic chain and whose second element is the
            largest neutron number in the isotopic chain.

        """

        z_min, z_max = self.get_z_lims()

        assert z_min <= z_c <= z_max

        return (self.lims["min_n"][z_c], self.lims["max_n"][z_c])

    def update_rates(self, t_9, rho):
        """Method to update the network reactions.

        Args:
            ``t_9`` (:obj:`float`): The temperature in billions of K.

            ``rho`` (:obj:`float`): The mass density in g/cc.

        Returns:
            On successful return, the network rates have been updated.

        """

        for key in self.reactions["ncap"]:
            rates = self.net.compute_rates_for_reaction(key, t_9, rho)
            r_map = self.reaction_map[key]
            self.rates["ncap"][r_map[1], r_map[2]] = rates[0] * rho
            self.rates["gamma"][r_map[1], r_map[2] + 1] = rates[1]

        for key, value in self.reactions["beta"].items():
            rates = self.net.compute_rates_for_reaction(key, t_9, rho)
            r_map = self.reaction_map[key]
            n_bdn = value.nuclide_products.count("n")
            self.rates["beta"][r_map[1], r_map[2], n_bdn] = rates[0]

        self.rates["beta total"] = np.sum(self.rates["beta"], axis=2)

    def compute_f_l(self, z_c, y_n, d_t):
        """Method to compute the F_L's.

        Args:
            ``z_c`` (:obj:`int`): The atomic number at which to compute F_L.

            ``y_n`` (:obj:`float`): The abundance of neutrons per nucleon.

            ``d_t`` (:obj:`float`): The time step (in seconds)

        Returns:
            :obj:`numpy.array`: A one-dimensional array containing the F_L's
            for the given *Z*.

        """

        result = np.zeros([self.lims["n_max"] + 1])

        lambda_ncap = self.rates["ncap"][z_c, :] * y_n * d_t
        lambda_gamma = self.rates["gamma"][z_c, :] * d_t
        lambda_beta_total = self.rates["beta total"][z_c, :] * d_t

        if z_c in self.lims["min_n"]:
            result[self.lims["min_n"][z_c]] = 1 / (
                1.0 + lambda_beta_total[self.lims["min_n"][z_c]]
            )
            for _n in range(self.lims["min_n"][z_c], self.lims["max_n"][z_c]):
                lambda_n_prime = lambda_ncap[_n] * result[_n]
                result[_n + 1] = (1.0 + lambda_n_prime) / (
                    (1.0 + lambda_beta_total[_n + 1]) * (1.0 + lambda_n_prime)
                    + lambda_gamma[_n + 1]
                )

        return result

    def compute_f_u(self, z_c, y_n, d_t):
        """Method to compute the F_U's.

        Args:
            ``z_c`` (:obj:`int`): The atomic number at which to compute F_L.

            ``y_n`` (:obj:`float`): The abundance of neutrons per nucleon.

            ``d_t`` (:obj:`float`): The time step (in seconds)

        Returns:
            :obj:`numpy.array`: A one-dimensional array containing the F_L's
            for the given *Z*.

        """

        result = np.zeros([self.lims["n_max"] + 1])

        lambda_ncap = self.rates["ncap"][z_c, :] * y_n * d_t
        lambda_gamma = self.rates["gamma"][z_c, :] * d_t
        lambda_beta_total = self.rates["beta total"][z_c, :] * d_t

        if z_c in self.lims["max_n"]:
            result[self.lims["max_n"][z_c]] = 1 / (
                1.0 + lambda_beta_total[self.lims["max_n"][z_c]]
            )
            for _n in range(
                self.lims["max_n"][z_c], self.lims["min_n"][z_c], -1
            ):
                lambda_g_prime = lambda_gamma[_n] * result[_n]
                result[_n - 1] = (1.0 + lambda_g_prime) / (
                    (1.0 + lambda_beta_total[_n - 1]) * (1.0 + lambda_g_prime)
                    + lambda_ncap[_n - 1]
                )

        return result

    def compute_y_l(self, z_c, y_0, y_n, d_t):
        """Method to compute the Y_L's.

        Args:
            ``z_c`` (:obj:`int`): The atomic number at which to compute F_L.

            ``y_0`` (:obj:`numpy.array`): A two-dimensional array giving
            the abundances to be used as input for each *Z* and *N*.

            ``y_n`` (:obj:`float`): The abundance of neutrons per nucleon.

            ``d_t`` (:obj:`float`): The time step (in seconds)

        Returns:
            :obj:`tuple`:  The first element of the tuple is a one-dimensional
            :obj:`numpy.array` array containing the Y_L's for the given *Z*.
            The second element is the F_L for the input *Z* that was used
            to compute the Y_L's.

        """

        y_l = np.zeros([self.lims["n_max"] + 1])

        f_l = self.compute_f_l(z_c, y_n, d_t)

        lambda_ncap = self.rates["ncap"][z_c, :] * y_n * d_t
        lambda_n_prime = np.multiply(lambda_ncap, f_l)
        lambda_gamma = self.rates["gamma"][z_c, :] * d_t
        lambda_beta_total = self.rates["beta total"][z_c, :] * d_t

        if z_c in self.lims["min_n"]:
            y_l[self.lims["min_n"][z_c]] = y_0[
                z_c, self.lims["min_n"][z_c]
            ] / (1.0 + lambda_beta_total[self.lims["min_n"][z_c]])
            for _n in range(self.lims["min_n"][z_c], self.lims["max_n"][z_c]):
                y_l[_n + 1] = (
                    (1.0 + lambda_n_prime[_n]) * y_0[z_c, _n + 1]
                    + lambda_ncap[_n] * y_l[_n]
                ) / (
                    (1.0 + lambda_beta_total[_n + 1])
                    * (1.0 + lambda_n_prime[_n])
                    + lambda_gamma[_n + 1]
                )

        return (y_l, f_l)

    def compute_y_u(self, z_c, y_0, y_n, d_t):
        """Method to compute the Y_U's.

        Args:
            ``z_c`` (:obj:`int`): The atomic number at which to compute F_L.

            ``y_0`` (:obj:`numpy.array`): A two-dimensional array giving
            the abundances to be used as input for each *Z* and *N*.

            ``y_n`` (:obj:`float`): The abundance of neutrons per nucleon.

            ``d_t`` (:obj:`float`): The time step (in seconds)

        Returns:
            :obj:`tuple`:  The first element of the tuple is a one-dimensional
            :obj:`numpy.array` array containing the Y_U's for the given *Z*.
            The second element is the F_U for the input *Z* that was used
            to compute the F_U's.


        """

        y_u = np.zeros([self.lims["n_max"] + 1])

        f_u = self.compute_f_u(z_c, y_n, d_t)

        lambda_ncap = self.rates["ncap"][z_c, :] * y_n * d_t
        lambda_gamma = self.rates["gamma"][z_c, :] * d_t
        lambda_g_prime = np.multiply(lambda_gamma, f_u)
        lambda_beta_total = self.rates["beta total"][z_c, :] * d_t

        if z_c in self.lims["max_n"]:
            y_u[self.lims["max_n"][z_c]] = y_0[
                z_c, self.lims["max_n"][z_c]
            ] / (1.0 + lambda_beta_total[self.lims["max_n"][z_c]])
            for _n in range(
                self.lims["max_n"][z_c], self.lims["min_n"][z_c], -1
            ):
                y_u[_n - 1] = (
                    (1.0 + lambda_g_prime[_n]) * y_0[z_c, _n - 1]
                    + lambda_gamma[_n] * y_u[_n]
                ) / (
                    (1.0 + lambda_beta_total[_n - 1])
                    * (1.0 + lambda_g_prime[_n])
                    + lambda_ncap[_n - 1]
                )

        return (y_u, f_u)

    def compute_r(self, z_c, y_0, y_n, d_t):
        """Method to compute the R_L's and R_U's.

        Args:
            ``z_c`` (:obj:`int`): The atomic number at which to compute F_L.

            ``y_0`` (:obj:`numpy.array`): A two-dimensional array giving
            the abundances to be used as input for each *Z* and *N*.

            ``y_n`` (:obj:`float`): The abundance of neutrons per nucleon.

            ``d_t`` (:obj:`float`): The time step (in seconds)

        Returns:
            :obj:`tuple`:  The first element of the tuple is a one-dimensional
            :obj:`numpy.array` array containing the R_L's for the given *Z*.
            The second element is a :obj:`numpy.array` array containing the
            R_U's for the given *Z*.


        """

        r_l = np.zeros([self.lims["n_max"] + 1])
        r_u = np.zeros([self.lims["n_max"] + 1])

        if z_c not in self.lims["max_n"]:
            return (r_l, r_u)

        y_l, f_l = self.compute_y_l(z_c, y_0, y_n, d_t)
        y_u, f_u = self.compute_y_u(z_c, y_0, y_n, d_t)

        lambda_ncap = self.rates["ncap"][z_c, :] * y_n
        lambda_gamma = self.rates["gamma"][z_c, :]

        for _n in range(self.lims["min_n"][z_c], self.lims["max_n"][z_c]):
            denom = f_u[_n + 1] * y_l[_n] + f_l[_n] * y_u[_n + 1]
            if lambda_gamma[_n + 1] * denom > 0:
                r_l[_n] = y_l[_n] / (lambda_gamma[_n + 1] * denom)
            if lambda_ncap[_n] * denom > 0:
                r_u[_n + 1] = y_u[_n + 1] / (lambda_ncap[_n] * denom)

        return (r_l, r_u)

    def compute_y(self, y_t, y_n, d_t, method="graph"):
        """Method to compute the Y's.

        Args:
            ``y_t`` (:obj:`numpy.array`): A two-dimensional array giving
            the abundances to be used as input for each *Z* and *N*.

            ``y_n`` (:obj:`float`): The abundance of neutrons per nucleon.

            ``d_t`` (:obj:`float`): The time step (in seconds)

            ``method`` (:obj:`string`, optional): Keyword to select between
            solving the isotopic abundances from recursive graph relations
            (*graph*--the default) or from standard matrix operations
            (*matrix*).

        Returns:
            :obj:`numpy.array`: A two-dimensional array containing the Y's
            for each *Z* and *N*.

        """

        assert method in ("graph", "matrix")

        if method == "graph":
            return self._compute_y_with_graph(y_t, y_n, d_t)
        return self._compute_y_with_matrix(y_t, y_n, d_t)

    def _solve_isotope_chain_with_graph(self, _z, y_0, y_n, d_t):
        z_result = np.zeros(self.lims["n_max"] + 1)
        y_l, f_l = self.compute_y_l(_z, y_0, y_n, d_t)
        y_u, f_u = self.compute_y_u(_z, y_0, y_n, d_t)

        lambda_ncap = self.rates["ncap"][_z, :] * y_n * d_t
        lambda_n_prime = np.multiply(lambda_ncap, f_l)
        lambda_gamma = self.rates["gamma"][_z, :] * d_t
        lambda_g_prime = np.multiply(lambda_gamma, f_u)
        lambda_beta_total = self.rates["beta total"][_z, :] * d_t

        z_result[self.lims["min_n"][_z]] = y_u[self.lims["min_n"][_z]]
        z_result[self.lims["max_n"][_z]] = y_l[self.lims["max_n"][_z]]

        for _n in range(self.lims["min_n"][_z] + 1, self.lims["max_n"][_z]):
            z_result[_n] = (
                (1.0 + lambda_g_prime[_n + 1])
                * (1.0 + lambda_n_prime[_n - 1])
                * y_0[_z, _n]
                + lambda_ncap[_n - 1]
                * (1.0 + lambda_g_prime[_n + 1])
                * y_l[_n - 1]
                + lambda_gamma[_n + 1]
                * (1.0 + lambda_n_prime[_n - 1])
                * y_u[_n + 1]
            ) / (
                (1.0 + lambda_beta_total[_n])
                * (
                    (1.0 + lambda_n_prime[_n - 1])
                    * (1.0 + lambda_g_prime[_n + 1])
                )
                + lambda_ncap[_n] * (1.0 + lambda_n_prime[_n - 1])
                + lambda_gamma[_n] * (1.0 + lambda_g_prime[_n + 1])
            )

        return z_result

    def _compute_y_with_graph(self, y_t, y_n, d_t):
        result = np.zeros([self.lims["z_max"] + 1, self.lims["n_max"] + 1])

        result[0, 1] = y_n

        if self.lims["z_min"] > 1:
            result[1, 0] = y_t[1, 0]

        y_0 = y_t.copy()

        z_min, z_max = self.get_z_lims()

        for _z in range(z_min, z_max + 1):
            result[_z, :] = self._solve_isotope_chain_with_graph(
                _z, y_0, y_n, d_t
            )

            if _z < z_max:
                for _n in range(len(result[_z, :])):
                    for n_bdn in range(self.rates["beta"].shape[2]):
                        if _n + 1 + n_bdn < len(result[_z, :]):
                            y_0[_z + 1, _n] += (
                                self.rates["beta"][_z, _n + 1 + n_bdn, n_bdn]
                                * result[_z, _n + 1 + n_bdn]
                                * d_t
                            )

        return result

    def _tridiag(self, a_mat, rhs):
        for i in range(len(rhs) - 1):
            if a_mat[0, i + 1] != 0 and a_mat[1, i] != 0:
                fac = -a_mat[0, i + 1] / a_mat[1, i]
                a_mat[1, i + 1] += fac * a_mat[2, i]
                rhs[i + 1] += fac * rhs[i]

        sol = np.zeros(len(rhs))

        for i in reversed(range(len(rhs))):
            sol[i] = rhs[i] / a_mat[1, i]
            if i < len(rhs) - 1:
                sol[i] -= a_mat[2, i] * sol[i + 1] / a_mat[1, i]

        return sol

    def _compute_y_with_matrix(self, y_0, y_n, d_t):
        _y = np.zeros([y_0.shape[0], y_0.shape[1]])
        z_tup = self.get_z_lims()

        for _z in range(z_tup[0], z_tup[1] + 1):
            # Find the limits on the isotope chain and the resulting width

            n_tup = self.get_n_lims(_z)
            _l = n_tup[1] - n_tup[0] + 1

            # Initialize the tridiagonal matrix and the right-hand-side vector

            a_mat = np.zeros([3, _l])
            rhs = np.zeros(_l)

            # Find the limits on the lower isotope chain

            if _z > z_tup[0]:
                n_l_tup = self.get_n_lims(_z - 1)

            # Loop on the isotopes

            for i in range(_l):
                a_mat[1, i] = (
                    1
                    + self.rates["ncap"][_z, n_tup[0] + i] * y_n * d_t
                    + self.rates["gamma"][_z, n_tup[0] + i] * d_t
                    + self.rates["beta total"][_z, n_tup[0] + i] * d_t
                )
                rhs[i] = y_0[_z, n_tup[0] + i]
                if i > 0:
                    a_mat[0, i] = (
                        -self.rates["ncap"][_z, n_tup[0] + i - 1] * y_n * d_t
                    )
                if i < _l - 1:
                    a_mat[2, i] = (
                        -self.rates["gamma"][_z, n_tup[0] + i + 1] * d_t
                    )

                if _z > z_tup[0]:
                    for n_bdn in range(self.rates["beta"].shape[2]):
                        if n_tup[0] + i + 1 + n_bdn <= n_l_tup[1]:
                            rhs[i] += (
                                self.rates["beta"][
                                    _z - 1, n_tup[0] + i + 1 + n_bdn, n_bdn
                                ]
                                * _y[_z - 1, n_tup[0] + i + 1 + n_bdn]
                                * d_t
                            )

            # Solve the matrix equation and update the abundances

            sol = self._tridiag(a_mat, rhs)

            for i in range(_l):
                _y[_z, n_tup[0] + i] = sol[i]

            _y[0, 1] = y_n

            if z_tup[0] > 1:
                _y[1, 0] = y_0[1, 0]

        return _y

    def compute_dyndt(self, y_current, y_n):
        """Method to compute the rate of change of the free neutron abundance
        in the network.

        Args:
            ``y_current`` (:obj:`numpy.array`): A two-dimensional array giving
            the abundances to be used as the current abundances for each *Z*
            and *N*.

            ``y_n`` (:obj:`float`): The abundance of neutrons per nucleon.

        Returns:
            :obj:`float`: The rate of change of the free neutron abundance.

        """

        result = 0

        lambda_ncap = self.rates["ncap"]
        lambda_gamma = self.rates["gamma"]
        lambda_beta = self.rates["beta"]

        for _z in self.lims["min_n"].keys():
            for _n in range(self.lims["min_n"][_z], self.lims["max_n"][_z]):
                result += (
                    -lambda_ncap[_z, _n] * y_n * y_current[_z, _n]
                    + lambda_gamma[_z, _n + 1] * y_current[_z, _n + 1]
                )

        for _z in self.lims["min_n"].keys():
            for _n in range(
                self.lims["min_n"][_z], self.lims["max_n"][_z] + 1
            ):
                for n_bdn in range(1, lambda_beta.shape[2]):
                    result += (
                        n_bdn * lambda_beta[_z, _n, n_bdn] * y_current[_z, _n]
                    )

        return result

    def get_rates(self):
        """Method to return the current rates.

        Returns:
            :obj:`dict`: A dictionary of current rates for valid reactions.
            The dictionary entries are themselves two-dimensional
            :obj:`numpy.array`, each with the given rate type indexed by
            *Z* and *N*.

        """

        return self.rates
