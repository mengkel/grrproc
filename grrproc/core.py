"""A module for the graphical r process."""

import math
import numpy as np


class GrRproc:
    """A class for handling graph-based r-process calculations.

    Args:
        ``net``: A `wnnet <https://wnnet.readthedocs.io>`_ network object.

        ``nuc_xpath`` (:obj:`str`, optional): An XPath expression definining the
        r-process network.

    """

    def __init__(self, net, nuc_xpath=""):
        self.net = net
        self.nucs = self.net.get_nuclides(nuc_xpath=nuc_xpath)

        self.z_array = []

        for key, value in self.nucs.items():
            if value["z"] not in self.z_array:
                self.z_array.append(value["z"])

        self.z_array.sort()

        self.lims = self._set_limits(self.nucs)

        arr = [self.lims["z_max"] + 1, self.lims["n_max"] + 1]

        self.rates = {}
        self.rates["ncap"] = np.zeros(arr)
        self.rates["gamma"] = np.zeros(arr)
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
            if _z < lims["z_min"] and _z != 0:
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

        return lims

    def get_z_lims(self):
        """Method to return the smallest and largest atomic numbers in the
        network.

        Returns:
            :obj:`tuple`: A tuple of two :obj:`int` objects.  The first element
            is the smallest atomic number present in the network (other than
            that for the neutron) and the second element is the largest
            atomic number.

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

        return (self.lims["min_n"][z_c], self.lims["max_n"][z_c])

    def update_rates(self, t_9, rho):
        """Method to update the network reactions.

        Args:
            ``t_9`` (:obj:`float`): The temperature in billions of K.

            ``rho`` (:obj:`float`): The mass density in g/cc.

        Returns:
            On successful return, the network rates have been updated.

        """

        for key in self.reactions["beta"]:
            rates = self.net.compute_rates_for_reaction(key, t_9, rho)
            r_map = self.reaction_map[key]
            self.rates["beta"][r_map[1], r_map[2]] = rates[0]

        for key in self.reactions["ncap"]:
            rates = self.net.compute_rates_for_reaction(key, t_9, rho)
            r_map = self.reaction_map[key]
            self.rates["ncap"][r_map[1], r_map[2]] = rates[0] * rho
            self.rates["gamma"][r_map[1], r_map[2] + 1] = rates[1]

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
        lambda_beta = self.rates["beta"][z_c, :] * d_t

        if z_c in self.lims["min_n"]:
            result[self.lims["min_n"][z_c]] = 1 / (
                1.0 + lambda_beta[self.lims["min_n"][z_c]]
            )
            for _n in range(self.lims["min_n"][z_c], self.lims["max_n"][z_c]):
                lambda_n_prime = lambda_ncap[_n] * result[_n]
                result[_n + 1] = (1.0 + lambda_n_prime) / (
                    (1.0 + lambda_beta[_n + 1]) * (1.0 + lambda_n_prime)
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
        lambda_beta = self.rates["beta"][z_c, :] * d_t

        if z_c in self.lims["max_n"]:
            result[self.lims["max_n"][z_c]] = 1 / (
                1.0 + lambda_beta[self.lims["max_n"][z_c]]
            )
            for _n in range(
                self.lims["max_n"][z_c], self.lims["min_n"][z_c], -1
            ):
                lambda_g_prime = lambda_gamma[_n] * result[_n]
                result[_n - 1] = (1.0 + lambda_g_prime) / (
                    (1.0 + lambda_beta[_n - 1]) * (1.0 + lambda_g_prime)
                    + lambda_ncap[_n - 1]
                )

        return result

    def compute_y_l(self, z_c, y_tilde, y_n, d_t):
        """Method to compute the Y_L's.

        Args:
            ``z_c`` (:obj:`int`): The atomic number at which to compute F_L.

            ``y_tilde`` (:obj:`numpy.array`): A two-dimensional array giving
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
        lambda_beta = self.rates["beta"][z_c, :] * d_t

        if z_c in self.lims["min_n"]:
            y_l[self.lims["min_n"][z_c]] = y_tilde[
                z_c, self.lims["min_n"][z_c]
            ] / (1.0 + lambda_beta[self.lims["min_n"][z_c]])
            for _n in range(self.lims["min_n"][z_c], self.lims["max_n"][z_c]):
                y_l[_n + 1] = (
                    (1.0 + lambda_n_prime[_n]) * y_tilde[z_c, _n + 1]
                    + lambda_ncap[_n] * y_l[_n]
                ) / (
                    (1.0 + lambda_beta[_n + 1]) * (1.0 + lambda_n_prime[_n])
                    + lambda_gamma[_n + 1]
                )

        return (y_l, f_l)

    def compute_y_u(self, z_c, y_tilde, y_n, d_t):
        """Method to compute the Y_U's.

        Args:
            ``z_c`` (:obj:`int`): The atomic number at which to compute F_L.

            ``y_tilde`` (:obj:`numpy.array`): A two-dimensional array giving
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
        lambda_beta = self.rates["beta"][z_c, :] * d_t

        if z_c in self.lims["max_n"]:
            y_u[self.lims["max_n"][z_c]] = y_tilde[
                z_c, self.lims["max_n"][z_c]
            ] / (1.0 + lambda_beta[self.lims["max_n"][z_c]])
            for _n in range(
                self.lims["max_n"][z_c], self.lims["min_n"][z_c], -1
            ):
                y_u[_n - 1] = (
                    (1.0 + lambda_g_prime[_n]) * y_tilde[z_c, _n - 1]
                    + lambda_gamma[_n] * y_u[_n]
                ) / (
                    (1.0 + lambda_beta[_n - 1]) * (1.0 + lambda_g_prime[_n])
                    + lambda_ncap[_n - 1]
                )

        return (y_u, f_u)

    def compute_r(self, z_c, y_tilde, y_n, d_t):
        """Method to compute the R_L's and R_U's.

        Args:
            ``z_c`` (:obj:`int`): The atomic number at which to compute F_L.

            ``y_tilde`` (:obj:`numpy.array`): A two-dimensional array giving
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

        y_l, f_l = self.compute_y_l(z_c, y_tilde, y_n, d_t)
        y_u, f_u = self.compute_y_u(z_c, y_tilde, y_n, d_t)

        lambda_ncap = self.rates["ncap"][z_c, :] * y_n
        lambda_gamma = self.rates["gamma"][z_c, :]

        for _n in range(self.lims["min_n"][z_c], self.lims["max_n"][z_c]):
            denom = f_u[_n + 1] * y_l[_n] + f_l[_n] * y_u[_n + 1]
            if lambda_gamma[_n + 1] * denom > 0:
                r_l[_n] = y_l[_n] / (lambda_gamma[_n + 1] * denom)
            if lambda_ncap[_n] * denom > 0:
                r_u[_n + 1] = y_u[_n + 1] / (lambda_ncap[_n] * denom)

        return (r_l, r_u)

    def compute_y(self, y_t, y_n, d_t):
        """Method to compute the Y's.

        Args:
            ``y_t`` (:obj:`numpy.array`): A two-dimensional array giving
            the abundances to be used as input for each *Z* and *N*.

            ``y_n`` (:obj:`float`): The abundance of neutrons per nucleon.

            ``d_t`` (:obj:`float`): The time step (in seconds)

        Returns:
            :obj:`numpy.array`: A two-dimensional array containing the Y_L's
            for each *Z* and *N*.

        """

        result = np.zeros([self.lims["z_max"] + 1, self.lims["n_max"] + 1])

        result[0, 1] = y_n

        y_tilde = y_t.copy()

        for _z in self.z_array:
            if _z in self.lims["min_n"] and _z != 0:

                y_l, f_l = self.compute_y_l(_z, y_tilde, y_n, d_t)
                y_u, f_u = self.compute_y_u(_z, y_tilde, y_n, d_t)

                lambda_ncap = self.rates["ncap"][_z, :] * y_n * d_t
                lambda_n_prime = np.multiply(lambda_ncap, f_l)
                lambda_gamma = self.rates["gamma"][_z, :] * d_t
                lambda_g_prime = np.multiply(lambda_gamma, f_u)
                lambda_beta = self.rates["beta"][_z, :] * d_t

                for _n in range(
                    self.lims["min_n"][_z], self.lims["max_n"][_z]
                ):
                    result[_z, _n] = (
                        (1.0 + lambda_g_prime[_n + 1])
                        * (1.0 + lambda_n_prime[_n - 1])
                        * y_tilde[_z, _n]
                        + lambda_ncap[_n - 1]
                        * (1.0 + lambda_g_prime[_n + 1])
                        * y_l[_n - 1]
                        + lambda_gamma[_n + 1]
                        * (1.0 + lambda_n_prime[_n - 1])
                        * y_u[_n + 1]
                    ) / (
                        (1.0 + lambda_beta[_n])
                        * (
                            (1.0 + lambda_n_prime[_n - 1])
                            * (1.0 + lambda_g_prime[_n + 1])
                        )
                        + lambda_ncap[_n] * (1.0 + lambda_n_prime[_n - 1])
                        + lambda_gamma[_n] * (1.0 + lambda_g_prime[_n + 1])
                    )

                _n_last = self.lims["max_n"][_z]
                result[_z, _n_last] = (
                    (1.0 + lambda_n_prime[_n_last - 1]) * y_tilde[_z, _n_last]
                    + lambda_ncap[_n_last - 1] * y_l[_n_last - 1]
                ) / (
                    (1.0 + lambda_beta[_n_last])
                    * (1.0 + lambda_n_prime[_n_last - 1])
                    + lambda_gamma[_n_last]
                )

                if _z < self.z_array[len(self.z_array) - 1]:
                    d_y = np.multiply(lambda_beta, result[_z, :])

                    for _n in range(1, d_y.shape[0]):
                        y_tilde[_z + 1, _n - 1] += d_y[_n]

        return result

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

        for _z in self.lims["min_n"].keys():
            for _n in range(self.lims["min_n"][_z], self.lims["max_n"][_z]):
                result += (
                    -lambda_ncap[_z, _n] * y_n * y_current[_z, _n]
                    + lambda_gamma[_z, _n + 1] * y_current[_z, _n + 1]
                )

        return result

    def get_abundances_from_zone(self, zone):
        """Method to return the abundances from a zone.

        Args:
            ``zone``: A `wnutils <https://wnutils.readthedocs.io>`_ zone object.

        Returns:
            :obj:`numpy.array`: A two-dimensional array containing the abundance
            from the input zone indexed by *Z* and *N*.

        """

        result = np.zeros([self.lims["z_max"] + 1, self.lims["n_max"] + 1])

        for key, value in zone["mass fractions"].items():
            _z = key[1]
            _a = key[2]
            result[_z, _a - _z] = float(value) / float(_a)

        return result

    def get_previous_abundances_from_zone(self, zone):
        """Method to return the previous timestep abundances from a zone.

        Args:
            ``zone``: A `wnutils <https://wnutils.readthedocs.io>`_ zone object.

        Returns:
            :obj:`numpy.array`: A two-dimensional array containing the previous
            timestep abundances from the input zone indexed by *Z* and *N*.

        """

        result = np.zeros([self.lims["z_max"] + 1, self.lims["n_max"] + 1])

        for key, value in zone["properties"].items():
            if isinstance(key, tuple):
                if key[0] == "y_previous":
                    nuc = self.nucs[key[1]]
                    result[nuc["z"], nuc["n"]] = value

        return result

    def assign_mass_fractions_to_zone(self, y_c, zone, y_previous=None):
        """Method to update the mass fractions in a zone with the input
        abundances.

        Args:
            ``y_c`` (:obj:`numpy.array`): A two-dimensional array giving
            the abundances indexed as *Z* and *N*.

            ``zone``: A `wnutils <https://wnutils.readthedocs.io>`_ zone object.

        Returns:
            On successful return, the zone mass fractions have been updated.

        """

        zone["mass fractions"].clear()

        sp_map = {}
        for key, value in self.net.get_nuclides().items():
            sp_map[(value["z"], value["n"])] = key

        for _z in range(y_c.shape[0]):
            for _n in range(y_c.shape[1]):
                if y_c[_z, _n] != 0:
                    if (_z, _n) in sp_map:
                        key = (sp_map[(_z, _n)], _z, _z + _n)
                        zone["mass fractions"][key] = y_c[_z, _n] * (_z + _n)

        if y_previous is not None:
            for _z in range(y_previous.shape[0]):
                for _n in range(y_previous.shape[1]):
                    if y_previous[_z, _n] != 0:
                        if (_z, _n) in sp_map:
                            key = ("y_previous", sp_map[(_z, _n)])
                            zone["properties"][key] = y_previous[_z, _n]

    def get_rates(self):
        """Method to return the current rates.

        Returns:
            :obj:`dict`: A dictionary of current rates for valid reactions.
            The dictionary entries are themselves two-dimensional
            :obj:`numpy.array`, each with the given rate type indexed by
            *Z* and *N*.

        """

        return self.rates
