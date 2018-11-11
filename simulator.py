import numpy as np


class Simulator(object):
    """Simulator is an event generator.
    It generates events in form of (tau, s),
    where tau is the happening time (continuous) of that event
    and s is the involved station id.
    """

    def __init__(self, data, delta_1, mu_r, t_r):
        """
        Args:
            data: provided by xingbo
            delta_1: the period length
            mu_r: trike speed
            t_r: loading time
        """
        self._data = data
        self._delta_1 = delta_1
        self._mu_r = mu_r
        self._t_r = t_r

    def simulate_rent_events(self, t):
        """
        Args:
            t: period begining time
        Returns:
            [(tau, s)]: tau is the rent timestamp (continuous), s is the station id
        """
        events = []
        for i in range(100):
            s = np.random.randint(0, 10)
            tau = np.random.random() * 10
            events.append((tau, s))
        return events

    def simulate_return_event(self, tau0, s0):
        """
        Args:
            tau0: the timestamp (continuous) when the bike is rent
            s0: the station where the bike is rent
        Returns:
            (tau, s): tau is the return timestamp (continuous), s is the station id
        """
        s = np.random.randint(0, 10)
        while s == s0:
            s = np.random.randint(0, 10)
        tau = tau0 + np.random.random() * 10
        return tau, s

    def simulate_reposition_event(self, tau0, s0, s1):
        """
        Args:
            tau0: the current time
            s0: the station the trike are
            s1: the station the trike will be
        Returns:
            (tau1, s1): tau1 the arrival time of the trike, s1 is the station the trike will be
        """
        d = self._data.get_distance(s0, s1)
        epsilon = np.random.random()
        # equation (6) in the paper
        tau1 = tau0 + d / self._mu_r + self._t_r + epsilon
        return tau1, s1
