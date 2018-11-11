import numpy as np


class Simulator(object):
    """Simulator is an event generator.
    It generates events in form of (tau, r),
    where tau is the happening time (continuous) of that event
    and r is the involved region id.
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
            [(tau, r)]: tau is the rent timestamp (continuous), r is the region id
        """
        events = []
        for i in range(100):
            r = np.random.randint(0, 10)
            tau = np.random.random() * 10
            events.append((tau, r))
        return events

    def simulate_return_event(self, tau0, r0):
        """
        Args:
            tau0: the timestamp (continuous) when the bike is rent
            r0: the region where the bike is rent
        Returns:
            (tau, r): tau is the return timestamp (continuous), r is the region id
        """
        r = np.random.randint(0, 10)
        while r == r0:
            r = np.random.randint(0, 10)
        tau = tau0 + np.random.random() * 10
        return tau, r

    def simulate_reposition_event(self, tau0, r0, r1):
        """
        Args:
            tau0: the current time
            r0: the region the trike are
            r1: the region the trike will be
        Returns:
            (tau1, r1): tau1 the arrival time of the trike, r1 is the region the trike will be
        """
        d = self._data.get_distance(r0, r1)
        epsilon = np.random.random()
        # equation (6) in the paper
        tau1 = tau0 + d / self._mu_r + self._t_r + epsilon
        return tau1, r1
