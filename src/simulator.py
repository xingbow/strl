from contrib.core import combine_interfaces, get_periods
from contrib.OIModel import NON_EMPTY_PROPORTION
import numpy as np
import pickle
import random


class Simulator(object):
    def __init__(self, date, episode, community, mu, tr, er, scale):
        self.I = combine_interfaces(date, episode, community)

        self.scale = scale
        self.mu = mu
        self.tr = tr
        self.er = er
        self.periods = get_periods(date, episode)

        self.limits = (self.I['limits'] * self.scale).astype(int)
        self.loads = np.round(0.75 * self.limits).astype(int)

        self.num_regions = len(self.limits)

        self._duration = self.I['duration']
        self._demands = self.I['demands']
        self._destination = self.I['destination']

        self.locations = self.I['locations']
        self.distance = self.I['distance']

        self.test_rents = self.I['real_rents']
        # since there are no customer loss
        # we resample it by assuming there are some
        # time the station is empty
        # whose propotion is (1 - NON_EMPTY_PROPOTION)
        num_rents = int(len(self.test_rents) / (NON_EMPTY_PROPORTION))
        ix = np.random.randint(0, len(self.test_rents), num_rents)
        print(len(ix), num_rents)
        self.test_rents = self.test_rents[ix]
        # resample finished
        self.test_trips = self.I['real_trips']

        self.resample()
        self.switch_mode(True)

    def _sample_trips(self):
        # sample rents
        rents = []

        for _ in range(2):
            for period in self.periods:
                for r in range(self.num_regions):
                    ts = self._demands(period[0], r)
                    rents += [(t, r) for t in ts]

        # estimate returns for each rent
        returns = []
        for t, r in rents:
            r1 = self._destination(t, r)
            t1 = t + self._duration(t, r, r1)
            returns += [(t1, r1)]

        # map rent to return
        trips = {tuple(k): tuple(v)
                 for k, v in zip(rents,
                                 returns)}

        return rents, returns, trips

    def _print_rent_demands(self, rents, info):
        ts, rs = zip(*rents)
        u, c = np.unique(rs, return_counts=True)
        print(info, len(rents), dict(zip(u, c)))

    def switch_mode(self, train):
        if train:
            self.env_rents = self.train_rents
            self.env_trips = self.train_trips
        else:
            self.env_rents = self.test_rents
            self.env_trips = self.test_trips

        self._print_rent_demands(self.env_rents, "running rents")
        self._print_rent_demands(self.estimated_rents, "estimate rents")

    def resample(self):
        self.train_rents, self.train_returns, self.train_trips = self._sample_trips()
        self.estimated_rents, self.estimated_returns, self.estimated_trips = self._sample_trips()

    def estimate_bike_arrival_time(self, t, r0, r1):
        return t + self._duration(t, r0, r1)

    def estimate_trike_arrival_time(self, t, r0, r1):
        return t + self.distance[r0, r1] / self.mu + self.tr + np.random.normal(0, self.er)

    def estimate_demands(self, duration=None):
        duration = duration or [self.start_time, self.end_time]
        o = np.zeros(self.num_regions)
        for t, r in self.estimated_rents:
            o[r] += 1
        for t, r in self.estimated_returns:
            o[r] -= 1
        return o

    def query_rents(self):
        return self.env_rents

    def query_return(self, t, r):
        return self.env_trips[(t, r)]

    def query_nearest_region(self, r):
        """
        Needed when handle the case when the station is full
        Args:
            r: region id
        Returns:
            nearest region
        """
        distance = np.array(self.distance[r])
        distance[r] = np.inf
        r1 = np.argmin(distance)
        assert r1 != r
        return r1

    @property
    def start_time(self):
        return self.periods[0][0]

    @property
    def end_time(self):
        return self.periods[-1][-1]

    @property
    def delta(self):
        return self.periods[0][1] - self.periods[0][0]
