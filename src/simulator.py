from contrib.core import combine_interfaces, get_periods
import numpy as np


class Simulator(object):
    def __init__(self, date, scale, episode, community, mu, tr, er, real):
        self.I = combine_interfaces(date, episode, community)
        self.scale = scale
        self.mu = mu
        self.tr = tr
        self.er = er
        self.periods = get_periods(date, episode)
        self.real = real
        self.limits = (self.I['limits'] * self.scale).astype(int)
        self.loads = np.round(0.5 * self.limits).astype(int)

        self.num_regions = len(self.limits)

        self._duration = self.I['duration']
        self._demands = self.I['demands']
        self._destination = self.I['destination']

        self.locations = self.I['locations']
        self.distance = self.I['distance']
        self.real_rents = self.I['real_rents']
        self.real_returns = self.I['real_returns']
        self.real_return_map = self.I['real_return_map']

        self.resample()

    def _sample_rent_return(self):
        # sample rents
        rents = []
        # for _ in range(self.scale):
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
        return_map = {tuple(k): tuple(v)
                      for k, v in zip(rents,
                                      returns)}

        return rents, returns, return_map

    def resample(self):
        if not self.real:  # the real data is sampled
            self.real_rents, self.real_returns, self.real_return_map\
                = self._sample_rent_return()

        self.estimated_rents, self.estimated_returns, self.estimated_return_map\
            = self._sample_rent_return()

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
        return self.real_rents

    def query_return(self, t, r):
        return self.real_return_map[(t, r)]

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
