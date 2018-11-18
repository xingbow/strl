import numpy as np


class ArtificialSimulator(object):
    def __init__(self, mu, tr, er):
        self.mu = mu
        self.tr = tr
        self.er = er
        self.periods = [[i * 3600, (i+1) * 3600] for i in range(5)]

        self.num_regions = 4
        self.expected_demands = np.array([1, 50, 1, 50])
        self.limits = np.array([100, 100, 100, 100])

        self.locations = np.array([[0, 0], [0, 1000],
                                   [1000, 1000], [1000, 0]])

        self.distance = np.zeros([self.num_regions,
                                  self.num_regions])

        for i in range(self.num_regions):
            for j in range(i + 1, self.num_regions):
                a = self.locations[i]
                b = self.locations[j]
                a, b = map(np.array, [a, b])
                self.distance[i, j] = np.sum((a - b)**2)**0.5
        self.distance += self.distance.T

        self.loads = np.round(0.5 * self.limits).astype(int)

        self.resample()

    def _duration(self, t, r0, r1):
        # 4m/s, +- 100s
        ret = self.distance[r0, r1] / 4 + np.random.normal(0, 20)
        assert ret > 0
        return ret

    def _destination(self, t, r):
        d = np.random.randint(self.num_regions)
        while d == r:
            d = np.random.randint(self.num_regions)
        return d

    def _demands(self, duration, r):
        # number -> [t1, t2, t3, t4] uniform
        # exponential
        lambda_ = 1 / self.expected_demands[r]
        tau = duration[0]
        ret = []
        while tau < duration[1]:
            tau += np.random.exponential(lambda_) * self.delta
            ret.append(tau)
        return ret

    def _sample_rent_return(self):
        # sample rents
        rents = []

        for period in self.periods:
            for r in range(self.num_regions):
                ts = self._demands(period, r)
                rents += [(t, r) for t in ts]

        # estimate returns for each rent
        returns = []
        for t, r in rents:
            r1 = self._destination(t, r)
            t1 = t + self._duration(t, r, r1)
            returns += [(t1, r1)]

        # sort by time, used for quick refer
        rents = sorted(rents, key=lambda x: x[0])
        returns = sorted(returns, key=lambda x: x[0])

        # map rent to return
        return_map = {tuple(k): tuple(v)
                      for k, v in zip(rents,
                                      returns)}

        return rents, returns, return_map

    def resample(self):
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

        def vectorize(events):
            ts, rs = zip(*events)
            l = np.searchsorted(ts, duration[0])
            r = np.searchsorted(ts, duration[1]) - 1

            o = np.zeros(self.num_regions)
            for i in range(l, r):
                o[rs[i]] += 1
            return o

        rents = vectorize(self.estimated_rents)
        returns = vectorize(self.estimated_rents)

        return rents - returns

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
