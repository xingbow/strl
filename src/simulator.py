import numpy as np
import pandas as pd
import time
import OIModel as om

from collections import defaultdict


def get_episode_duration(ix):
    durations = [
        [[7, 11]],
        [[11, 12], [16, 17]],
        [[12, 16]],
        [[17, 18]],
        [[18, 23]],
    ]

    base = 1391209200
    d = durations[ix][0]

    return [base + d[0]*3600,
            base + d[1]*3600]


def get_periods(episode):
    delta = 3600
    periods = []
    duration = get_episode_duration(episode)
    t = duration[0]
    while t < duration[1]:
        periods.append([t, t + delta])
        t += delta
    return periods


class Simulator(object):
    def _init_interfaces(self, episode, community):
        # type in file loc
        transitionDataLoc = '../data/test.csv'  # transition file
        weatherDataLoc = '../data/weather.csv'  # weather info file

        # Initialize function, no need to modify
        hisInputData, transitionMatrixDuration, transitionMatrixDestination, weatherMatrix = om.read_inData(
            transitionDataLoc, weatherDataLoc)

        gdf = pd.read_csv('../data/geo.csv')
        gdf = gdf[(gdf['episode'] == episode) &
                  (gdf['community'] == community)]

        gdf = gdf.sort_values('rid')

        self.rids = rids = gdf['rid'].values
        rid_to_ix = {v: i for i, v in enumerate(rids)}

        points = gdf[['x', 'y']].values

        def distance_between(i, j):
            a = points[i]
            b = points[j]
            a, b = map(np.array, [a, b])
            return np.sum((a - b)**2)**0.5

        self.dist = dist = np.ones([len(rids), len(rids)]) * np.inf
        for i in range(len(rids)):
            for j in range(i + 1, len(rids)):
                dist[i, j] = distance_between(i, j)
                dist[j, i] = dist[i, j]

        self.limits = gdf['limit']

        def demands(t, i):
            assert 0 <= i < len(rids)
            return om.get_expectDepartureNumber(
                t, rids[i],
                hisInputData,
                weatherMatrix)

        def destination(t, i):
            assert 0 <= i < len(rids)
            return rid_to_ix[om.get_predictedDestination(
                t, rids[i],
                transitionMatrixDuration,
                transitionMatrixDestination)[0]]

        def duration(t, i, j):
            assert 0 <= i < len(rids) and 0 <= j < len(rids)
            return om.get_predictedDuration(
                t, rids[i], rids[j],
                transitionMatrixDuration,
                transitionMatrixDestination)

        self._demands = demands
        self._destination = destination
        self._duration = duration
        self.periods = get_periods(episode)

    def __init__(self, episode, community, mu, tr, er):
        self.mu = mu
        self.tr = tr
        self.er = er

        self._init_interfaces(episode, community)

    def get_likely_destination(self, t, r):
        """
        Needed when generate return event
        """
        return self._destination(t, r)

    def get_bike_arrival_time(self, t, r0, r1):
        """
        Needed when generate return event
        """
        return t + self._duration(t, r0, r1)

    def get_trike_arrival_time(self, t, r0, r1):
        """
        Needed when generate reposition
        """
        return t + self.get_distance(r0, r1) / self.mu + self.tr + np.random.normal(0, self.er)

    def get_nearest_region(self, r):
        """
        Needed when handle the case when the station is full
        Args:
            r: region id
        Returns:
            nearest region
        """
        r1 = np.argmin(self.dist[r])
        assert r1 != r
        return r1

    def get_distance(self, r0, r1):
        """
        Needed when estimate time for trike reposition
        """
        assert 0 <= r0 < len(self.rids) and 0 <= r1 < len(self.rids)
        return self.dist[r0, r1]

    def get_limits(self):
        return self.limits

    @property
    def rent_events(self):
        """
        Args:
            episode: [a, b] the episode required
        Returns:
            [(t, r)]: t is the rent timestamp (continuous), r is the region id
        """
        ret = []
        for period in self.periods:
            for i in range(len(self.limits)):
                taus = self._demands(period[0] + 1, i)
                ret += [(tau, i) for tau in taus]
        return ret

    @property
    def start_time(self):
        return self.periods[0][0]

    @property
    def end_time(self):
        return self.periods[-1][-1]
