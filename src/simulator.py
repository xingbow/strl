import numpy as np
import pandas as pd
import time
import OIModel as om
from datetime import datetime
from collections import defaultdict


def parse_timestamp(ts):
    return datetime.fromtimestamp(ts)


def get_episode_duration(ix):
    durations = [
        [[7, 11]],
        [[11, 12], [16, 17]],
        [[12, 16]],
        [[17, 18]],
        [[18, 23]],
    ]

    base = 1392825600
    d = durations[ix][0]

    episode = [base + d[0]*3600,
               base + d[1]*3600]

    return episode


def get_periods(episode):
    delta = 3600
    periods = []
    duration = get_episode_duration(episode)
    t = duration[0]
    while t < duration[1]:
        periods.append([t, t + delta])
        t += delta
    return periods


def extract_geo_info(df):
    tdf = df[['start station id',
              'start id',
              'rx1',
              'ry1']].drop_duplicates()

    tdf.columns = ['sid', 'rid', 'x', 'y']

    sldf = pd.read_csv('../data/stationStatus.csv')
    sldf.columns = ['sid', 'name', 'limit']
    del sldf['name']
    sldf['limit'] = sldf['limit'].fillna(0)

    sldf = pd.merge(tdf, sldf, how='left', on='sid')

    rldf = sldf.groupby('rid')[['limit']].sum().reset_index(level=0)
    gdf = pd.merge(tdf, rldf, how='left', on='rid')

    del gdf['sid']

    gdf = gdf.drop_duplicates()
    gdf = gdf.sort_values('rid')

    return gdf


class Simulator(object):
    def _init_interfaces(self, episode, community):
        # type in file loc
        transitionDataLoc = '../data/test.csv'  # transition file
        weatherDataLoc = '../data/weather.csv'  # weather info file

        # Initialize function, no need to modify
        hisInputData, transitionMatrixDuration, transitionMatrixDestination, weatherMatrix = om.read_inData(
            transitionDataLoc, weatherDataLoc)

        tdf = pd.read_csv(transitionDataLoc)
        tdf['community'] = 1

        tdf = tdf[(tdf['episode'] == episode) &
                  (tdf['community'] == community)]

        gdf = extract_geo_info(tdf)
        assert len(gdf['rid'].unique()) == len(gdf['rid'])

        self.rids = rids = gdf['rid'].values.astype(int)
        rid_to_ix = {v: i for i, v in enumerate(rids)}

        tdf['start id'] = tdf['start id'].replace(rid_to_ix)
        tdf['end id'] = tdf['end id'].replace(rid_to_ix)
        tdf['end timestamp'] = tdf['start timestamp'] + tdf['tripduration']

        self.periods = get_periods(episode)

        def demands(t, i):
            assert 0 <= i < len(rids)
            t = int(t)
            return om.get_expectDepartureNumber(
                t, rids[i],
                hisInputData,
                weatherMatrix)

        def destination(t, i):
            assert 0 <= i < len(rids)
            t = int(t)
            return rid_to_ix[int(om.get_predictedDestination(
                t, rids[i],
                transitionMatrixDuration,
                transitionMatrixDestination))]

        def duration(t, i, j):
            assert 0 <= i < len(rids) and 0 <= j < len(rids)
            t = int(t)
            return om.get_predictedDuration(
                t, rids[i], rids[j],
                transitionMatrixDuration,
                transitionMatrixDestination)

        self.real_rent_events = tdf[['start timestamp',
                                     'start id']].values.astype(int)

        self.real_return_events = tdf[[
            'end timestamp', 'end id']].values.astype(int)

        self.real_return_map = {tuple(k): tuple(v)
                                for k, v in zip(self.real_rent_events,
                                                self.real_return_events)}

        self.limits = np.round(gdf['limit'].values)

        self.estimated_rent_events = []
        for r in range(len(self.limits)):
            ts = demands(self.periods[0][0], r)
            self.estimated_rent_events += [(t, r) for t in ts]

        self.estimated_return_events = []
        for t, r in self.estimated_rent_events:
            r1 = destination(t, r)
            t1 = duration(t, r, r1)
            self.estimated_return_events += [(t1, r1)]

        self.estimated_return_map = {tuple(k): tuple(v)
                                     for k, v in zip(self.estimated_rent_events,
                                                     self.estimated_return_events)}

        self.locations = locations = gdf[['x', 'y']].values

        def distance_between(i, j):
            a = locations[i]
            b = locations[j]
            a, b = map(np.array, [a, b])
            return np.sum((a - b)**2)**0.5

        self.dist = dist = np.zeros([len(rids), len(rids)])
        for i in range(len(rids)):
            for j in range(i + 1, len(rids)):
                dist[i, j] = distance_between(i, j)
                dist[j, i] = dist[i, j]

        self._demands = demands
        self._destination = destination
        self._duration = duration

    def __init__(self, episode, community, mu, tr, er):
        self.mu = mu
        self.tr = tr
        self.er = er

        self._init_interfaces(episode, community)

    def estimate_likely_destination(self, t, r):
        """
        Needed when generate return event
        """
        return self._destination(t, r)

    def estimate_bike_arrival_time(self, t, r0, r1):
        """
        Needed when generate return event
        """
        return t + self._duration(t, r0, r1)

    def estimate_trike_arrival_time(self, t, r0, r1):
        """
        Needed when generate reposition
        """
        return t + self.dist[r0, r1] / self.mu + self.tr + np.random.normal(0, self.er)

    def estimate_rent_events(self, duration=None):
        """Estimate rent events in the next period
        Args:
            period: [a, b] the episode required
        Returns:
            [(t, r)]: t is the rent timestamp (continuous), r is the region id
        """
        duration = duration or [self.start_time, self.end_time]
        return [(t, r)
                for t, r in self.estimated_rent_events
                if duration[0] <= t < duration[1]]

    def estimate_return_events(self, duration=None):
        duration = duration or [self.start_time, self.end_time]
        return [(t, r)
                for t, r in self.estimated_return_events
                if duration[0] <= t < duration[1]]

    def estimate_return_event(self, t, r):
        return self.estimated_return_map[(t, r)]

    def query_rent_events(self):
        return self.real_rent_events

    def query_return_event(self, t, r):
        return self.real_return_map[(t, r)]

    def query_nearest_region(self, r):
        """
        Needed when handle the case when the station is full
        Args:
            r: region id
        Returns:
            nearest region
        """
        r1 = np.argmin(np.delete(self.dist[r], r))
        assert r1 != r
        return r1

    def get_limits(self):
        return self.limits

    @property
    def start_time(self):
        return self.periods[0][0]

    @property
    def end_time(self):
        return self.periods[-1][-1]

    @property
    def delta(self):
        return self.periods[0][1] - self.periods[0][0]


if __name__ == "__main__":
    num_trikes = 5
    capacity = 10
    num_epochs = 20
    batch_size = 32
    rho = 10
    mu = 200 / 60
    tr = 60 * 3
    er = 3 * 60

    config = {
        "num_epochs": num_epochs,
        "batch_size": batch_size,
    }

    simulator = Simulator(episode=1,
                          community=1,
                          mu=mu,
                          tr=tr,
                          er=er)
