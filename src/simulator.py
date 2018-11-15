import numpy as np
import pandas as pd
import time
import OIModel as om
from datetime import datetime
from collections import defaultdict
import pickle
import os


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

    base = 1379606400
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

        startTime = 1375286400  # xingbo bound
        timeStampBound = 1379606400  # 2013/9/29

        dayNumBound = (timeStampBound - startTime) // (24 * 3600)

        try:
            with open('../data/tmp.pkl', 'rb') as f:
                hisInputData, transitionMatrixDuration,\
                    transitionMatrixDestination, weatherMatrix, \
                    overAllTest, pred_y, dayNumBound = pickle.load(f)
        except:
            with open('../data/tmp.pkl', 'wb') as f:
                hisInputData, transitionMatrixDuration, transitionMatrixDestination,\
                    weatherMatrix, entireSituation, weatherMatrixAll\
                    = om.read_inData(transitionDataLoc, weatherDataLoc)
                overAllTest, pred_y, dayNumBound = om.predicted_overallSituation(
                    dayNumBound, entireSituation, weatherMatrixAll)
                pickle.dump([hisInputData, transitionMatrixDuration,
                             transitionMatrixDestination, weatherMatrix,
                             overAllTest, pred_y, dayNumBound], f)

        tdf = pd.read_csv(transitionDataLoc)
        tdf = tdf[(tdf['episode'] == episode) &
                  (tdf['community'] == community)]

        gdf = extract_geo_info(tdf)
        assert len(gdf['rid'].unique()) == len(gdf['rid'])

        self.rids = rids = gdf['rid'].values.astype(int)
        rid_to_ix = {v: i for i, v in enumerate(rids)}

        tdf['start timestamp'] += 1372608000
        tdf['start id'] = tdf['start id'].replace(rid_to_ix)
        tdf['end id'] = tdf['end id'].replace(rid_to_ix)
        tdf['end timestamp'] = tdf['start timestamp'] + tdf['tripduration']

        def demands(t, i):
            assert 0 <= i < len(rids)
            t = int(t)
            return om.get_expectDepartureNumber(
                t, rids[i], hisInputData,
                weatherMatrix, overAllTest, pred_y, dayNumBound)

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
            try:
                return om.get_predictedDuration(
                    t, rids[i], rids[j],
                    transitionMatrixDuration,
                    transitionMatrixDestination)
            except:
                return 0

        self._demands = demands
        self._destination = destination
        self._duration = duration

        tdf = tdf[(tdf['start timestamp'] >= self.start_time)
                  & (tdf['start timestamp'] < self.end_time)]

        tdf = tdf[(tdf['end timestamp'] >= self.start_time)
                  & (tdf['end timestamp'] < self.end_time)]

        tdf = tdf[tdf['end id'].isin(tdf['start id'])]

        self.real_rent_events = tdf[['start timestamp',
                                     'start id']].values.astype(int)

        self.real_return_events = tdf[[
            'end timestamp', 'end id']].values.astype(int)

        self.real_return_map = {tuple(k): tuple(v)
                                for k, v in zip(self.real_rent_events,
                                                self.real_return_events)}

        self.limits = np.round(gdf['limit'].values)

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

        self.resample()

    def resample(self, scale=5):
        def sample():
            rent_events = []

            for _ in range(scale):
                for period in self.periods:
                    for r in range(len(self.limits)):
                        ts = self._demands(period[0], r)
                        rent_events += [(t, r) for t in ts]

            return_events = []

            for t, r in rent_events:
                r1 = self._destination(t, r)
                t1 = t + self._duration(t, r, r1)
                return_events += [(t1, r1)]

            return_map = {tuple(k): tuple(v)
                          for k, v in zip(rent_events,
                                          return_events)}

            return rent_events, return_events, return_map

        if not self.real:  # the real data is sampled
            self.real_rent_events, self.real_return_events, self.real_return_map = sample()

        self.estimated_rent_events, self.estimated_return_events, self.estimated_return_map = sample()

    def __init__(self, episode, community, mu, tr, er, real):
        self.mu = mu
        self.tr = tr
        self.er = er
        self.periods = get_periods(episode)
        self.real = real

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
        dist = np.array(self.dist[r])
        dist[r] = np.inf
        r1 = np.argmin(dist)
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
