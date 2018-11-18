"""
Interfaces with other guys
"""
import numpy as np
import pandas as pd
import time
import contrib.OIModel as om
from datetime import datetime
from collections import defaultdict
import pickle
import os

DURATIONS = [
    [[7, 11]],
    [[11, 12], [16, 17]],
    [[12, 16]],
    [[17, 18]],
    [[18, 23]],
]


def date_to_timestamp(date):
    return int(datetime.strptime(date, '%Y/%m/%d').timestamp())


def get_episode(date, ep_i):
    base = date_to_timestamp(date)

    d = DURATIONS[ep_i][0]
    episode = [base + d[0]*3600,
               base + d[1]*3600]

    return episode


def get_periods(date, ep_i):
    delta = 3600
    periods = []
    episode = get_episode(date, ep_i)
    t = episode[0]
    while t < episode[1]:
        periods.append([t, t + delta])
        t += delta
    return periods


def zh_interfaces(date):
    # type in file loc
    transitionDataLoc = '../data/test.csv'  # transition file
    weatherDataLoc = '../data/weather.csv'  # weather info file

    startTime = 1375286400  # xingbo magic number (timestamp base)
    timeStampBound = date_to_timestamp(date)

    dayNumBound = (timeStampBound - startTime) // (24 * 3600)

    try:
        with open('../data/zh_tmp.pkl', 'rb') as f:
            hisInputData, transitionMatrixDuration,\
                transitionMatrixDestination, weatherMatrix, \
                overAllTest, pred_y, dayNumBound = pickle.load(f)
    except:
        with open('../data/zh_tmp.pkl', 'wb') as f:
            hisInputData, transitionMatrixDuration, transitionMatrixDestination,\
                weatherMatrix, entireSituation, weatherMatrixAll\
                = om.read_inData(transitionDataLoc, weatherDataLoc)
            overAllTest, pred_y, dayNumBound = om.predicted_overallSituation(
                dayNumBound, entireSituation, weatherMatrixAll)
            pickle.dump([hisInputData, transitionMatrixDuration,
                         transitionMatrixDestination, weatherMatrix,
                         overAllTest, pred_y, dayNumBound], f)

    def demands(t, r):
        t = int(t)
        return om.get_expectDepartureNumber(
            t, r, hisInputData,
            weatherMatrix, overAllTest, pred_y, dayNumBound)

    def destination(t, r):
        t = int(t)
        return int(om.get_predictedDestination(
            t, r,
            transitionMatrixDuration,
            transitionMatrixDestination))

    def duration(t, r0, r1):
        t = int(t)
        try:
            return om.get_predictedDuration(
                t, r0, r1,
                transitionMatrixDuration,
                transitionMatrixDestination)
        except:
            print('Warning: duration yields an exception.')
            return 0

    return demands, destination, duration


def extract_region(df):
    station_df = df[['start station id',
                     'start id',
                     'rx1',
                     'ry1']].drop_duplicates()

    station_df.columns = ['sid', 'rid', 'x', 'y']

    limit_df = pd.read_csv('../data/stationStatus.csv')
    limit_df.columns = ['sid', 'name', 'limit']
    del limit_df['name']
    limit_df['limit'] = limit_df['limit'].fillna(0)

    limit_df = pd.merge(station_df, limit_df, how='left', on='sid')

    limit_df = limit_df.groupby('rid')[['limit']].sum().reset_index(level=0)
    region_df = pd.merge(station_df, limit_df, how='left', on='rid')

    del region_df['sid']

    region_df = region_df.drop_duplicates()
    region_df = region_df.sort_values('rid')

    # assert there won't be region at different places
    assert len(region_df['rid'].unique()) == len(region_df['rid'])

    return region_df


def xb_interfaces(date, episode, community):
    transitionDataLoc = '../data/test.csv'  # transition file
    startTime = 1375286400  # xingbo magic number (timestamp base)

    # get all of the trip data on specific episode & community
    # after this, relation ship between region and station should be fixed
    tdf = pd.read_csv(transitionDataLoc)
    tdf = tdf[(tdf['episode'] == episode) &
              (tdf['community'] == community)]

    tdf['start timestamp'] += startTime
    tdf['end timestamp'] = tdf['start timestamp'] + tdf['tripduration']

    # extract the region information
    rdf = extract_region(tdf)

    # today trip dataframe
    ep = get_episode(date, episode)
    tdf = tdf[(tdf['start timestamp'] >= ep[0])
              & (tdf['start timestamp'] < ep[1])]

    tdf = tdf[(tdf['end timestamp'] >= ep[0])
              & (tdf['end timestamp'] < ep[1])]

    # remove unknown end region
    tdf = tdf[tdf['end id'].isin(tdf['start id'])]

    return tdf, rdf


def combine_interfaces(date, episode, community):
    demands, destination, duration = zh_interfaces(date)
    tdf, rdf = xb_interfaces(date, episode, community)

    # region ids, which not starts from 0
    rids = rdf['rid'].values.astype(int)
    # region index, starts form 0
    rid_to_ix = {v: i for i, v in enumerate(rids)}

    tdf['start id'] = tdf['start id'].replace(rid_to_ix)
    tdf['end id'] = tdf['end id'].replace(rid_to_ix)

    real_rents = tdf[['start timestamp',
                      'start id']].values.astype(int)

    real_returns = tdf[['end timestamp',
                        'end id']].values.astype(int)

    real_return_map = {tuple(k): tuple(v)
                       for k, v in zip(real_rents,
                                       real_returns)}

    limits = np.round(rdf['limit'].values)

    locations = locations = rdf[['x', 'y']].values

    def distance_between(i, j):
        a = locations[i]
        b = locations[j]
        a, b = map(np.array, [a, b])
        return np.sum((a - b)**2)**0.5

    distance = np.zeros([len(rids), len(rids)])
    for i in range(len(rids)):
        for j in range(i + 1, len(rids)):
            distance[i, j] = distance_between(i, j)
            distance[j, i] = distance[i, j]

    def reindexed_demands(t, r):
        return demands(t, rids[r])

    def reindexed_destination(t, r):
        return rid_to_ix[destination(t, rids[r])]

    def reindexed_duration(t, r0, r1):
        return duration(t, rids[r0], rids[r1])

    return {
        "demands": reindexed_demands,
        "destination":  reindexed_destination,
        "duration": reindexed_duration,
        "distance": distance,
        "limits": limits,
        "rids": rids,
        "real_rents": real_rents,
        "real_returns": real_returns,
        "real_return_map": real_return_map,
        "locations": locations,
    }


class Simulator(object):
    def __init__(self, date, episode, community, mu, tr, er, real=True):
        self.I = combine_interfaces(date, episode, community)
        self.mu = mu
        self.tr = tr
        self.er = er
        self.periods = get_periods(date, episode)
        self.real = real
        self.limits = self.I['limits']
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

    def _sample_rent_return(self, scale=1):
        # sample rents
        rents = []
        for _ in range(scale):
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