import numpy as np
import pandas as pd
import time
import OIModel as om

from collections import defaultdict


def timing_wrapper(func):
    def wrapped(*args, **kwargs):
        # begin_time = time.time()
        ret = func(*args, **kwargs)
        # print(func.__name__, 'takes', time.time() - begin_time)
        return ret
    return wrapped


def initialization():
    # type in file loc
    transitionDataLoc = '../data/dataOIModel.csv'  # transition file
    stationStatusDataLoc = '../data/stationStatus.csv'  # station status file
    weatherDataLoc = '../data/weather.csv'  # weather info file

    # Initialize function, no need to modify
    hisInputData, transitionMatrixDuration, transitionMatrixDestination, weatherMatrix = om.read_inData(
        transitionDataLoc, stationStatusDataLoc, weatherDataLoc)

    df = pd.read_csv('../data/station.csv')
    df['limit'] = df['limit'].fillna(df['limit'].mean())
    df = df.drop_duplicates('id')
    df = df.sort_values('id')
    df = df.head(30)
    ids = df['id'].values
    id_to_ix = {v: i for i, v in enumerate(ids)}

    points = df[['x', 'y']].values

    def distance_between(i, j):
        a = points[i]
        b = points[j]
        a, b = map(np.array, [a, b])
        return np.sum((a - b)**2)**0.5

    dist = defaultdict(dict)
    for id1 in ids:
        for id2 in ids:
            i = id_to_ix[id1]
            j = id_to_ix[id2]
            dist[i][j] = distance_between(i, j)
            dist[j][i] = dist[i][j]

    def distance(i, j):
        assert 0 <= i < len(ids) and 0 <= j < len(ids)
        return dist[i][j]

    def nearest(i):
        assert 0 <= i < len(ids)
        return max(dist[i].items(), key=lambda kv: kv[1])[0]

    # Inputs time stamp : e.g. 1373964540
    # Inputs station ID : e.g.212
    # return: expected departure(rent) number of this station in a period(hour)
    def demands(t, i):
        assert 0 <= i < len(ids)
        try:
            return om.get_expectDepartureNumber(
                t, ids[i],
                hisInputData,
                weatherMatrix)
        except:
            return 0

    # Inputs time stamp : e.g. 1373964540
    # Inputs station ID : e.g.212
    # return: expected destination and departure time

    def destination(t, i):
        assert 0 <= i < len(ids)
        try:
            return id_to_ix[om.get_predictedDestination(
                t, ids[i],
                transitionMatrixDuration,
                transitionMatrixDestination)[0]]
        except:
            return 0

    # Inputs time stamp : e.g. 1373964540
    # Inputs start station ID : e.g.212
    # Inputs end station ID : e.g.404
    # return: expected time used
    def duration(t, i, j):
        assert 0 <= i < len(ids) and 0 <= j < len(ids)
        try:
            return om.get_predictedDuration(
                t, ids[i], ids[j],
                transitionMatrixDuration,
                transitionMatrixDestination)
        except:
            return 0

    def query_limits():
        return df['limit'].values

    return map(timing_wrapper, [demands, destination, duration, distance, nearest, query_limits])


demands, destination, duration, \
    distance, nearest, query_limits = initialization()


class Simulator(object):
    def __init__(self, mu, tr, er):
        self.mu = mu
        self.tr = tr
        self.er = er
        self.limits = query_limits()
        print('#regions: {}'.format(len(self.limits)))

    def get_rent_events(self, episode):
        """
        Args:
            episode: [a, b] the episode required
        Returns:
            [(t, r)]: t is the rent timestamp (continuous), r is the region id
        """
        ret = []
        for t in range(episode[0], episode[1], 3600):
            for i in range(len(self.limits)):
                n = demands(t, i)
                ret += [(t + np.random.uniform(0, 3600), i) for _ in range(n)]
        return ret

    def get_likely_destination(self, t, r):
        """
        Needed when generate return event
        """
        return destination(t, r)

    def get_bike_arrival_time(self, t, r0, r1):
        """
        Needed when generate return event
        """
        return t + duration(t, r0, r1)

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
        return nearest(r)

    def get_distance(self, r0, r1):
        """
        Needed when estimate time for trike reposition
        """
        return distance(r0, r1)

    def get_limits(self):
        return self.limits
