import numpy as np


class Simulator(object):
    def __init__(self, num_regions, real_mode=False):
        """
        Args:
            real_mode: real world case or not, if real, simulate all, otherwise, query all
        """
        self._num_regions = num_regions
        self._distance = 1000 * \
            np.abs(np.random.normal(
                size=[self._num_regions, self._num_regions]))

    def get_rent_events(self, episode):
        """
        Args:
            episode: [a, b] the episode required
        Returns:
            [(t, r)]: t is the rent timestamp (continuous), r is the region id
        """
        events = []
        for i in range(1000):
            r = np.random.randint(0, self._num_regions)
            t = np.random.random() * 1000
            events.append((t, r))
        return events

    def get_likely_region(self, t, r):
        """
        Needed when generate return event
        """
        return np.random.randint(0, self._num_regions)

    def get_bike_arrival_time(self, t, ra, rb):
        """
        Needed when generate return event
        """
        return t + self.get_distance(ra, rb) / 10

    def get_trike_arrival_time(self, t, ra, rb):
        """
        Needed when generate reposition
        """
        return t + self.get_distance(ra, rb) / 50

    def get_nearest_region(self, r):
        """
        Needed when handle the case when the station is full
        Args:
            r: region id
        Returns:
            nearest region
        """
        return np.argmax(self._distance[r])

    def get_distance(self, ra, rb):
        """
        Needed when estimate time for trike reposition
        """
        return self._distance[ra, rb]
