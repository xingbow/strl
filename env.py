import numpy as np
import heapq
import time
from datetime import datetime
import matplotlib.pyplot as plt

from simulator import Simulator


def timestamp_to_timestr(ts):
    if ts < 3600 * 24:
        ts += datetime.today().timestamp()
    return datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')


class Event(object):
    def __init__(self, t, r, a):
        self.t = t  # tau
        self.r = r  # region
        self.a = a  # amount

    def __lt__(self, other):
        return self.t < other.t

    def __str__(self):
        return self.__class__.__name__ + ' ' + str(vars(self))


class RentEvent(Event):
    def __init__(self, t, r, a):
        super().__init__(t, r, a)


class ReturnEvent(Event):
    def __init__(self, t, r, a):
        super().__init__(t, r, a)


class GetEvent(Event):
    def __init__(self, t, r, a, r_put):
        super().__init__(t, r, a)
        self.r_put = r_put


class PutEvent(Event):
    def __init__(self, t, r, a):
        super().__init__(t, r, a)


class Env(object):
    def __init__(self, simulator,
                 episode, num_regions,
                 num_trikes, capacity):

        # register varaibles
        self._simulator = simulator
        self._episode = episode
        self._num_trikes = num_trikes
        self._num_regions = num_regions
        self._capacity = capacity

        # init render
        plt.show(block=False)

        self.reset()

    def step(self, action):
        """
        Args:
            action: [region, #bikes]
        Returns:
            next_state: the state observed
            reward: the reward for this action
        """
        if self.done:
            raise Exception("Env has been done.")

        if len(action) != 3:
            raise Exception("Error: Unknown action")

        e = self._pop_event()
        r_get, r_put, a = action
        self._register_get_event(t_cur=e.t, r_cur=e.r,
                                 r_get=r_get, r_put=r_put, a=a)
        self._process_to_next_reposition_event()

        return self.observation, self._reward

    def reset(self):
        self._loads = np.random.randint(
            10, 20, self._num_regions)    # region loads
        self._limits = np.random.randint(
            20, 40, self._num_regions)     # region capacity

        t = self._episode[0]

        rent_events = [RentEvent(t, r, 1)
                       for t, r in self._simulator.get_rent_events(t)]

        put_events = [PutEvent(t, np.random.randint(0, self._num_regions), 0)
                      for _ in range(self._num_trikes)]

        self._events = rent_events + put_events
        heapq.heapify(self._events)

        self._process_to_next_reposition_event()

    def _increase(self, r, a):
        assert a >= 0
        self._loads[r] += a
        miss = 0
        if self._loads[r] > self._limits[r]:
            miss = self._loads[r] - self._limits[r]
            self._loads[r] = self._limits[r]
        return miss

    def _decrease(self, r, a):
        assert a >= 0
        self._loads[r] -= a
        miss = 0
        if self._loads[r] < 0:
            miss = 0 - self._loads[r]
            self._loads[r] = 0
        return miss

    def _register_return_event(self, e: RentEvent, nearest=False):
        if nearest:
            r = self._simulator.get_nearest_region(e.r)
        else:
            r = self._simulator.get_likely_region(e.t, e.r)
        t = self._simulator.get_bike_arrival_time(e.t, e.r, r)
        self._push_event(ReturnEvent(t=t, r=r, a=e.a))

    def _register_put_event(self, e: Event, nearest=False):
        if nearest:
            r = self._simulator.get_nearest_region(e.r)
        else:
            r = e.r_put
        t = self._simulator.get_trike_arrival_time(e.t, e.r, r)
        self._push_event(PutEvent(t=t, r=r, a=e.a))

    def _register_get_event(self, t_cur, r_cur, r_get, r_put, a):
        t_get = self._simulator.get_trike_arrival_time(t_cur, r_cur, r_get)
        self._push_event(GetEvent(t=t_get, r=r_get, a=a, r_put=r_put))

    def _push_event(self, event):
        heapq.heappush(self._events, event)

    def _pop_event(self):
        return heapq.heappop(self._events)

    def _process_to_next_reposition_event(self):
        """
        Returns:
            event: the latest repo-return event
            reward: reward during this process
        """
        reward = 0

        while True:
            e = self._pop_event()
            print(e)

            if isinstance(e, RentEvent):    # bike rent
                miss = self._decrease(e.r, e.a)
                reward -= miss
                if miss == 0:   # success rent, register return
                    self._register_return_event(e)

            if isinstance(e, ReturnEvent):  # bike return
                miss = self._increase(e.r, e.a)
                reward -= miss
                if miss != 0:  # return failed, register next return
                    e.a = miss
                    self._register_return_event(e, True)

            if isinstance(e, GetEvent):  # trike get bikes
                miss = self._decrease(e.r, e.a)
                e.a -= miss  # real load
                self._register_put_event(e, False)

            if isinstance(e, PutEvent):  # trike put bikes
                miss = self._increase(e.r, e.a)
                reward -= miss
                if miss != 0:  # return failed, register next return
                    e.a -= miss
                    self._register_put_event(e, True)
                else:
                    self._push_event(e)  # put it back
                    break  # mission done for this reposition

        self._reward = reward

    def render(self):
        plt.clf()
        plt.bar(range(len(self._loads)), self._loads)
        plt.pause(1e-3)

    @property
    def observation(self):
        o = list(self._loads)
        put_events = [e for e in self._events if isinstance(e, PutEvent)]
        for e in put_events:
            o[e.r] += e.a
        return o

    @property
    def episode(self):
        """
        Current episode
        """
        return self._episode

    @property
    def done(self):
        return self._events[0].t > self._episode[1]

    @property
    def num_running_trike(self):
        return len([e for e in self._events if isinstance(e, PutEvent)])


def main():
    num_regions = 10
    num_trikes = 5
    episode = [0, 3600 * 1]
    capacity = 5

    env = Env(simulator=Simulator(num_regions),
              episode=episode,
              num_regions=num_regions,
              num_trikes=num_trikes,
              capacity=capacity)

    def random_action(state):
        return [np.random.randint(num_regions), np.random.randint(num_regions), np.random.randint(1, capacity)]

    state = np.zeros(num_regions)
    while not env.done:
        env.render()
        state, reward = env.step(random_action(state))
        print(reward)


if __name__ == "__main__":
    main()
