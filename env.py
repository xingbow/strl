import numpy as np
import heapq
import time
from datetime import datetime
import json
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
        return self.__class__.__name__ + '\n' + str(vars(self))


class RentEvent(Event):
    def __init__(self, t, r, a):
        super().__init__(t, r, a)


class ReturnEvent(Event):
    def __init__(self, t, r, a):
        super().__init__(t, r, a)


class RepositionEvent(Event):
    def __init__(self, t, r, a, b):
        super().__init__(t, r, a)
        self.b = b  # loaded bikes


class Env(object):
    def __init__(self, simulator,
                 episode, num_regions,
                 num_trikes, capacity,
                 rho, delta):

        # register variables
        self.simulator = simulator
        self.episode = episode
        self.num_trikes = num_trikes
        self.num_regions = num_regions
        self.capacity = capacity
        self.delta = delta
        self.rho = rho

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
            raise Exception("Env is done.")

        if len(action) != 2:
            raise Exception("Error: Unknown action")

        # pruning
        if np.min(self.loads) <= self.rho:  # deficient
            r = np.argmin(self.loads)
            a = self.cre.b
            action = [r, a]
            # print('deficient at {}'.format(r))

        if np.min(self.limits - self.loads) <= self.rho:  # congested
            r = np.argmin(self.limits - self.loads)
            a = self.capacity - self.cre.b
            action = [r, a]
            # print('congested at {}'.format(r))

        r, a = action

        self._register_reposition_event(self.cre, r, a)
        reward = self._process_to_next_reposition_event()

        return self._get_obs(), action, reward

    def reset(self):
        self.cre = None  # current reposition event
        self.loads = np.random.randint(
            3, 10, self.num_regions)    # region loads
        self.limits = np.random.randint(
            20, 40, self.num_regions)     # region capacity
        self.loss = 0

        t = self.episode[0]

        rent_events = [RentEvent(t, r, 1)
                       for t, r in self.simulator.get_rent_events(t)]

        reposition_events = [RepositionEvent(t, np.random.randint(self.num_regions), 0, self.capacity)
                             for _ in range(self.num_trikes)]

        self.events = rent_events + reposition_events
        heapq.heapify(self.events)

        self._process_to_next_reposition_event()
        return self._get_obs()

    def _increase(self, r, a):
        assert a >= 0
        self.loads[r] += a
        miss = 0
        if self.loads[r] > self.limits[r]:
            miss = self.loads[r] - self.limits[r]
            self.loads[r] = self.limits[r]
        return miss

    def _decrease(self, r, a):
        assert a >= 0
        self.loads[r] -= a
        miss = 0
        if self.loads[r] < 0:
            miss = 0 - self.loads[r]
            self.loads[r] = 0
        return miss

    def _register_return_event(self, e: RentEvent, nearest=False):
        if nearest:
            r = self.simulator.get_nearest_region(e.r)
        else:
            r = self.simulator.get_likely_region(e.t, e.r)
        t = self.simulator.get_bike_arrival_time(e.t, e.r, r)
        self._push_event(ReturnEvent(t=t, r=r, a=e.a))

    def _register_reposition_event(self, e: RepositionEvent, r, a):
        e.t = self.simulator.get_trike_arrival_time(e.t, e.r, r)
        e.r = r
        e.a = a
        self._push_event(e)

    def _push_event(self, event):
        heapq.heappush(self.events, event)

    def _pop_event(self):
        return heapq.heappop(self.events)

    def _process_to_next_reposition_event(self):
        reward = 0

        while True:
            e = self._pop_event()

            if isinstance(e, RentEvent):    # bike rent
                miss = self._decrease(e.r, e.a)
                reward -= miss
                self.loss += miss
                if miss == 0:   # success rent, register return
                    self._register_return_event(e)

            if isinstance(e, ReturnEvent):  # bike return
                miss = self._increase(e.r, e.a)
                reward -= miss
                if miss != 0:  # return failed, register next return
                    e.a = miss
                    self._register_return_event(e, True)

            if isinstance(e, RepositionEvent):
                assert 0 <= e.b <= self.capacity
                if e.a > 0:  # unload
                    e.a = min(e.a, e.b)  # check capacity
                    miss = self._increase(e.r, e.a)
                    e.b -= (e.a - miss)
                elif e.a < 0:  # load
                    e.a = abs(e.a)
                    e.a = min(e.a, self.capacity - e.b)  # check capacity
                    miss = self._decrease(e.r, e.a)
                    e.b += (e.a - miss)
                self.cre = e
                break

        return reward

    def render(self):
        plt.clf()
        plt.bar(range(len(self.loads)), self.loads)
        plt.pause(1e-3)

    def _get_obs(self):
        # current state
        b1 = np.array(self.loads)

        maxt = self.tau + self.delta

        # demand state
        b2 = np.zeros_like(b1)
        for e in self.events:
            if isinstance(e, RentEvent) and e.t < maxt:
                b2[e.r] += e.a

        # demand state
        d2 = np.zeros_like(b1)
        for e in self.events:
            if isinstance(e, ReturnEvent) and e.t < maxt:
                d2[e.r] += e.a

        # bike state
        o1 = b1 - b2 + d2

        # other state
        o2 = np.zeros_like(o1)
        o2[self.cre.r] = 1

        # self state
        o3 = [self.cre.a]

        return np.concatenate([o1, o2, o3])

    @property
    def tau(self):
        return self.cre.t

    @property
    def done(self):
        return self.events[0].t > self.episode[1]


def main():
    num_regions = 10
    num_trikes = 2
    episode = [0, 3600 * 1]
    capacity = 5

    env = Env(simulator=Simulator(num_regions),
              episode=episode,
              num_regions=num_regions,
              num_trikes=num_trikes,
              capacity=capacity,
              delta=600)

    def random_action(state):
        return [np.argmax(np.random.uniform(size=[num_regions])),
                np.random.randint(-capacity, capacity)]

    state = np.zeros(num_regions)

    while not env.done:
        env.render()
        action = random_action(state)
        state, action, reward = env.step(action)
        print(action, reward)


if __name__ == "__main__":
    main()
