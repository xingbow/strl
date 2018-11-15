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
    def __init__(self, t, r, a, b, i):
        super().__init__(t, r, a)
        self.b = b  # loaded bikes
        self.i = i


class Env(object):
    def __init__(self, simulator,
                 num_trikes, capacity,
                 rho):
        # register variables
        self.simulator = simulator
        self.num_trikes = num_trikes
        self.capacity = capacity
        self.delta = simulator.delta
        self.rho = rho
        self.limits = self.simulator.get_limits()
        # init render
        plt.show(block=False)

        self.reset()

    def pruning(self):
        # pruning
        action = None

        if max(np.min(self.loads + self.predicted_loads + self.trike_status), 0) <= self.rho:  # deficient
            if self.cre.b / self.capacity > 0.5:
                r = np.argmin(self.loads + self.predicted_loads)
                a = self.cre.b
                action = self.encode_action(r, a)
            else:
                r = np.argmax(self.loads + self.predicted_loads)
                a = -(self.capacity - self.cre.b)
                action = self.encode_action(r, a)

        if action is not None:
            pass
            print('pruning')

        return action

    def step(self, action):
        """
        Args:
            action: an integer in [0, #regions * (2 * capacity + 1)]
        Returns:
            next_state: the state observed
            reward: the reward for this action
        """
        if self.done:
            raise Exception("Env is done.")

        if action > self.action_size or action < 0:
            raise Exception("Unknown action")

        r, a = self.decode_action(action)

        self.trike_loss += [[self.cre.i, self.loss]]

        self._register_reposition_event(self.cre, r, a)
        reward = self._process_to_next_reposition_event()

        return self._get_obs(), reward

    @property
    def rewards(self):
        for i in range(len(self.trike_loss)):
            for j in range(i + 1, len(self.trike_loss)):
                if self.trike_loss[i][0] == self.trike_loss[j][0]:
                    self.trike_loss[i][1] = self.trike_loss[j][1] - self.trike_loss[i][1]
                    break
        ret = [-l for _, l in self.trike_loss]
        assert all(np.array(ret) <= 0)
        return ret

    def reset(self):
        self.cre = None  # current reposition event
        self.loads = np.round(0.5 * self.limits)
        self.loss = 0
        self.simulator.resample()
        self.trike_loss = []

        tau = self.simulator.start_time

        rent_events = [RentEvent(t, r, 2)
                       for t, r in self.simulator.query_rent_events()]

        reposition_events = [RepositionEvent(tau,
                                             0,
                                             0,
                                             self.capacity,
                                             i)
                             for i in range(self.num_trikes)]

        self.events = rent_events + reposition_events
        heapq.heapify(self.events)

        self._process_to_next_reposition_event()
        return self._get_obs()

    def _increase(self, r, a):
        assert 0 <= a
        assert 0 <= r <= self.num_regions

        self.loads[r] += a
        miss = 0
        if self.loads[r] > self.limits[r]:
            miss = self.loads[r] - self.limits[r]
            self.loads[r] = self.limits[r]
        return miss

    def _decrease(self, r, a):
        assert 0 <= a
        assert 0 <= r <= self.num_regions
        self.loads[r] -= a
        miss = 0
        if self.loads[r] < 0:
            miss = 0 - self.loads[r]
            self.loads[r] = 0
        return miss

    def _register_likely_return_event(self, e: RentEvent):
        t, r = self.simulator.query_return_event(e.t, e.r)
        self._push_event(ReturnEvent(t=t, r=r, a=e.a))

    def _register_nearest_return_event(self, e: ReturnEvent):
        r = self.simulator.query_nearest_region(e.r)
        t = self.simulator.estimate_bike_arrival_time(e.t, e.r, r)
        self._push_event(ReturnEvent(t=t, r=r, a=e.a))

    def _register_reposition_event(self, e: RepositionEvent, r, a):
        e.t = self.simulator.estimate_trike_arrival_time(e.t, e.r, r)
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
                    self._register_likely_return_event(e)

            if isinstance(e, ReturnEvent):  # bike return
                miss = self._increase(e.r, e.a)
                # reward -= miss
                if miss != 0:  # return failed, register nearest return
                    e.a = miss
                    self._register_nearest_return_event(e)

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
        # plt.bar(range(len(self.loads)), self.loads)
        loc = self.simulator.locations
        x, y = loc[:, 0], loc[:, 1]
        future_change = -self.future_rent_demands + self.future_return_demands
        plt.scatter(x=x, y=y, s=self.loads * 3)

        for i in range(self.num_regions):
            plt.annotate('{:.0f}'.format(self.loads[i]),
                         loc[i])

        plt.pause(1e-3)

    def _get_obs(self):
        b1 = np.array(self.loads)
        b2 = self.future_rent_demands
        d2 = self.future_return_demands
        p = self.trike_status

        a = b1 - b2 + d2 + p
        sb = self.current_trike_status

        return np.concatenate([a, sb])

    @property
    def observation2(self):
        # n + n + n + 1 + n
        return np.concatenate([self.loads,
                               self.trike_status,
                               self.predicted_loads,
                               self.current_trike_status2])

    @property
    def predicted_loads(self):
        a = self.future_rent_demands
        b = self.future_return_demands
        return b - a

    @property
    def future_rent_demands(self):
        o = np.zeros(self.num_regions)
        es = self.simulator.estimate_rent_events(
            [self.tau, self.tau+self.delta])
        for t, r in es:
            o[r] += 1
        return o

    @property
    def future_return_demands(self):
        o = np.zeros(self.num_regions)
        es = self.simulator.estimate_return_events(
            [self.tau, self.tau+self.delta])
        for t, r in es:
            o[r] += 1
        return o

    @property
    def reposition_events(self):
        ret = []
        for e in self.events:
            if isinstance(e, RepositionEvent):
                ret.append(e)
        return ret

    @property
    def trike_status(self):
        o = np.zeros(self.num_regions)
        for e in self.events:
            if isinstance(e, RepositionEvent):
                o[e.r] += e.a
        return o

    @property
    def current_trike_status(self):
        o = np.zeros(self.num_regions)
        o[self.cre.r] = 1
        o = np.concatenate([o, [self.cre.a]])
        return o

    @property
    def current_trike_status2(self):
        o = np.zeros(self.num_regions)
        o[self.cre.r] = self.cre.b
        return o

    @property
    def tau(self):
        return self.cre.t

    @property
    def done(self):
        return self.tau >= self.simulator.end_time

    def decode_action(self, action):
        r = int(action % self.num_regions)
        a = int(action // self.num_regions - self.capacity)
        return r, a

    def encode_action(self, r, a):
        action = r + (a + self.capacity) * self.num_regions
        return int(action)

    def featurize_action(self, action):
        r, a = self.decode_action(action)
        s = np.eye(self.num_regions)[r]
        b = [a]
        return np.concatenate([s, b])

    @property
    def observation_size(self):
        return self._get_obs().shape

    @property
    def action_size(self):
        return self.num_regions * (self.capacity * 2 + 1)

    @property
    def num_regions(self):
        return len(self.limits)

# def main():
#     num_regions = 10
#     num_trikes = 2
#     episode = [0, 3600 * 1]
#     capacity = 5

#     env = Env(simulator=Simulator(num_regions),
#               episode=episode,
#               num_regions=num_regions,
#               num_trikes=num_trikes,
#               capacity=capacity,
#               delta=600)

#     def random_action(state):
#         return [np.argmax(np.random.uniform(size=[num_regions])),
#                 np.random.randint(-capacity, capacity)]

#     state = np.zeros(num_regions)

#     while not env.done:
#         env.render()
#         action = random_action(state)
#         state, reward = env.step(action)
#         print(action, reward)


# if __name__ == "__main__":
#     main()
