import numpy as np
import heapq
from simulator import Simulator


class Env(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.load(None)
        self.reset()

# ------------------------ initialization ------------------------------------

    def load(self, simulator):
        """Load the static information of an episode
        """
        self.simulator = Simulator(0, 0, 0, 0, 0, 0)
        self.num_regions = simulator.num_regions
        self.limits = simulator.limits
        self.init_loads = simulator.init_loads

    def reset(self):
        # static

        # region state
        self.loads = []
        self.demands = []

        # loss
        self.losses = []

        # events
        self.events = []    # ordered
        self.history = []   # unordered, to visualize

        # time
        self.tau = self.simulator.start_time


# ------------------------ event management  ------------------------------------


    def _push_event(self, event):
        self.history.append(event)
        heapq.heappush(self.events, (event['t'], event))

    def _pop_event(self):
        _, event = heapq.heappop(self.events)
        return event

    def _push_likely_return_event(self, t0, r0, n):
        t1, r1 = self.simulator.query_return_event(t0, r0)
        self._push_event({
            't': t1,
            'r': r1,
            'n': n,
            'tag': 'return',
        })

    def _push_nearest_return_event(self, t0, r0, n):
        r1 = self.simulator.query_nearest_region(r0)
        t1 = self.simulator.estimate_bike_arrival_time(t0, r0, r1)
        self._push_event({
            't': t1,
            'r': r1,
            'n': n,
            'tag': 'return'
        })

    def _push_reposition_event(self, t0, r0, r1, n, l):
        if n + l > self.capacity:   # attempt to load too much
            n = self.capacity - l   # load just to full
        if n + l < 0:               # attempt to unload too much
            n = -l                  # just unload all

        t1 = self.simulator.estimate_trike_arrival_time(t0, r0, r1)
        self._push_event({
            't': t1,
            'r': r1,
            't0': t0,
            'r0': r0,
            'n': n,  # delta bikes
            'l': l,  # loads
            'tag': 'reposition'
        })

# ------------------------ state update ------------------------------------

    def step(self, action):
        # register reposition action
        self._register_action(action)

        # run reposition action
        self.lre, loss = self._run_until_next_reposition()
        self.losses.append(loss)

        return self._get_obs(), loss, self.done, None

    def _register_action(self, action):
        r1, n = self.decode_action(action)
        t0, r0, l = (self.lre[k] for k in ['t', 'r', 'l'])
        self._push_reposition_event(t0, r0, r1, n, l)

    def _update_bikes(self, r, n):
        self.loads[r] += n
        reject = 0
        if self.loads[r] < 0:
            reject = self.loads[r]
            self.loads[r] = 0
        if self.loads[r] > self.limits[r]:
            reject = self.loads[r] - self.limits[r]
            self.loads[r] = self.limits[r]
        return reject

    def _run_until_next_reposition(self):
        loss = 0
        while True:
            e = self._pop_event()

            if e['tag'] == 'rent':
                reject = self._update_bikes(e['r'], e['n'])
                if reject == 0:   # success rent, register return
                    self._push_likely_return_event(e['t'], e['r'], e['n'])
                else:
                    assert reject < 0
                    loss += abs(reject)

            if e['tag'] == 'return':
                reject = self._update_bikes(e['r'], e['n'])
                if reject != 0:  # return failed, register nearest return
                    assert reject > 0
                    self._push_nearest_return_event(e['t'], e['r'], reject)

            if e['tag'] == 'reposition':
                reject = self._update_bikes(e['r'], -e['n'])
                e['l'] = (e['l'] + e['n'] + reject)
                break

        return e, loss

#  ------------------------ properties  ------------------------------------

    def _get_obs(self):
        pass

    @property
    def done(self):
        return self.tau >= self.simulator.end_time

    @property
    def tau(self):
        return self.lre['t']

# ------------------------ action encoding decoding ------------------------

    def decode_action(self, action):
        r = action % self.num_regions
        n = action // self.num_regions - self.capacity
        return r, n

    def encode_action(self, r, n):
        action = r + (n + self.capacity) * self.num_regions
        return action

    def featurize_action(self, action):
        r, n = self.decode_action(action)
        s = np.eye(self.num_regions)[r]
        b = [n]
        return np.concatenate([s, b])
