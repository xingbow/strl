import numpy as np
import heapq
import matplotlib.pyplot as plt


class Env(object):
    def __init__(self, simulator, capacity, num_trikes, rho):
        self.capacity = capacity
        self.num_trikes = num_trikes
        self.rho = rho
        self._init_renderer()

        self.load(simulator)
        self.reset()

# ------------------------ initialization ------------------------------------

    def load(self, simulator):
        """Load the static information of an episode
        """
        self.simulator = simulator
        self.num_regions = simulator.num_regions
        self.limits = simulator.limits
        self.delta = simulator.delta
        self.start_time = simulator.start_time

    def reset(self):
        # region
        self.loads = np.round(0.5 * self.limits)

        # loss
        self.losses = []

        # events
        self.events = []    # ordered
        self.history = []   # unordered, to visualize

        self.simulator.resample()

        for t, r in self.simulator.query_rents():
            self._push_rent_event(t, r, 1)

        for i in range(self.num_trikes):
            self._push_reposition_event(self.start_time, 0, 0, 0, 0)

        self.le, loss = self._run_until_next_reposition()
        return self._get_obs()

# ------------------------ event management  ------------------------------------

    def _push_event(self, event):
        i = len(self.history)
        self.history.append(event)
        heapq.heappush(self.events, (event['t'], i))

    def _pop_event(self):
        _, i = heapq.heappop(self.events)
        return self.history[i]

    def _push_rent_event(self, t, r, n):
        self._push_event({
            't': t,
            'r': r,
            'n': n,
            'tag': 'rent',
        })

    def _push_likely_return_event(self, t0, r0, n):
        t1, r1 = self.simulator.query_return(t0, r0)
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
        self.le, loss = self._run_until_next_reposition()

        self.losses.append(loss)
        return self._get_obs(), loss, self.done, None

    def _register_action(self, action):
        r1, n = self.decode_action(action)
        t0, r0, l = (self.le[k] for k in ['t', 'r', 'l'])
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
                reject = self._update_bikes(e['r'], -e['n'])
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

    @property
    def done(self):
        return self.tau >= self.simulator.end_time

    @property
    def observation_size(self):
        return self._get_obs().shape

    @property
    def action_size(self):
        return self.num_regions * (self.capacity * 2 + 1)

    @property
    def tau(self):
        return self.le['t']

    @property
    def state_size(self):
        return np.prod(self._get_obs().shape)

# ------------------------ observations  ------------------------------------

    def _get_obs(self):
        return np.concatenate([self.loads,
                               self.demands,
                               self.other_trikes_status,
                               self.current_trike_status])

    @property
    def demands(self):
        return self.simulator.estimate_demands([self.tau, self.tau + self.delta])

    @property
    def other_trikes_status(self):
        o = np.zeros(self.num_regions)
        for t, i in self.events:
            e = self.history[i]
            if e['tag'] == 'reposition':
                o[e['r']] += e['n']
        return o

    @property
    def current_trike_status(self):
        o = np.zeros(self.num_regions)
        o[self.le['r']] = self.le['l']  # at region r with load l
        return o

# ------------------------ action encoding decoding ------------------------

    def decode_action(self, action):
        action = int(action)
        r = action % self.num_regions
        n = action // self.num_regions - self.capacity
        return r, n

    def encode_action(self, r, n):
        action = r + (n + self.capacity) * self.num_regions
        return int(action)

    def featurize_action(self, action):
        r, n = self.decode_action(action)
        s = np.eye(self.num_regions)[r]
        b = [n]
        return np.concatenate([s, b])

#  ------------------------ render ------------------------

    def _init_renderer(self):
        plt.show(block=False)

    def _trike_position(self, e):
        loc = self.simulator.locations

        p0 = loc[e['r0']]
        p1 = loc[e['r']]
        t0 = e['t0']
        t1 = e['t']

        p = (p1 - p0) * ((self.tau - t0) / (t1 - t0)) + p0
        return p0, p1, p

    def _render_trike(self, e):
        p0, p1, p = self._trike_position(e)
        plt.plot(*zip(p0, p1), 'b--', lw=0.5)
        plt.annotate('{}/{}'.format(e['l'], e['n']), p, color='blue')
        plt.scatter(*p, c='r' if e['n'] > 0 else 'g', s=e['l']*2)

    def render(self):
        plt.clf()
        # plt.bar(range(len(self.loads)), self.loads)

        loc = self.simulator.locations
        x, y = loc[:, 0], loc[:, 1]

        for i in range(self.num_regions):
            plt.annotate('{:.0f}'.format(self.loads[i]),
                         loc[i])

        for t, i in self.events:
            e = self.history[i]
            if e['tag'] == 'reposition':
                self._render_trike(e)
        self._render_trike(self.le)

        plt.scatter(x=x, y=y, s=self.loads * 3)
        plt.pause(1e-2)

    def replay(self):
        pass
