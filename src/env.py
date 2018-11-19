import numpy as np
import heapq
import os
import json


def default(o):
    if isinstance(o, np.int64):
        return int(o)
    if isinstance(o, np.int32):
        return int(o)
    if isinstance(o, np.float32):
        return float(o)
    if isinstance(o, np.float64):
        return float(o)
    raise TypeError(type(o), o)


class Env(object):
    def __init__(self, simulator, capacity, num_trikes, rho):
        self.capacity = capacity
        self.num_trikes = num_trikes
        self.rho = rho
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
        self.end_time = simulator.end_time
        self.locations = self.simulator.locations

    def reset(self):
        # region
        self.loads = np.array(self.simulator.loads).astype(int)

        # loss
        self.losses = []

        # events
        self.events = []    # ordered
        self.history = []   # unordered, to visualize

        self.simulator.resample()  # resample every reset

        for t, r in self.simulator.query_rents():
            self._push_rent_event(t, r, 1)

        for i in range(self.num_trikes):
            self._push_reposition_event(self.start_time, 0, 0, 0, 0, i)

        self.le, _ = self._run_until_next_reposition()
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

    def _push_reposition_event(self, t0, r0, r1, n, l, i):
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
            'i': i,
            'tag': 'reposition'
        })

    def _push_snapshot_event(self, t):
        self._push_event({
            't': t,
            'tag': 'snapshot',
        })

# ------------------------ state update ------------------------------------

    def pruning(self):
        action = None

        if self.rho < 0:  # pruning off
            return action

        expected = self.expected_bikes

        if np.min(expected) <= self.rho:  # deficient
            if self.le['l'] / self.capacity > 0.5:
                # find the region with minimal expected bikes
                r = np.argmin(expected)
                # unload all
                n = -self.le['l']
                action = self.encode_action(r, n)
            else:
                # find the region with maximal expected bikes
                r = np.argmax(expected)
                # load to full
                n = (self.capacity - self.le['l'])
                action = self.encode_action(r, n)

        return action

    def step(self, action):
        # register reposition action
        self._register_action(action)

        # run reposition action
        self.le, loss = self._run_until_next_reposition()
        self.losses.append(loss)

        return self._get_obs(), -loss, self.done, None

    def _register_action(self, action):
        r1, n = self.decode_action(action)
        t0, r0, l = (self.le[k] for k in ['t', 'r', 'l'])
        self._push_reposition_event(t0, r0, r1, n, l, self.le['i'])

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
                e['n'] = 0
                break

            if e['tag'] == 'snapshot':
                self._snapshot(e['t'])

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

    @property
    def loss(self):
        return np.sum(self.losses)

# ------------------------ observations  ------------------------------------

    def _get_obs(self):
        max_limit = np.max(self.limits)
        return np.concatenate([self.loads / max_limit,
                               self.future_demands / max_limit,
                               self.other_trikes_status / max_limit,
                               self.current_trike_status / self.capacity])

        # return np.concatenate([self.expected_bikes / max_limit,
        #                        self.current_trike_status])

    @property
    def future_demands(self):
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

    @property
    def expected_bikes(self):
        return self.loads - self.future_demands - self.other_trikes_status


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

#  ------------------------ render ------------------------

    def render(self):
        """
        deprecated, use snapshot instead
        """
        pass

    def _trike_position(self, e, t):
        p0 = self.locations[e['r0']]
        p1 = self.locations[e['r']]
        t0 = e['t0']
        t1 = e['t']
        p = (p1 - p0) * ((t - t0) / (t1 - t0)) + p0
        return p0, p1, p
        
    def _snapshot(self, tau):
        frame = {
            "timestamp": tau,
            "regions": [],
            "trikes": []
        }

        # add trikes
        for _, i in self.events:
            e = self.history[i]
            if e['tag'] == 'reposition':
                p0, p1, p = self._trike_position(e, tau)
                trike = {
                    "x": p[0],
                    "y": p[1],
                    "bike_to_load": e['n'],
                    "bike_loaded": e['l'],
                    "src_x": p0[0],
                    "src_y": p0[1],
                    "dst_x": p1[0],
                    "dst_y": p1[1],
                    "src": e['r0'],
                    "dst": e['r'],
                    "src_t": e['t0'],
                    "dst_t": e['t'],
                    "id": e['i'],
                }
                frame["trikes"].append(trike)

        # add regions
        for i in range(self.num_regions):
            region = {
                "x": self.locations[i][0],
                "y": self.locations[i][1],
                "id": i,
                "bike": self.loads[i],
                "capacity": self.limits[i],
            }
            frame["regions"].append(region)

        self.snapshot_frames.append(frame)
        if len(self.snapshot_frames) == self.snapshot_num_frames:
            with open(self.snapshot_path, 'w') as f:
                json.dump(self.snapshot_frames, f, default=default)

    def register_snapshots(self, path, num_frames=200):
        ts = np.linspace(self.start_time, self.end_time, num_frames)
        self.snapshot_path = path
        self.snapshot_frames = []
        self.snapshot_num_frames = num_frames
        os.makedirs(os.path.dirname(path), exist_ok=True)
        for t in ts:
            self._push_snapshot_event(t)
