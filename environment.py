import numpy as np
import heapq

from simulator import Simulator


class Environment(object):
    def __init__(self, simulator, num_regions, num_trikes, episode, capacity):
        """
        Args:
            data: episode data, initialize state, otherwise random generate
            simulator: episode data, initialize state, otherwise random
            episode: [tau0, tau0 + delta0], tau0 is the episode begining time, delta0 is the episode length
            plen: period duration
            wlen: window duration
        """
        self._simulator = simulator
        self._episode = episode
        self._capacity = capacity
        self._num_regions = num_regions
        self._num_trikes = num_trikes

        self._bikes = []
        self._events = []
        self._latest_repo_event = None
        self.reset()

    def step(self, action):
        """
        Returns:
            next_state: the state observed
            reward: the reward for this action
            done: whether the episode finished
        """
        assert len(action) == self._num_regions + self._capacity

        if self.timestamp > self.episode[1]:
            return None, None, True

        self._reposition(action)
        reward = self._process_to_next_reposition_event()

        return self._observe(), reward, False

    def reset(self):
        tau = self.episode[0]
        self._bikes = np.random.randint(10, 50, self._num_regions)

        rent_events = [e + (-1, 'rent')
                       for e in self._simulator.simulate_rent_events(tau)]
        repo_events = [(tau,
                        np.random.randint(0, self._num_regions),
                        0,
                        'repo') for _ in range(self._num_trikes)]

        self._events = rent_events + repo_events
        heapq.heapify(self._events)

        self._latest_repo_event = None

    def _reposition(self, action):
        tau0, r0, _, _ = self._latest_repo_event
        r1 = np.argmax(action[self._num_regions:])  # destination
        loads = np.argmax(action[self._num_regions:])
        tau1 = self._simulator.simulate_reposition_event(tau0, r0, r1)
        new_event = (tau1, r1, loads, "repo")
        heapq.heappush(self._events, new_event)

    def _process_to_next_reposition_event(self):
        """
        Returns:
            event: the latest reposition event
            reward: reward during this process
        """
        reward = 0
        while True:
            event = heapq.heappop(self._events)
            tau, r, n, tag = event
            self._bikes[r] += n
            if self._bikes[r] < 0:  # run out of bikes, penalty
                reward += self._bikes[r]
                self._bikes[r] = 0
            if tag == 'rent':       # register return for the rent
                return_event = self._simulator.simulate_return_event(tau, r)
                heapq.heappush(self._events, return_event + (1, 'return'))
            if tag == "repo":  # find a reposition, break
                break
        self._latest_repo_event = event
        return reward

    @property
    def observation(self):
        o = self._bikes
        repo_events = [e for e in self._events if e[-1] == "repo"]
        for event in repo_events:
            _, r, n, _ = event
            o[r] += n
        return o

    @property
    def timestamp(self):
        return self._events[0][0]

    @property
    def episode(self):
        """
        Current episode
        """
        return self._episode


if __name__ == "__main__":
    pass
