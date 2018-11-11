import numpy as np
import heapq
import time

from simulator import Simulator, Data


class Environment(object):
    def __init__(self, simulator,
                 episode, num_regions, num_trikes,
                 trike_capacity, region_capacity):

        self._simulator = simulator
        self._episode = episode
        self._num_trikes = num_trikes
        self._num_regions = num_regions
        self._trike_capacity = trike_capacity
        self._region_capacity = region_capacity

        self._bikes = []
        self._events = []
        self._latest_repo_event = None
        self.reset()

    def step(self, action):
        """
        Args:
            action: a vector with length #region + trike_capacity 
        Returns:
            next_state: the state observed
            reward: the reward for this action
            done: whether the episode finished
        """
        assert len(action) == self._num_regions + self._trike_capacity

        if self.done:
            raise Exception("Environment has been done.")

        self._reposition(action)
        reward = self._process_to_next_reposition_event()

        return self.observation, reward

    def reset(self):
        self._latest_repo_event = None
        self._bikes = np.random.randint(10, 20, self._num_regions)

        tau = self.episode[0]
        rent_events = [e + (-1, 'rent')
                       for e in self._simulator.simulate_rent_events(tau)]

        repo_events = [(tau,
                        np.random.randint(0, self._num_regions),
                        0,
                        'repo') for _ in range(self._num_trikes)]

        self._events = rent_events + repo_events
        heapq.heapify(self._events)

        self._process_to_next_reposition_event()

    def _reposition(self, action):
        tau0, r0, _, _ = self._latest_repo_event
        r1 = np.argmax(action[self._num_regions:])  # destination
        loads = np.argmax(action[self._num_regions:])
        tau1, r1 = self._simulator.simulate_reposition_event(tau0, r0, r1)
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
            if self._bikes[r] > self._region_capacity:
                overload = self._bikes[r] - self._region_capacity
                self._bikes[r] = self._region_capacity
                # return to another station
                tau1, r1 = self._simulator.simulate_return_event(tau, r)
                heapq.heappush(self._events, (tau1, r1, overload, tag))
                reward -= overload
            if self._bikes[r] < 0:  # run out of bikes, penalty
                reward += self._bikes[r]
                self._bikes[r] = 0
            if tag == 'rent':       # register return for the rent
                tau1, r1 = self._simulator.simulate_return_event(tau, r)
                heapq.heappush(self._events, (tau1, r1, 1, 'return'))
            if tag == "repo":  # find a reposition, break
                break

        self._latest_repo_event = event

        return reward

    @property
    def observation(self):
        o = list(self._bikes)
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

    @property
    def done(self):
        return self.timestamp > self.episode[1]


def main():
    num_regions = 20
    num_trikes = 5
    episode = [0, 3600 * 1]
    trike_capacity = 5
    region_capacity = 50

    env = Environment(simulator=Simulator(Data(num_regions)),
                      episode=episode,
                      num_regions=num_regions,
                      num_trikes=num_trikes,
                      trike_capacity=trike_capacity,
                      region_capacity=region_capacity)

    def random_action(state):
        return np.random.random(num_regions + trike_capacity)

    state = np.zeros(num_regions + trike_capacity)
    while not env.done:
        state, reward = env.step(random_action(state))
        print(env._bikes, reward)


if __name__ == "__main__":
    main()
