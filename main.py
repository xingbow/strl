from agent import DQNAgent, VanillaDQNAgent, VanillaPGAgent
from env import Env
from simulator import Simulator

import numpy as np


def run_baseline(env, config):
    num_epochs = config['num_epochs']

    losses = []
    for epoch in range(num_epochs):
        env.reset()
        while not env.done:
            action = env.pruning()
            if action is None:
                action = [0, 0]
            env.step(action)
        losses += [env.loss]
        print('{}/{} loss {}, smooth loss {}'.format(
            epoch, num_epochs,
            losses[-1], np.mean(losses[-10:])))


def run_vanilla_dqn(env, config):
    num_epochs = config['num_epochs']
    batch_size = config['batch_size']

    num_regions = env.num_regions
    capacity = env.capacity

    action_size = num_regions * (2 * capacity + 1)
    state_size = env._get_obs().shape[0]
    agent = VanillaDQNAgent(state_size, action_size)

    def encode_action(action):
        if action is None:
            return None
        r, a = action
        ret = r + (a + capacity) * num_regions
        return ret

    def decode_action(action):
        return [action % num_regions, action // num_regions - capacity]

    assert all([encode_action(decode_action(a)) == a for a in range(1000)])

    losses = []

    for epoch in range(num_epochs):
        state = env.reset()
        while not env.done:
            # if epoch % 10 == 0:
            #     env.render()

            action = encode_action(env.pruning())
            if action is None:
                action = agent.act(state)

            next_state, reward = env.step(decode_action(action))

            agent.remember(state, action, reward, next_state, env.done)
            state = next_state

        agent.replay(batch_size)
        losses += [env.loss]

        print('{}/{} loss {}, smooth loss {}'.format(
            epoch, num_epochs,
            losses[-1], np.mean(losses[-10:])))


def run_dqn(env, config):
    num_epochs = config['num_epochs']
    batch_size = config['batch_size']

    num_regions = env.num_regions
    capacity = env.capacity
    action_size = num_regions * (2 * capacity + 1)

    agent = DQNAgent(num_regions=num_regions,
                     action_size=action_size)

    def encode_action(action):
        if action is None:
            return None
        r, a = action
        ret = r + (a + capacity) * num_regions
        return ret

    def decode_action(action):
        return [action % num_regions, action // num_regions - capacity]

    def create_action_state(a):
        a = decode_action(a)
        s = np.eye(num_regions)[a[0]]
        b = [a[1]]
        return np.concatenate([s, b])

    def create_state(o0, a0, o1, a1):
        a0, a1 = map(lambda a: create_action_state(a), [a0, a1])
        return np.concatenate([o0, a0, o1, a1])

    assert all([encode_action(decode_action(a)) == a for a in range(1000)])

    losses = []

    for epoch in range(num_epochs):
        observations = [env.reset()] * 2
        actions = [0, 0]
        state = create_state(observations[-2],
                             actions[-2],
                             observations[-1],
                             actions[-1])

        while not env.done:
            # if epoch % 10 == 0:
            #     env.render()

            action = encode_action(env.pruning())
            if action is None:
                action = agent.act(state)

            observation, reward = env.step(decode_action(action))

            actions.append(action)
            observations.append(observation)

            next_state = create_state(observations[-2],
                                      actions[-2],
                                      observations[-1],
                                      actions[-1])

            agent.remember(state, action, reward, next_state, env.done)
            state = next_state

        agent.replay(batch_size)
        losses += [env.loss]

        print('{}/{} loss {}, smooth loss {}'.format(
            epoch, num_epochs,
            losses[-1], np.mean(losses[-10:])))


def run_vanilla_pg(env, config):
    num_epochs = config['num_epochs']

    num_regions = env.num_regions
    capacity = env.capacity

    action_size = num_regions * (2 * capacity + 1)
    state_size = env._get_obs().shape[0]
    agent = VanillaPGAgent(state_size, action_size)

    def encode_action(action):
        if action is None:
            return None
        r, a = action
        ret = r + (a + capacity) * num_regions
        return ret

    def decode_action(action):
        return [action % num_regions, action // num_regions - capacity]

    assert all([encode_action(decode_action(a)) == a for a in range(1000)])

    losses = []

    for epoch in range(num_epochs):
        S = []
        A = []
        R = []

        state = env.reset()

        while not env.done:
            action = encode_action(env.pruning())
            if action is None:
                action = agent.act(state)

            next_state, reward = env.step(decode_action(action))

            S.append(state)
            A.append(action)
            R.append(reward)

            state = next_state

        S = np.array(S)
        A = np.array(A)
        R = np.array(R)

        agent.replay(S, A, R)
        losses += [env.loss]

        print('{}/{} loss {}, smooth loss {}'.format(
            epoch, num_epochs,
            losses[-1], np.mean(losses[-10:])))


def main():
    num_regions = 10
    num_trikes = 5
    episode = [0, 3600 * 1]
    capacity = 5
    num_epochs = 40
    batch_size = 32
    delta = 600
    rho = 4

    config = {
        "num_epochs": num_epochs,
        "batch_size": batch_size,
    }

    env = Env(simulator=Simulator(num_regions),
              episode=episode,
              num_regions=num_regions,
              num_trikes=num_trikes,
              capacity=capacity,
              delta=delta,
              rho=rho)

    # run_baseline(env, config)

    # run_vanilla_dqn(env, config)
    # run_dqn(env, config)

    run_vanilla_pg(env, config)


if __name__ == "__main__":
    main()
