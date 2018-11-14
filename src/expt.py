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
            # env.render()
            action = env.pruning()
            if action is None:
                action = [0, 0]
            _, reward = env.step(action)
            print(reward)
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

    losses = []

    for epoch in range(num_epochs):
        state = env.reset()
        while not env.done:
            # if epoch % 10 == 0:
            #     env.render()

            action = env.pruning()
            if action is None:
                action = agent.act(state)

            next_state, reward = env.step(action)

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

    losses = []

    for epoch in range(num_epochs):
        observations = [env.reset()] * 2
        actions = [0, 0]

        def create_state():
            return np.concatenate([observations[-2],
                                   env.featurize_action(actions[-2]),
                                   observations[-1],
                                   env.featurize_action(actions[-1])])

        state = create_state()

        while not env.done:
            # if epoch % 10 == 0:
            #     env.render()

            action = env.pruning()
            if action is None:
                action = agent.act(state)

            observation, reward = env.step(action)

            actions.append(action)
            observations.append(observation)
            next_state = create_state()

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

    losses = []

    for epoch in range(num_epochs):
        S = []
        A = []
        R = []

        state = env.reset()

        while not env.done:
            action = env.pruning()
            if action is None:
                action = agent.act(state)

            next_state, reward = env.step(action)

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
    episode = [1373964540, 1373964540 + 3600 * 3]
    capacity = 10
    num_epochs = 20
    batch_size = 32
    delta = 600
    rho = 10
    mu = 200 / 60
    tr = 60 * 3
    er = 3 * 60

    config = {
        "num_epochs": num_epochs,
        "batch_size": batch_size,
    }

    simulator = Simulator(mu=mu, tr=tr, er=er)

    env = Env(simulator=simulator,
              episode=episode,
              num_trikes=num_trikes,
              capacity=capacity,
              delta=delta,
              rho=rho)

    run_baseline(env, config)
    # run_vanilla_dqn(env, config)
    run_dqn(env, config)
    run_vanilla_pg(env, config)


if __name__ == "__main__":
    main()
