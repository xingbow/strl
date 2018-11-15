from agent import DQNAgent, PGAgent
from env import Env
from simulator import Simulator

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

summary = {}


def save_losses_curve(func):
    def wrapper(*args, **kwargs):
        loss = func(*args, **kwargs)
        plt.plot(loss, marker='x', label=func.__name__.strip('run_'))
        summary[func.__name__.strip('run_')] = loss
        return loss
    return wrapper


@save_losses_curve
def run_baseline(env, config):
    num_epochs = config['num_epochs']

    losses = []
    for epoch in range(num_epochs):
        env.reset()
        while not env.done:
            # env.render()
            action = env.pruning()
            if action is None:
                action = np.random.randint(env.action_size)

            _, reward = env.step(action)

        losses += [env.loss]
        print('{}/{} loss {}, smooth loss {}'.format(
            epoch, num_epochs,
            losses[-1], np.mean(losses)))

    return losses


@save_losses_curve
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
            # env.render()

            action = env.pruning()
            if action is None:
                action = agent.act(state)

            observation, reward = env.step(action)

            actions.append(action)
            observations.append(observation)
            next_state = create_state()

            agent.remember(state, action, reward, next_state, env.done)
            state = next_state

        agent.memory = [l[:2] + (r,) + l[3:]
                        for l, r in zip(agent.memory, env.rewards)]
        agent.replay(batch_size)
        losses += [env.loss]

        print('{}/{} loss {}, smooth loss {}'.format(
            epoch, num_epochs,
            losses[-1], np.mean(losses)))

    return losses


@save_losses_curve
def run_pg(env, config):
    num_epochs = config['num_epochs']

    action_size = env.action_size
    state_size = env.observation2.shape[0]
    agent = PGAgent(state_size, action_size)

    losses = []
    for epoch in range(num_epochs):

        _ = env.reset()
        state = env.observation2
        while not env.done:
            # env.render()

            action = env.pruning()
            if action is None:
                action, prob = agent.act(state)
            else:
                agent.act(state)
                prob = 1

            env.step(action)

            agent.remember(state, action, prob, None)

            next_state = env.observation2
            state = next_state

        agent.rewards = env.rewards
        agent.replay()
        losses += [env.loss]

        print('{}/{} loss {}, smooth loss {}'.format(
            epoch, num_epochs,
            losses[-1], np.mean(losses)))

    return losses


def main():
    num_trikes = 10
    capacity = 10
    num_epochs = 25
    batch_size = 32
    rho = -1
    mu = 200 / 60
    tr = 60 * 3
    er = 3 * 60
    real = False
    episode = 0
    community = 1

    config = {
        "num_epochs": num_epochs,
        "batch_size": batch_size,
    }

    simulator = Simulator(episode=episode,
                          community=community,
                          mu=mu,
                          tr=tr,
                          er=er,
                          real=real)

    env = Env(simulator=simulator,
              num_trikes=num_trikes,
              capacity=capacity,
              rho=rho)

    # run_baseline(env, config)
    run_dqn(env, config)
    run_pg(env, config)
    plt.legend()

    plt.savefig('../fig/result.png')

    pd.DataFrame(summary).to_csv('../fig/results.csv', index=None)


if __name__ == "__main__":
    main()
