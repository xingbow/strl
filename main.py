from agent import DQNAgent
from env import Env
from simulator import Simulator

import numpy as np

num_regions = 10
num_trikes = 5
episode = [0, 600 * 1]
capacity = 5
num_epochs = 1000
batch_size = 32
delta = 60
rho = 2


def run_dqn():
    env = Env(simulator=Simulator(num_regions),
              episode=episode,
              num_regions=num_regions,
              num_trikes=num_trikes,
              capacity=capacity,
              delta=delta,
              rho=rho)

    action_size = num_regions * (2 * capacity + 1)

    agent = DQNAgent(num_regions, action_size)

    def encode_action(action):
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

            action = agent.act(state)
            observation, raw_action, reward = env.step(decode_action(action))
            action = encode_action(raw_action)
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

        print(env.loss)
        print('mean', np.mean(losses[-10:]))

        print("episode: {}/{}, e: {:.2}".format(epoch, num_epochs, agent.epsilon))


def run_baseline():
    env = Env(simulator=Simulator(num_regions),
              episode=episode,
              num_regions=num_regions,
              num_trikes=num_trikes,
              capacity=capacity,
              delta=delta,
              rho=rho)

    losses = []
    num_epochs = 20
    for epoch in range(num_epochs):
        env.reset()
        while not env.done:
            env.step(action=[0, 0])
        losses += [env.loss]
        print('{}/{} loss {}, smooth loss {}'.format(
            epoch, num_epochs,
            losses[-1], np.mean(losses[-10:])))


if __name__ == "__main__":
    run_baseline()
    run_dqn()
