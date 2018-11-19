import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from agent import DQNAgent, PGAgent, RandomAgent, DumbAgent
from env import Env
# from simulator import Simulator

from artificial_simulator import ArtificialSimulator


from sacred import Experiment
from sacred.observers import MongoObserver

ex = Experiment("bike-reposition-artificial")
ex.observers.append(MongoObserver.create())


@ex.config
def configuration():
    # experiment
    num_train_epochs = 200
    num_test_epochs = 10

    # agent
    num_trikes = 3
    capacity = 10
    rho = -1
    mu = 30 / 3.6  # 30km/h
    tr = 60 * 3
    er = 3 * 60

    # reinforcement learning
    gamma = 0.95        # discount rate

    # neural net
    hidden_dims = [64, 64]
    eta = 1e-3          # learning rate
    batch_size = 128


@ex.capture
def train(env, agent, _config):
    num_epochs = _config['num_train_epochs']
    name = agent.__class__.__name__
    snapshot_epochs = [0, num_epochs // 2]

    for epoch in range(num_epochs):
        done = False
        state = env.reset()

        if epoch in snapshot_epochs:
            snapshots_path = '../fig/{}-{}/'.format(name, epoch)
            # env.book_snapshots(snapshots_path, 200)

        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state)
            state = next_state

        # train
        agent.replay()

        loss = env.loss
        print('train epoch {}, loss {}'.format(epoch, loss))
        ex.log_scalar('{}.train.loss'.format(name), loss)


@ex.capture
def test(env, agent, _config):
    num_epochs = _config['num_test_epochs']
    name = agent.__class__.__name__
    snapshot_epochs = [0, num_epochs - 1]

    for epoch in range(num_epochs):
        done = False
        state = env.reset()

        if epoch in snapshot_epochs:
            snapshots_path = '../fig/{}-{}/'.format(name, epoch)
            # env.book_snapshots(snapshots_path, 200)

        while not done:
            action = agent.act(state)
            next_state, _, done, _ = env.step(action)
            state = next_state

        loss = env.loss
        print('test epoch {}, loss {}'.format(epoch, loss))
        ex.log_scalar('{}.test.loss'.format(name), loss)


@ex.capture
def run_dumb_agent(env, _config):
    agent = DumbAgent()
    test(env, agent)


@ex.capture
def run_random_agent(env, _config):
    agent = RandomAgent(env.action_size)
    test(env, agent)


@ex.capture
def run_dqn_agent(env, _config):
    agent = DQNAgent(env.state_size,
                     env.action_size,
                     batch_size=_config['batch_size'],
                     hidden_dims=_config['hidden_dims'],
                     gamma=_config['gamma'],
                     eta=_config['eta'])

    train(env, agent)
    test(env, agent)


@ex.capture
def run_pg_agent(env, _config):
    agent = PGAgent(env.state_size,
                    env.action_size,
                    batch_size=_config['batch_size'],
                    hidden_dims=_config['hidden_dims'],
                    gamma=_config['gamma'],
                    eta=_config['eta'])

    train(env, agent)
    test(env, agent)


@ex.automain
def main(_config):
    simulator = ArtificialSimulator(mu=_config['mu'],
                                    tr=_config['tr'],
                                    er=_config['er'])

    env = Env(simulator=simulator,
              num_trikes=_config['num_trikes'],
              capacity=_config['capacity'],
              rho=_config['rho'])

    run_dumb_agent(env)
    run_random_agent(env)
    run_pg_agent(env)
    run_dqn_agent(env)
