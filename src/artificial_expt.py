import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import os

from agent import DQNAgent, PGAgent, RandomAgent, DumbAgent
from env import Env
from artificial_simulator import ArtificialSimulator

FILE_DIR = os.path.dirname(os.path.realpath(__file__))
ROOT_DIR = os.path.dirname(FILE_DIR)


from sacred import Experiment
from sacred.observers import MongoObserver, FileStorageObserver

ex = Experiment("bike-reposition-real")
# ex.observers.append(MongoObserver.create())
ex.observers.append(FileStorageObserver.create(
    os.path.join(ROOT_DIR, 'results')))


@ex.config
def configuration():
    # experiment
    num_rounds = 200

    # real world parameters
    num_trikes = 3
    capacity = 10
    rho = 24  # for pruning
    mu = 200 / 60
    tr = 60 * 3
    er = 60 * 3

    # agents
    gamma = 0.95        # discount rate

    # neural net
    hidden_dims = [128, 128]
    eta = 1e-3          # learning rate
    batch_size = 128
    epochs = 5

    snapshots_path = os.path.join(
        ROOT_DIR, 'snapshots', 'json', str(int(time.time())))


@ex.capture
def train(env, agent, round_, snapshot, snapshots_path):
    # switch to test mode, use real data
    name = agent.__class__.__name__

    done = False
    state = env.reset()

    if snapshot:
        snapshots_path += '/train/{}-{}.json'.format(
            name, round_)
        # reset() will clear all snapshot events, so please register after reset()
        env.register_snapshots(snapshots_path, 200)

    while not done:
        action = agent.act(state)
        # action = env.pruning(prob=0.5) or action
        next_state, reward, done, _ = env.step(action)
        agent.remember(state, action, reward, next_state)
        state = next_state

    # train
    agent.replay()

    # print training info
    loss = env.loss
    print('train round {}, loss {}'.format(round_, loss))
    ex.log_scalar('{}.train.loss'.format(name), loss, round_)


@ex.capture
def test(env, agent, round_, snapshot, snapshots_path):
    # switch to test mode, use real data
    name = agent.__class__.__name__

    done = False
    state = env.reset()

    if snapshot:
        snapshots_path += '/test/{}-{}.json'.format(
            name, round_)
        env.register_snapshots(snapshots_path, 200)

    while not done:
        action = agent.act(state)
        next_state, _, done, _ = env.step(action)
        state = next_state

    loss = env.loss
    print('test round {}, loss {}'.format(round_, loss))
    ex.log_scalar('{}.test.loss'.format(name), loss, round_)


@ex.capture
def run_on_agents(env, _config):
    dumb_agent = DumbAgent()

    random_agent = RandomAgent(env.action_size)

    dqn_agent = DQNAgent(env.state_size,
                         env.action_size,
                         batch_size=_config['batch_size'],
                         hidden_dims=_config['hidden_dims'],
                         gamma=_config['gamma'],
                         eta=_config['eta'],
                         epochs=_config['epochs'])

    pg_agent = PGAgent(env.state_size,
                       env.action_size,
                       batch_size=_config['batch_size'],
                       hidden_dims=_config['hidden_dims'],
                       gamma=_config['gamma'],
                       eta=_config['eta'],
                       epochs=_config['epochs'])

    num_rounds = _config['num_rounds']
    num_snapshots = 10
    snapshot_rounds = [num_rounds-1] + \
        np.linspace(0, num_rounds-1, num_snapshots-1, dtype=int).tolist()

    for round_ in range(num_rounds):
        env.simulator.resample()  # train on resampled environment
        snapshot = round_ in snapshot_rounds
        train(env, pg_agent, round_=round_, snapshot=snapshot)
        train(env, dqn_agent, round_=round_, snapshot=snapshot)

        if round_ % 2 == 0:
            # test
            test(env, dumb_agent, round_, snapshot=snapshot)
            test(env, random_agent, round_, snapshot=snapshot)
            test(env, pg_agent, round_, snapshot=snapshot)
            test(env, dqn_agent, round_, snapshot=snapshot)


@ex.automain
def main(_config):
    simulator = ArtificialSimulator(mu=_config['mu'],
                                    tr=_config['tr'],
                                    er=_config['er'])

    env = Env(simulator=simulator,
              num_trikes=_config['num_trikes'],
              capacity=_config['capacity'],
              rho=_config['rho'])

    run_on_agents(env)
