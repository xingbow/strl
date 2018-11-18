import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from agent import DQNAgent, PGAgent, RandomAgent, DumbAgent
from env import Env
# from simulator import Simulator

from artificial_simulator import ArtificialSimulator


from sacred import Experiment
from sacred.observers import MongoObserver

ex = Experiment("bike-reposition")
ex.observers.append(MongoObserver.create())


@ex.config
def configuration():
    num_trikes = 4
    capacity = 10
    num_epochs = 200
    batch_size = 32
    rho = -1
    mu = 30 / 3.6  # 30km/h
    tr = 60 * 3
    er = 3 * 60
    real = False
    episode = 0
    community = 1
    scale = 1
    date = "2013/08/20"
    resample = False


def run(env, agent, num_epochs):
    num_renders = 1
    name = agent.__class__.__name__

    state = env.reset()

    for epoch in range(1, num_epochs + 1):
        done = False
        state = env.reset()

        while not done:
            if epoch % (num_epochs // num_renders) == 0:
                pass
                # env.render()

            action = agent.act(state)

            action = env.pruning() or action

            next_state, reward, done, _ = env.step(action)

            if reward == 0:  # if there are no loss, encourage this action
                reward == 1

            agent.remember(state, action, reward, next_state)

            state = next_state

        agent.replay()

        loss = env.loss
        print('epoch {}, loss {}'.format(epoch, loss))
        ex.log_scalar('{}.loss'.format(name), loss)


@ex.capture
def run_dumb_agent(env, _config):
    agent = DumbAgent(env.state_size,
                      env.action_size,
                      batch_size=_config['batch_size'])
    run(env, agent, num_epochs=10)


@ex.capture
def run_random_agent(env, _config):
    agent = RandomAgent(env.state_size,
                        env.action_size,
                        batch_size=_config['batch_size'])
    run(env, agent, num_epochs=10)


@ex.capture
def run_dqn_agent(env, _config):
    agent = DQNAgent(env.state_size,
                     env.action_size,
                     batch_size=_config['batch_size'])

    run(env, agent, num_epochs=_config['num_epochs'])


@ex.capture
def run_pg_agent(env, _config):
    agent = PGAgent(env.state_size,
                    env.action_size)

    run(env, agent, num_epochs=_config['num_epochs'])


@ex.automain
def main(_config):
    # simulator = Simulator(date=_config['date'],
    #                       scale=_config['scale'],
    #                       episode=_config['episode'],
    #                       community=_config['community'],
    #                       mu=_config['mu'],
    #                       tr=_config['tr'],
    #                       er=_config['er'],
    #                       real=_config['real'])

    simulator = ArtificialSimulator(scale=_config['scale'],
                                    mu=_config['mu'],
                                    tr=_config['tr'],
                                    er=_config['er'])

    env = Env(simulator=simulator,
              num_trikes=_config['num_trikes'],
              capacity=_config['capacity'],
              rho=_config['rho'],
              resample=_config['resample'])

    run_dumb_agent(env)
    run_random_agent(env)
    run_pg_agent(env)
    run_dqn_agent(env)
