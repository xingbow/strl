import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from agent import DQNAgent, PGAgent, RandomAgent
from env import Env
from simulator import Simulator


from sacred import Experiment
from sacred.observers import MongoObserver

ex = Experiment("bike-reposition")
ex.observers.append(MongoObserver.create())


@ex.config
def configuration():
    num_trikes = 4
    capacity = 10
    num_epochs = 25
    batch_size = 32
    rho = 10
    mu = 200 / 60
    tr = 60 * 3
    er = 3 * 60
    real = False
    episode = 0
    community = 1
    scale = 1
    date = "2013/08/20"


def run(env, agent, num_epochs):
    num_renders = 1
    name = agent.__class__.__name__

    state = env.reset()

    for epoch in range(0, num_epochs):
        done = False
        state = env.reset()

        loss = 0
        while not done:
            if (epoch + 1) % (num_epochs // num_renders) == 0:
                pass
                env.render()

            action = agent.act(state)

            # if pruning mode is on, the env will return some action
            action = env.pruning() or action

            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state)

            state = next_state
            loss = env.loss

        print('epoch {}, loss {}'.format(epoch, loss))
        agent.replay()
        ex.log_scalar('{}.loss'.format(name), loss)


@ex.capture
def run_random_agent(env, _config):
    agent = RandomAgent(env.state_size,
                        env.action_size,
                        batch_size=_config['batch_size'])
    run(env, agent, num_epochs=_config['num_epochs'])


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
    simulator = Simulator(date=_config['date'],
                          scale=_config['scale'],
                          episode=_config['episode'],
                          community=_config['community'],
                          mu=_config['mu'],
                          tr=_config['tr'],
                          er=_config['er'],
                          real=_config['real'])

    env = Env(simulator=simulator,
              num_trikes=_config['num_trikes'],
              capacity=_config['capacity'],
              rho=_config['rho'])

    # run_random_agent(env)
    run_dqn_agent(env)
    run_pg_agent(env)
