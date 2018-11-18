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
    num_trikes = 3
    capacity = 10
    num_epochs = 300
    batch_size = 128
    rho = -1
    mu = 30 / 3.6  # 30km/h
    tr = 60 * 3
    er = 3 * 60


def run(env, agent, num_epochs):
    name = agent.__class__.__name__

    state = env.reset()

    for epoch in range(num_epochs):
        done = False
        state = env.reset()

        frame = 0
        while not done:
            if epoch == 0 or epoch == num_epochs - 1:
                # env.render()
                env.save_frame(
                    '../fig/{}-{}/{}.svg'.format(name, epoch, frame))
                frame += 1

            action = agent.act(state)

            action = env.pruning() or action

            next_state, reward, done, _ = env.step(action)

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
    run(env, agent, num_epochs=_config['num_epochs'])


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
