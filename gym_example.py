import gym


class Agent():
    def __init__(self):
        pass

    def act(self, state):
        return [0]

    def reward(self, amt):
        pass


def combine(action1, action2):
    return 0


def main():
    env = gym.make('CartPole-v1')

    print('Shape of observation of the game: '
          '{}'.format(env.observation_space.shape[0]))

    print('Shape of action of the game: '
          '{}'.format(env.action_space.n))

    state = env.reset()

    done = False

    trike1 = Agent()
    trike2 = Agent()

    while not done:             # main loop
        # env.render()        print(env.info())

        action1 = trike1.act(state)
        action2 = trike2.act(state)

        action = combine(action1, action2)
        next_state, reward, done, _ = env.step(action)

        trike1.reward(reward)
        trike2.reward(reward)

    print('Done!')
    env.close()


if __name__ == '__main__':
    main()
