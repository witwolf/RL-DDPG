__author__ = 'witwolf'

import gym
from ddpg import Ddpg

ENV = 'Pendulum-v0'
EPISODE = 200000
STEP = 300


def main():
    env = gym.make(ENV)
    action_dim = env.action_space.shape[0]
    state_dim = env.observation_space.shape[0]
    ddpg = Ddpg(state_dim, action_dim, logdir='/data/log/Pendulum-v0', save_path='/data/model/Pendulum-v0')

    for episode in xrange(1, EPISODE):
        state = env.reset()
        total_reward = 0
        for step in xrange(STEP):
            env.render()

            action = ddpg.exploration_with_noise(state)
            next_state, reward, terminate, _ = env.step(action * 2)
            ddpg.observe_action(state, action, reward, next_state, terminate)
            total_reward += reward
            state = next_state
            if terminate:
                break
        print 'EPISODE %s, REWARD %f' % (episode, total_reward)

        if episode % 100 == 0:
            total_reward = 0
            for i in range(5):
                state = env.reset()
                for step in xrange(STEP):
                    env.render()

                    action = ddpg.exploration(state)
                    next_state, reward, terminate, _ = env.step(action * 2)
                    ddpg.observe_action(state, action, reward, next_state, terminate)
                    total_reward += reward
                    state = next_state
                    if terminate:
                        break
            print 'EPISODE %d, AVE REWARD  %f' % (episode, total_reward / 5.0)


if __name__ == '__main__':
    main()
