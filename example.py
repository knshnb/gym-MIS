import gym
import gym_MIS

if __name__ == "__main__":
    env = gym.make('MIS-v0')
    observation = env.reset()
    done = False
    while done == False:
        env.render()
        observation, reward, done, info = env.step(0)
        print('observation:\n{}\nreward:\n{}\ninfo:\n{}'.format(observation, reward, info))

    env.render()