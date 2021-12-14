from collections import defaultdict
from collections import deque

import numpy as np
import random
import pickle
import time
import gym

EPISODES =   20000
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.99
EPSILON = 1
EPSILON_DECAY = 0.999

def default_Q_value():
  return 0

if __name__ == "__main__":
  random.seed(1)
  np.random.seed(1)
  env = gym.envs.make("FrozenLake-v0")
  env.seed(1)
  env.action_space.np_random.seed(1)

  Q_table = defaultdict(default_Q_value)
  episode_reward_record = deque(maxlen=100)

  for i in range(EPISODES):
    episode_reward = 0
    state = env.reset()
    action = None
    if random.uniform(0, 1) < EPSILON:
      action = env.action_space.sample()
    else:
      predictions = np.array([Q_table[(state, j)] for j in range(env.action_space.n)])
      action = np.argmax(predictions)
    reward = 0
    done = False
    while not done:
      next_state, next_reward, done, _ = env.step(action)
      next_action = None
      if random.uniform(0, 1) < EPSILON:
        next_action = env.action_space.sample()
      else:
        predictions = np.array([Q_table[(next_state, j)] for j in range(env.action_space.n)])
        next_action = np.argmax(predictions)
      
      target = reward + DISCOUNT_FACTOR * Q_table[(next_state, next_action)]
      Q_table[(state, action)] = Q_table[(state, action)] + LEARNING_RATE * (target - Q_table[(state, action)])

      if done:
        Q_table[(next_state, next_action)] = Q_table[(next_state, next_action)] + LEARNING_RATE * (next_reward - Q_table[(next_state, next_action)])

      state = next_state
      action = next_action
      reward = next_reward

      episode_reward += next_reward

    episode_reward_record.append(episode_reward)
    EPSILON *= EPSILON_DECAY

    if i % 100 == 0 and i > 0:
      print("LAST 100 EPISODE AVERAGE REWARD: " + str(sum(list(episode_reward_record)) / 100))
      print("EPSILON: " + str(EPSILON))

  #### DO NOT MODIFY ####
  model_file = open('SARSA_Q_TABLE.pkl' ,'wb')
  pickle.dump([Q_table, EPSILON], model_file)
  #######################
