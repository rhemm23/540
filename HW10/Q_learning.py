from collections import defaultdict
from collections import deque

import numpy as np
import random
import pickle
import time
import gym

EPISODES = 20000
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

  Q_table = defaultdict(default_Q_value) # starts with a pessimistic estimate of zero reward for each state.
  episode_reward_record = deque(maxlen=100)

  for i in range(EPISODES):
    episode_reward = 0

    #TODO PERFORM Q LEARNING

    if i % 100 ==0 and i > 0:
      print("LAST 100 EPISODE AVERAGE REWARD: " + str(sum(list(episode_reward_record)) / 100))
      print("EPSILON: " + str(EPSILON))

  ####DO NOT MODIFY######
  model_file = open('Q_TABLE.pkl' ,'wb')
  pickle.dump([Q_table,EPSILON],model_file)
  #######################
