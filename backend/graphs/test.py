import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from stable_baselines3 import PPO

from config.config import config
from envs.graph_env import GraphEnv
from data.make_dataset import base_graph, dataset

import networkx as nx
import matplotlib.pyplot as plt


# Init gym graph environment
env = GraphEnv()
env.init(base_graph=base_graph, dataset=dataset)
env.seed(101)

# Load model
model = PPO.load("saved_models/model_trained_test/best_model.zip", env=env)

# Start generating
done = False
obs = env.reset()

# generate 10 graphs
for i in range(10):
  done = False
  while not done:
    action, _states = model.predict(obs, deterministic=False)
    obs, rewards, done, info = env.step(action)
    nx.draw(info["graph"], with_labels=True)
    plt.show()

  