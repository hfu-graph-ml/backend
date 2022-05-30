import click

from stable_baselines3 import PPO

import config.config as config
from envs.graph_env import GraphEnv
from data.make_dataset import base_graph, dataset

import networkx as nx
import matplotlib.pyplot as plt

# Load config
config_path = 'example.toml'
cfg, err = config.read(config_path)
if err != None:
  click.echo(f'Failed to load config \'{config_path}\': {err}')

# Init gym graph environment
env = GraphEnv()
env.init(cfg, base_graph=base_graph, dataset=dataset)
env.seed(101)

# Load model
model = PPO.load("saved_models/model_trained_test", env=env)

# Start generating
done = False
obs = env.reset()

# while not done or max steps reached:
for i in range(1000):
  action, _states = model.predict(obs, deterministic=True)
  obs, rewards, done, info = env.step(action)

  if done:
    break

nx.draw(info["graph"], with_labels=True)
plt.show()