import matplotlib.pyplot as plt
import networkx as nx
from data.make_dataset import base_graph, dataset
from envs.graph_env import GraphEnv
from config.config import config
from stable_baselines3 import PPO
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)


# Init gym graph environment
env = GraphEnv()
env.init(base_graph=base_graph, dataset=dataset)
env.seed(101)

# Load model
model_id = "2022-06-07_08-51-24"
model = PPO.load(f"saved_models/{model_id}/best_model.zip", env=env)

# Start generating 10 graphs
generated_graphs = []
for i in range(10):
  done = False
  obs = env.reset()
  while not done:
    action, _ = model.predict(obs, deterministic=False)
    obs, rewards, done, info = env.step(action)
    if done:
      print(info["final_stat"])
      nx.draw(info["graph"], with_labels=True)
      plt.show()
      generated_graphs.append((info["final_stat"], info["graph"]))

generated_graphs.sort(key=lambda graph: graph[0], reverse=True)
print(generated_graphs)