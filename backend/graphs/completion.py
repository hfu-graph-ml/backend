from envs.graph_env import GraphEnv
from utils import draw_graph
from stable_baselines3 import PPO
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)


def complete_graph(graph, model_id, n_samples=10, draw_generated_graphs=False):
  # Init gym graph environment
  env = GraphEnv()
  env.init(base_graph=graph)

  # Load model
  model = PPO.load(f"saved_models/{model_id}/best_model.zip", env=env)

  # Generate multiple graphs
  generated_graphs = []

  for _ in range(n_samples):
    done = False
    obs = env.reset()

    while not done:
      action, _ = model.predict(obs, deterministic=False)
      obs, _, done, info = env.step(action)

      if done:
        if info["graph_valid"]:
          generated_graphs.append((info["mood_score"], info["graph"]))
          if draw_generated_graphs == "only_valid":
            draw_graph(info["graph"])

        if draw_generated_graphs == "all":
          draw_graph(info["graph"], layout=None)

  if len(generated_graphs) == 0:
    return False

  # Sort generated graphs by score
  generated_graphs.sort(key=lambda graph: graph[0], reverse=True)

  # Return best graph
  return generated_graphs[0]
