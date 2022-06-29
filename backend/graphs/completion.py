import os
import time
from datetime import timedelta
import warnings
from envs.graph_env import GraphEnv
from utils import draw_graph
from stable_baselines3 import PPO

warnings.simplefilter(action='ignore', category=FutureWarning)


def complete_graph(graph, model_id, run_for_min_seconds=5, run_for_max_seconds=10, draw_generated_graphs=False):
  timerStart = time.time()
  counter = 0

  # Init gym graph environment
  env = GraphEnv()
  env.init(base_graph=graph)

  # Load model
  model = PPO.load(f"{os.path.join(os.path.dirname(os.path.realpath(__file__)), '../..')}/saved_models/{model_id}/best_model.zip", env=env)

  # Generate multiple graphs
  generated_graphs = []

  print(f"Graph completion started (model {model_id}).")

  while True:
    done = False
    obs = env.reset()

    while not done:
      action, _ = model.predict(obs, deterministic=False)
      obs, _, done, info = env.step(action)

      current_time = time.time()-timerStart
      if current_time >= run_for_min_seconds and generated_graphs:
        print("")
        print(f"Finished after {timedelta(seconds=current_time)}")

        print("Graph completion succeeded!")

        # Sort generated graphs by score
        generated_graphs.sort(key=lambda graph: graph[0], reverse=True)
        
        print(f"{len(generated_graphs)}/{counter} graphs were valid.")
        print(f"Best score was {generated_graphs[0][0]}.")

        # Return best graph
        return generated_graphs[0]

      if current_time >= run_for_max_seconds:
        print("")
        print(f"Finished after {timedelta(seconds=current_time)}")

        print("Graph completion failed!")

        return False

      if done:
        counter = counter+1

        if info["graph_valid"]:
          print("✓", end="")
          generated_graphs.append((info["mood_score"], info["graph"]))
          if draw_generated_graphs == "only_valid":
            draw_graph(info["graph"])
        else:
          print(".", end="")

        if draw_generated_graphs == "all":
          draw_graph(info["graph"], layout=None)

  # should never be reached
  return False
