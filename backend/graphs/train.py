from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.utils import set_random_seed
from callbacks import ConfigLoggerCallback, SaveOnBestTrainingRewardCallback, LogValidActions, LogValidGraphs, LogMoodScores, LogBestGraph
from data import base_graph
from envs.graph_env import GraphEnv
from policy.feature_extractor import GCNFeaturesExtractor
from policy.policy_network import CustomActorCriticPolicy
from config import config
from datetime import datetime
import os
import warnings
import cProfile
warnings.simplefilter(action='ignore', category=FutureWarning)


test_run = False
profiling = False
num_profiling_epochs = 100

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

if test_run:
  log_dir = None
  model_dir = None
else:
  training_id = timestamp
  log_dir = f"./tensorboard_logs/{training_id}"
  model_dir = f"saved_models/{training_id}/"
  os.makedirs(model_dir, exist_ok=True)


def make_env(rank, seed=0):
  def _init():
    env = GraphEnv()
    env.init(base_graph=base_graph, preconnect_nodes_probability=config["training"]["preconnect_nodes_probability"])
    env.seed(seed + rank)
    return env
  set_random_seed(seed)
  return _init


# Init gym vectorized graph environments
venv = DummyVecEnv([make_env(i) for i in range(4)])
venv = VecMonitor(venv, model_dir)

# Init Proximal Policy Optimization (PPO)
policy_kwargs = dict(features_extractor_class=GCNFeaturesExtractor)
model = PPO(CustomActorCriticPolicy, venv, policy_kwargs=policy_kwargs,
            n_steps=config["training"]["steps_per_epoch"], batch_size=config["training"]["batch_size"], tensorboard_log=log_dir, verbose=1)


if test_run:
  callback_list = None
else:
  callback_list = [
      SaveOnBestTrainingRewardCallback(model_dir),
      ConfigLoggerCallback(),
      LogValidActions(),
      LogValidGraphs(),
      LogMoodScores(),
      LogBestGraph()
  ]

# Start training
if profiling:
  with cProfile.Profile() as pr:
    model.learn(config["training"]["steps_per_epoch"] * config["training"]
                ["num_training_envs"] * num_profiling_epochs, callback=callback_list)
    profiling_stats_dir = "profiling_stats"
    os.makedirs(profiling_stats_dir, exist_ok=True)
    pr.dump_stats(profiling_stats_dir + "/stats.profile")
else:
  model.learn(config["training"]["steps_per_epoch"] * config["training"]
              ["num_training_envs"] * config["training"]["epochs"], callback=callback_list)
