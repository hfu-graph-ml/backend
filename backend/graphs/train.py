from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
from stable_baselines3.common.utils import set_random_seed
from callbacks.save_best_model import SaveOnBestTrainingRewardCallback
from callbacks.config_logger import ConfigLoggerCallback
from data.make_dataset import base_graph, dataset
from envs.graph_env import GraphEnv
from policy.feature_extractor import GCNFeaturesExtractor
from policy.policy_network import CustomActorCriticPolicy
from config.config import config
from datetime import datetime
import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
training_id = timestamp

training_id = "test_temp" # uncomment for test runs

# Create log dir
model_dir = f"saved_models/{training_id}/"
os.makedirs(model_dir, exist_ok=True)


def make_env(rank, seed=0):
  def _init():
    env = GraphEnv()
    env.init(base_graph=base_graph, dataset=dataset)
    env.seed(seed + rank)
    return env
  set_random_seed(seed)
  return _init


# Init gym vectorized graph environments
venv = DummyVecEnv([make_env(i) for i in range(4)])
# venv = SubprocVecEnv([make_env(i) for i in range(8)])
venv = VecMonitor(venv, model_dir)

# Init Proximal Policy Optimization (PPO)
policy_kwargs = dict(features_extractor_class=GCNFeaturesExtractor)
model = PPO(CustomActorCriticPolicy, venv, policy_kwargs=policy_kwargs,
            n_steps=config["training"]["steps_per_epoch"], tensorboard_log=f"./tensorboard_logs/{training_id}", verbose=1)

# Start training
model.learn(config["training"]["steps_per_epoch"] * config["training"]["epochs"],
            callback=[SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=model_dir), ConfigLoggerCallback(config)])
