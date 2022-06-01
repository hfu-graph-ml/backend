from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO
from callbacks.save_best_model import SaveOnBestTrainingRewardCallback
from callbacks.config_logger import ConfigLoggerCallback
from data.make_dataset import base_graph, dataset
from envs.graph_env import GraphEnv
from policy.feature_extractor import GCNFeaturesExtractor
from policy.policy_network import CustomActorCriticPolicy
from config.config import config
import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# Create log dir
log_dir = "saved_models/model_trained_test/"
os.makedirs(log_dir, exist_ok=True)

# Init gym graph environment
env = GraphEnv()
env.init(base_graph=base_graph, dataset=dataset)
env.seed(101)
env = Monitor(env, log_dir)

# Init Proximal Policy Optimization (PPO)
policy_kwargs = dict(
    features_extractor_class=GCNFeaturesExtractor,
)
model = PPO(CustomActorCriticPolicy, env, policy_kwargs=policy_kwargs,
            n_steps=config["training"]["steps_per_epoch"], tensorboard_log="./tensorboard_logs/model_trained_test", verbose=1)

# Start training
model.learn(config["training"]["steps_per_epoch"] * config["training"]["epochs"],
            callback=[SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir), ConfigLoggerCallback(config)])
