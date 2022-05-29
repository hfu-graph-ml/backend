import click

from stable_baselines3 import PPO

import config.config as config
from policy.policy_network import CustomActorCriticPolicy
from policy.feature_extractor import GCNFeaturesExtractor
from envs.graph_env import GraphEnv
from data.make_dataset import base_graph, dataset

# Load config
config_path = 'example.toml'
cfg, err = config.read(config_path)
if err != None:
  click.echo(f'Failed to load config \'{config_path}\': {err}')

# Init gym graph environment
env = GraphEnv()
env.init(cfg, base_graph=base_graph, dataset=dataset)
env.seed(101)

# Init Proximal Policy Optimization (PPO)
policy_kwargs = dict(
    features_extractor_class=GCNFeaturesExtractor,
    features_extractor_kwargs=dict(cfg=cfg),
    cfg=cfg
)
model = PPO(CustomActorCriticPolicy, env, policy_kwargs=policy_kwargs, verbose=1)

# Start training
model.learn(2048 * 10)
