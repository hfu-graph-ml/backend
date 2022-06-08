import gym
import torch as th
import torch.nn as nn

from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from config import config


class GCNFeaturesExtractor(BaseFeaturesExtractor):
  """
  :param observation_space: (gym.spaces.Dict)
  :param features_dim: (int) Number of features extracted.
      This corresponds to the number of unit for the last layer.
  """

  def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = config['model']['emb_size']):
    super(GCNFeaturesExtractor, self).__init__(observation_space, features_dim)

    self.config = config

    self.emb_layer = nn.Linear(observation_space['node'].shape[2], self.config['model']['emb_size'], bias=False)

    shared_weights = th.empty([1, observation_space['adj'].shape[1], self.config['model']['emb_size'],
                              self.config['model']['emb_size']])  # TODO: is this right?
    # shared_weights = th.empty([1, observation_space['adj'].shape[1], observation_space['node'].shape[-1], self.config['model']['emb_size']])
    nn.init.xavier_uniform_(shared_weights)
    self.GCN_layer_shared_weights = nn.Parameter(shared_weights)

  def forward(self, observations) -> th.Tensor:
    # compute node embeddings with GCN (eq 2)

    emb_node = self.emb_layer(observations['node'])
    emb_node = GCN_layer(self, observations['adj'], emb_node)
    for i in range(self.config['model']['layer_num_g']-2):
      emb_node = GCN_layer(self, observations['adj'], emb_node)
    emb_node = GCN_layer(self, observations['adj'], emb_node, is_act=False)
    emb_node = th.squeeze(emb_node, axis=1)

    if self.config['debugging']['print_extracted_features']:
      print(emb_node)

    return emb_node


def GCN_layer(self, adj, node_feature, is_act=True):
  edge_dim = adj.size(1)
  batch_size = adj.size(0)
  in_channels = node_feature.size(-1)

  node_embedding = adj @ th.tile(node_feature, [1, edge_dim, 1, 1])
  node_embedding = node_embedding @ th.tile(self.GCN_layer_shared_weights, [batch_size, 1, 1, 1])
  if is_act:
    node_embedding = nn.ReLU()(node_embedding)
  node_embedding = th.sum(node_embedding, axis=1, keepdim=True)
  return node_embedding
