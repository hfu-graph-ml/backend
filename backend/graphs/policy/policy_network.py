from typing import Callable, Dict, List, Optional, Tuple, Type, Union
import sys

import gym
import torch as th
from torch import nn

from stable_baselines3.common.policies import ActorCriticPolicy


class CustomNetwork(nn.Module):
  """
  Custom network for policy and value function.
  It receives as input the features extracted by the feature extractor.

  :param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
  :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
  :param last_layer_dim_vf: (int) number of units for the last layer of the value network
  """

  def __init__(
      self,
      cfg: dict,
      feature_dim: int,
      last_layer_dim_pi: int = 42,
      last_layer_dim_vf: int = 64,
  ):
    super(CustomNetwork, self).__init__()

    self.cfg = cfg
    # IMPORTANT:
    # Save output dimensions, used to create the distributions
    self.latent_dim_pi = last_layer_dim_pi
    self.latent_dim_vf = last_layer_dim_vf

    # Policy network (GCPN eq 4)
    self.a_f_mlp = nn.Sequential(
        nn.Linear(feature_dim, feature_dim),
        nn.ReLU(),
        nn.Linear(feature_dim, 1),
        nn.Flatten(),
        nn.Softmax(dim=1),
    )

    self.a_s_mlp = nn.Sequential(
        nn.Linear(feature_dim + 1, feature_dim),
        nn.ReLU(),
        nn.Linear(feature_dim, 1),
        nn.Flatten(),
        nn.Softmax(dim=1),
    )

    self.a_t_mlp_1 = nn.Sequential(
        nn.Linear(feature_dim, feature_dim, bias=False),
        nn.ReLU(),
    )
    self.a_t_mlp_2 = nn.Sequential(
        nn.Linear(feature_dim, 2),
        nn.Softmax(dim=1),
    )

    # Value network
    # TODO: is value network where the discriminator ("expert"?) should be used?
    self.value_net = nn.Sequential(
        nn.Linear(feature_dim, last_layer_dim_vf), nn.ReLU()
    )

  def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
    """
    :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
        If all layers are shared, then ``latent_policy == latent_value``
    """
    return self.forward_actor(features), self.forward_critic(features)

  def forward_actor(self, features: th.Tensor) -> th.Tensor:
    # (GCPN eq 4)

    a_f_dist = self.a_f_mlp(features)
    a_f = th.multinomial(a_f_dist, 1)
    a_f_dist = th.where(th.tile(a_f, (1, a_f_dist.shape[1])) == th.tile(
        th.arange(0, a_f_dist.shape[1]), (a_f_dist.shape[0], 1)), th.ones_like(a_f_dist), th.zeros_like(a_f_dist))
    # TODO: does first node have to be from the connected graph?

    a_f_repeated = th.tile(th.reshape(a_f, (a_f.shape[0], 1, 1)), (1, features.shape[1], 1))
    emb_cat_s = th.cat((a_f_repeated, features), axis=2)
    a_s_dist = self.a_s_mlp(emb_cat_s)
    test_s = a_s_dist
    a_s_dist = a_s_dist + sys.float_info.epsilon  # TODO: is adding epsilon a good idea?
    a_s_dist = th.where(th.tile(a_f, (1, a_s_dist.shape[1])) == th.tile(th.arange(0, a_s_dist.shape[1]), (
        a_s_dist.shape[0], 1)), th.zeros_like(a_s_dist), a_s_dist)  # make sure that first node is not picked as second node

    # if th.sum(a_s_dist) == 0: # fix 0 distribution
    #   a_s_dist = th.where(th.tile(a_f, (1, a_s_dist.shape[1])) == th.tile(th.arange(0, a_s_dist.shape[1]), (a_s_dist.shape[0], 1)), th.zeros_like(a_s_dist), th.ones_like(a_s_dist))

    a_s = th.multinomial(a_s_dist, 1)

    a_s_dist = th.where(th.tile(a_s, (1, a_s_dist.shape[1])) == th.tile(th.arange(0, a_s_dist.shape[1]), (a_s_dist.shape[0], 1)), th.ones_like(
        a_s_dist), th.zeros_like(a_s_dist))  # make distribution so that the already sampled node is sampled again later

    a_t_dist = self.a_t_mlp_1(features)
    a_t_dist = th.sum(a_t_dist, dim=1)
    a_t_dist = self.a_t_mlp_2(a_t_dist)
    a_t = th.multinomial(a_t_dist, 1)
    a_t_dist = th.where(th.tile(a_t, (1, a_t_dist.shape[1])) == th.tile(th.arange(0, a_t_dist.shape[1]), (a_t_dist.shape[0], 1)), th.ones_like(
        a_t_dist), th.zeros_like(a_t_dist))  # make distribution so that the already sampled termination is sampled again later

    a = th.cat((a_f_dist, a_s_dist, a_t_dist), dim=1)
    return a

  def forward_critic(self, features: th.Tensor) -> th.Tensor:
    v = self.value_net(features)
    v = th.sum(v, axis=1)
    return v


class CustomActorCriticPolicy(ActorCriticPolicy):
  def __init__(
      self,
      observation_space: gym.spaces.Space,
      action_space: gym.spaces.Space,
      lr_schedule: Callable[[float], float],
      net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
      activation_fn: Type[nn.Module] = nn.Tanh,
      *args,
      **kwargs,
  ):
    self.cfg = kwargs.pop('cfg')

    super(CustomActorCriticPolicy, self).__init__(
        observation_space,
        action_space,
        lr_schedule,
        net_arch,
        activation_fn,
        *args,
        **kwargs,
    )
    # Disable orthogonal initialization
    self.ortho_init = False

  def _build_mlp_extractor(self) -> None:
    self.mlp_extractor = CustomNetwork(self.cfg, self.features_dim)
