import imp
import os
import gym
import copy
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from utils import draw_graph

from config import config
from evaluation import valid_table_graph, highest_possible_edge_mood_score
from data.edge_score_matrix import get_edge_scores


class GraphEnv(gym.Env):
  def __init__(self):
    pass

  def init(self, base_graph, reward_step_total=1):
    self.config = config
    self.base_graph = base_graph
    self.graph = copy.deepcopy(self.base_graph)
    self.reward_step_total = reward_step_total

    self.counter = 0

    self.num_nodes = self.base_graph.number_of_nodes()
    self.max_edges = (((self.num_nodes * 3) - 4) / 2)

    self.action_space = gym.spaces.MultiDiscrete([self.num_nodes, self.num_nodes])
    self.observation_space = gym.spaces.Dict({
        'adj': gym.spaces.Box(low=0, high=self.num_nodes, shape=(1, self.num_nodes, self.num_nodes), dtype=np.uint8),
        'node': gym.spaces.Box(low=0, high=self.num_nodes, shape=(1, self.num_nodes, self.config['data']['num_node_features']), dtype=np.uint8)
    })

    self.level = 0  # for curriculum learning, level starts with 0, and increase afterwards

    # draw graph
    if self.config['debugging']['draw_graph']:
      draw_graph(self.graph, layout=None)

  def step(self, action):
    # init
    info = {}  # info we care about

    graph_is_valid = None
    mood_scores = []

    # take action or not
    edge_added = self._add_edge(action)

    # print actions
    if self.config['debugging']['print_actions']:
      print(action)

    # draw graph
    if self.config['debugging']['draw_graph']:
      draw_graph(self.graph, layout=None)

    # get observation
    ob = self.get_observation()

    # wheter to stop after this step
    stop = self.graph.number_of_edges() >= (
        ((self.config['data']['num_nodes'] * 3) - 4) / 2) or self.counter >= self.config['training']['max_steps']

    # calculate intermediate rewards
    if edge_added:
      reward_step = self.config['rewards']['step_edge_correct']
    else:
      reward_step = self.config['rewards']['step_edge_incorrect']

    # calculate and use terminal reward
    if stop:
      new = True  # end of episode

      graph_is_valid = valid_table_graph(self.graph, self.num_nodes, self.max_edges)

      if graph_is_valid:
        mood_scores = get_edge_scores(self.graph.edges())
        reward_terminal = (highest_possible_edge_mood_score - np.mean(mood_scores)) / \
            highest_possible_edge_mood_score * self.config['rewards']['terminal_valid_score_multiplier']

        # draw finalized graph
        if self.config['debugging']['draw_correct_graphs']:
          print(reward_terminal)
          draw_graph(self.graph)

      else:
        reward_terminal = self.config['rewards']['terminal_invalid']

      reward = reward_step + reward_terminal
      # print terminal graph information
      info['final_stat'] = reward_terminal
      info['reward'] = reward
      info['stop'] = stop

    # use stepwise reward
    else:
      new = False
      reward = reward_step

    self.counter += 1
    if new:
      self.counter = 0

    info['graph'] = self.graph
    info['action_valid'] = int(edge_added)
    info['graph_valid'] = int(graph_is_valid) if graph_is_valid != None else -np.inf
    info['mood_score'] = np.sum(mood_scores) if len(mood_scores) > 0 else np.nan

    return ob, reward, new, info

  def reset(self):
    self.graph = copy.deepcopy(self.base_graph)
    self.counter = 0
    ob = self.get_observation()
    return ob

  def render(self):
    return

  def _add_edge(self, action):
    """
    :param action: [first_node, second_node]
    :return:
    """

    if self.graph.has_edge(int(action[0]), int(action[1])) or int(action[0]) == int(action[1]) or self.graph.degree(action[0]) >= 3 or self.graph.degree(action[1]) >= 3:
      return False
    else:
      graph_old = copy.deepcopy(self.graph)
      self.graph.add_edge(int(action[0]), int(action[1]))
      if not nx.is_bipartite(self.graph):
        self.graph = graph_old
        return False
      return True

  def get_observation(self):
    """
    :return: ob, where ob['adj'] is E with dim 1 x n x n and ob['node']
    is F with dim 1 x n x m.
    """

    ob = {}
    ob['adj'] = np.expand_dims(nx.adjacency_matrix(self.graph).todense(), axis=0)
    ob['node'] = np.expand_dims(np.array([[feature for feature in node[1].values()]
                                for node in self.base_graph.nodes(data=True)]), axis=0)

    return ob
