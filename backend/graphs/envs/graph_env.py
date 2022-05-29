import gym
import copy
import time
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

import config.config as config


class GraphEnv(gym.Env):
  def __init__(self):
    pass

  def init(self, cfg: config.Config, base_graph, dataset, reward_step_total=1):
    self.cfg = cfg
    self.base_graph = base_graph
    self.graph = copy.deepcopy(self.base_graph)
    self.reward_step_total = reward_step_total

    self.counter = 0

    self.dataset = dataset
    self.num_nodes = self.base_graph.number_of_nodes()

    self.action_space = gym.spaces.MultiDiscrete([self.num_nodes, self.num_nodes, 2])
    self.observation_space = gym.spaces.Dict({
        'adj': gym.spaces.Box(low=0, high=self.num_nodes, shape=(1, self.num_nodes, self.num_nodes), dtype=np.uint8),
        'node': gym.spaces.Box(low=0, high=self.num_nodes, shape=(1, self.num_nodes, self.cfg['general']['num_node_features']), dtype=np.uint8)
    })

    self.level = 0  # for curriculum learning, level starts with 0, and increase afterwards

  def step(self, action):
    # init
    info = {}  # info we care about
    self.graph_old = copy.deepcopy(self.graph)

    # take action
    if action[2] == 0:   # not stop
      stop = False
      self._add_edge(action)  # add new edge
    else:   # stop
      stop = True

    # calculate intermediate rewards
    # todo: add neccessary rules for the task
    if self.graph.number_of_edges() - self.graph_old.number_of_edges() > 0:
      reward_step = self.reward_step_total / self.num_nodes
      # successfully added node/edge
    else:
      reward_step = -self.reward_step_total / self.num_nodes  # edge
      self.graph = self.graph_old
      # already exists

    # calculate and use terminal reward
    if stop or len(list(nx.isolates(self.graph))) == 0:
      # property rewards
      # todo: add property reward
      reward_terminal = 1  # arbitrary choice

      new = True  # end of episode
      reward = reward_step + reward_terminal

      # print terminal graph information
      info['final_stat'] = reward_terminal
      info['reward'] = reward
      info['stop'] = stop
    # use stepwise reward
    else:
      new = False
      reward = reward_step

    # get observation
    ob = self.get_observation()

    self.counter += 1
    if new:
      self.counter = 0

    print(action)
    nx.draw(self.graph, with_labels=True)
    plt.show()

    return ob, reward, new, info

  def reset(self):  # TODO: is reset even called?
    print('reset')
    self.graph = copy.deepcopy(self.base_graph)
    self.counter = 0
    ob = self.get_observation()
    return ob

  def render(self, mode='human'):
    return

  def _add_edge(self, action):
    """
    :param action: [first_node, second_node]
    :return:
    """

    if self.graph.has_edge(int(action[0]), int(action[1])) or int(action[0]) == int(action[1]):
      return False
    else:
      self.graph.add_edge(int(action[0]), int(action[1]))
      return True

  # for graphs without features
  def get_observation(self, feature='deg'):
    """
    :return: ob, where ob['adj'] is E with dim 1 x n x n and ob['node']
    is F with dim 1 x n x m.
    """

    ob = {}
    ob['adj'] = np.expand_dims(nx.adjacency_matrix(self.graph).todense(), axis=0)
    ob['node'] = np.expand_dims(np.array([[feature for feature in node[1].values()]
                                for node in self.base_graph.nodes(data=True)]), axis=0)

    return ob

  def get_expert(self, batch_size, is_final=False, curriculum=0,
                 level_total=6, level=0):

    ac = np.zeros((batch_size, 3))

    # TODO: finish get_expert
    print('get_expert()')

    # select graph
    dataset_len = len(self.dataset)
    for i in range(batch_size):
      # get a subgraph
      if curriculum == 1:
        ratio_start = level / float(level_total)
        ratio_end = (level + 1) / float(level_total)
        idx = np.random.randint(int(ratio_start * dataset_len),
                                int(ratio_end * dataset_len))
      else:
        idx = np.random.randint(0, dataset_len)
      graph = self.dataset[idx]
      edges = graph.edges()
      # select the edge num for the subgraph
      if is_final:
        edges_sub_len = len(edges)
      else:
        edges_sub_len = random.randint(1, len(edges))
      edges_sub = random.sample(edges, k=edges_sub_len)
      graph_sub = nx.Graph(edges_sub)
      graph_sub = max(nx.connected_component_subgraphs(graph_sub),
                      key=len)
      if is_final:  # when the subgraph the whole graph, the expert show
        # stop sign
        node1 = random.randint(0, graph.number_of_nodes() - 1)
        while True:
          node2 = random.randint(0, graph.number_of_nodes())
          if node2 != node1:
            break
        edge_type = 0
        ac[i, :] = [node1, node2, edge_type, 1]  # stop
      else:
        # random pick an edge from the subgraph, then remove it
        edge_sample = random.sample(graph_sub.edges(), k=1)
        graph_sub.remove_edges_from(edge_sample)
        graph_sub = max(nx.connected_component_subgraphs(graph_sub),
                        key=len)
        edge_sample = edge_sample[0]  # get value
        # get action
        if edge_sample[0] in graph_sub.nodes() and edge_sample[
                1] in graph_sub.nodes():
          node1 = graph_sub.nodes().index(edge_sample[0])
          node2 = graph_sub.nodes().index(edge_sample[1])
        elif edge_sample[0] in graph_sub.nodes():
          node1 = graph_sub.nodes().index(edge_sample[0])
          node2 = graph_sub.number_of_nodes()
        elif edge_sample[1] in graph_sub.nodes():
          node1 = graph_sub.nodes().index(edge_sample[1])
          node2 = graph_sub.number_of_nodes()
        else:
          print('Expert policy error!')
        edge_type = 0
        ac[i, :] = [node1, node2, edge_type, 0]  # don't stop
        # print('action',[node1,node2,edge_type,0])
      # print('action',ac)
      # plt.axis("off")
      # nx.draw_networkx(graph_sub)
      # plt.show()
      # get observation
      n = graph_sub.number_of_nodes()
      F = np.zeros((1, self.num_nodes, 1))
      F[0, :n + 1, 0] = 1
      if self.is_normalize:
        ob['adj'][i] = self.normalize_adj(F)
      else:
        ob['node'][i] = F
      # print(F)
      E = np.zeros((1, self.num_nodes, self.num_nodes))
      E[0, :n, :n] = np.asarray(nx.to_numpy_matrix(graph_sub))[np.newaxis, :, :]
      E[0, :n + 1, :n + 1] += np.eye(n + 1)
      ob['adj'][i] = E
      # print(E)

    return ob, ac
