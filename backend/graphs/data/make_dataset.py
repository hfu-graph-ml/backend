import random
import networkx as nx
import copy
import matplotlib.pyplot as plt

from config.config import config

random.seed(101)

# the basket of nodes we use
base_graph = nx.Graph()
for i in range(0, config['data']['num_nodes']):
  base_graph.add_node(i, age=i//2+1, location=i//2+1)
# nx.draw(base_graph, with_labels=True)


# handcrafted graph
dataset_hand = copy.deepcopy(base_graph)
edge_list = [
  (0, 1),
  (1, 3),
  (3, 7),
  (7, 8),
  (8, 19),
  (19, 10),
  (10, 11),
  (11, 4),
  (4, 13),
  (13, 16),
  (16, 6),
  (6, 2),
  (2, 14),
  (14, 5),
  (5, 17),
  (17, 12),
  (12, 9),
  (9, 18),
  (18, 15),
  (15, 0),
  (15, 3),
  (7, 18),
  (8, 9),
  (12, 19),
  (10, 17),
  (5, 11),
  (14, 4),
  (2, 13),
]
dataset_hand.add_edges_from(edge_list)
# nx.draw_networkx(dataset_hand, with_labels=True)
# plt.show()

# randomly generated grid datasets with nodes from basket
dataset = [copy.deepcopy(dataset_hand) for i in range(10)]
