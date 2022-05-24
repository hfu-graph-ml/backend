from networkx.generators.lattice import grid_2d_graph
import random
import networkx as nx
import numpy as np

random.seed(101)

# the basket of nodes we use
base_graph = nx.Graph()
for i in range(0, 20):
  base_graph.add_node(i, age=i+1, name=i+1)
base_graph.add_edge(0, 1)

# nx.draw(base_graph, with_labels=True)


# handcrafted graph
dataset_hand = base_graph
dataset_hand.add_edge(1, 3)
dataset_hand.add_edge(3, 7)
dataset_hand.add_edge(7, 8)
dataset_hand.add_edge(8, 19)
dataset_hand.add_edge(19, 10)
dataset_hand.add_edge(10, 11)
dataset_hand.add_edge(11, 4)
dataset_hand.add_edge(4, 13)
dataset_hand.add_edge(13, 16)
dataset_hand.add_edge(16, 6)
dataset_hand.add_edge(6, 2)
dataset_hand.add_edge(2, 14)
dataset_hand.add_edge(14, 5)
dataset_hand.add_edge(5, 17)
dataset_hand.add_edge(17, 12)
dataset_hand.add_edge(12, 9)
dataset_hand.add_edge(9, 18)
dataset_hand.add_edge(18, 15)
dataset_hand.add_edge(15, 0)
dataset_hand.add_edge(15, 3)
dataset_hand.add_edge(7, 18)
dataset_hand.add_edge(8, 9)
dataset_hand.add_edge(12, 19)
dataset_hand.add_edge(10, 17)
dataset_hand.add_edge(5, 11)
dataset_hand.add_edge(14, 4)
dataset_hand.add_edge(2, 13)
nx.draw_networkx(dataset_hand, with_labels=True)


# randomly generated grid datasets with nodes from basket
dataset = [dataset_hand for i in range(10)]
# dataset = dataset[0]
# while_counter = 0

# while True:
#   while_counter = while_counter+1
#   deg_hist = np.array(nx.degree_histogram(dataset))
#   while len(deg_hist) < 4:
#     deg_hist = np.append(deg_hist, 0)

#   print(deg_hist)

#   if while_counter < 20 and not np.array_equal(deg_hist, [0, 0, 4, 16]):
#     nodes_deg_2 = [n[0] for n in dataset.degree() if n[1] == 2]
#     nodes_deg_3 = [n[0] for n in dataset.degree() if n[1] == 3]
  
#     if deg_hist[3] < 16:
#       nodes_to_connect = np.concatenate((nodes_deg_2, nodes_deg_3))
#       nodes_to_connect = nodes_deg_2

#       if deg_hist[2] == 4:
#         nodes_to_connect = np.concatenate((nodes_deg_2, nodes_deg_3))
#     else:
#       nodes_to_connect = dataset.nodes
#     print(nodes_to_connect)
#     sampled_nodes = random.sample(nodes_to_connect, 2)
#     dataset.add_edge(sampled_nodes[0], sampled_nodes[1])
#   else:
#     break