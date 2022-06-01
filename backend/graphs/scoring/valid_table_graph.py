import networkx as nx
import numpy as np
from networkx.algorithms import bipartite

def valid_table_graph(graph, num_nodes, max_edges):
    correct_number_of_edges = graph.number_of_edges() == max_edges
    correct_degree_histogram = np.array_equal(nx.degree_histogram(graph), [0, 0, 4, num_nodes - 4])
    graph_is_bipartite = bipartite.is_bipartite(graph)

    return correct_number_of_edges and correct_degree_histogram and graph_is_bipartite
