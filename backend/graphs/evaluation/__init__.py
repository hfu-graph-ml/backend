import networkx as nx
import numpy as np
from networkx.algorithms import bipartite

def valid_table_graph(graph, num_nodes, max_edges):
    correct_number_of_edges = graph.number_of_edges() == max_edges
    correct_degree_histogram = np.array_equal(nx.degree_histogram(graph), [0, 0, 4, num_nodes - 4])
    graph_is_bipartite = bipartite.is_bipartite(graph)

    return correct_number_of_edges and correct_degree_histogram and graph_is_bipartite

def calculate_mood_score(graph):
    score = 0

    for edge in graph.edges():
        edge_score = (2 - abs(graph.nodes[edge[0]]["age"] - graph.nodes[edge[1]]["age"])) / 2
        score = score + edge_score

    avg_score = score / graph.number_of_edges()

    return avg_score