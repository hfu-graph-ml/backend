import networkx as nx
import numpy as np

def calculate_mood_score(graph):
    score = 0

    for edge in graph.edges():
        edge_score = (graph.nodes[edge[0]]["age"] - graph.nodes[edge[1]]["age"])**2 + (graph.nodes[edge[0]]["name"] - graph.nodes[edge[1]]["name"])**2
        score = score + edge_score

    return score / graph.number_of_edges()

def valid_graph(graph):
    return graph.number_of_edges() == 28 and np.array_equal(nx.degree_histogram(graph), [0, 0, 4, 16])