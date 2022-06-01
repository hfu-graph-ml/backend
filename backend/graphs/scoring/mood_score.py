import networkx as nx
import numpy as np

def calculate_mood_score(graph):
    score = 0

    for edge in graph.edges():
        edge_score = (2 - ((graph.nodes[edge[0]]["age"] - graph.nodes[edge[1]]["age"])**2 + (graph.nodes[edge[0]]["location"] - graph.nodes[edge[1]]["location"])**2)) / 2
        score = score + edge_score

    avg_score = score / graph.number_of_edges()

    return avg_score
