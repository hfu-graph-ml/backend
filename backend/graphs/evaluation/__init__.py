import networkx as nx
import numpy as np
from networkx.algorithms import bipartite


def valid_table_graph(graph, num_nodes, max_edges):
  correct_number_of_edges = graph.number_of_edges() == max_edges
  correct_degree_histogram = np.array_equal(nx.degree_histogram(graph), [0, 0, 4, num_nodes - 4])
  graph_is_bipartite = bipartite.is_bipartite(graph)

  return correct_number_of_edges and correct_degree_histogram and graph_is_bipartite

highest_possible_edge_mood_score = 3.5

def calculate_mood_scores_from_graph(graph):
  edge_scores = []

  for edge in graph.edges():
    edge_scores.append(calculate_mood_score(graph.nodes[edge[0]], graph.nodes[edge[1]]))

  return edge_scores

def calculate_mood_score(edge_1, edge_2):
  edge_score_age = calculate_age_score(edge_1["age"], edge_2["age"])
  edge_score_country = calculate_country_score(edge_1["country"], edge_2["country"])
  edge_score_drinker = calculate_drinker_score(edge_1["drinker"], edge_2["drinker"])
  edge_score_relationship = calculate_relationship_score(edge_1["relationship"], edge_2["relationship"])

  return np.sum([edge_score_age, edge_score_country, edge_score_drinker, edge_score_relationship])


def calculate_age_score(age_1: int, age_2: int) -> float:
  '''
  Returns the age score of the edge.

  The age score is calculated as followed:
  - if age group difference is 0, score is 1
  - if age group difference is 1, score is 0.5
  - if age group difference is 2 or more, score is 0

    Parameters:
      age_1 (int): Age group value of node 1.
      age_2 (int): Age group value of node 2.

    Returns:
      float: The age score.
  '''

  difference = abs(age_1 - age_2)
  if difference == 0:
    return 1.0
  elif difference == 1:
    return 0.5
  else:
    return 0.0


def calculate_country_score(country_1: int, country_2: int) -> float:
  '''
  Returns the country score of the edge.

  The country score is calculated as followed:
  - if country is the same, score is 1
  - if country is different but its language is the same, score is 0.5
  - if country and its language is different, score is 0

  Country codes and language:
  - 0: Germany (german)
  - 1: Swiss (german)
  - 2: Spain (spanish)
  - 3: Mexico (spanish)

    Parameters:
      country_1 (int): Country value of node 1.
      country_2 (int): Country value of node 2.

    Returns:
      float: The country score.
  '''

  if country_1 == country_2:
    return 1.0
  elif (country_1**2 + country_2**2) == (0**2 + 1**2) or (country_1**2 + country_2**2) == (2**2 + 3**2):
    return 0.5
  else:
    return 0.0


def calculate_drinker_score(drinker_1: int, drinker_2: int) -> float:
  '''
  Returns the drinker score of the edge.

  The drinker score is calculated as followed:
  - if drinker value is the same, score is 1
  - else, score is 0

    Parameters:
      drinker_1 (int): drinker value of node 1.
      drinker_2 (int): drinker value of node 2.

    Returns:
      float: The drinker score.
  '''

  return 1.0 if drinker_1 == drinker_2 else 0.0


def calculate_relationship_score(relationship_1: int, relationship_2: int) -> float:
  '''
  Returns the relationship score of the edge.

  The relationship score is calculated as followed:
  - if relationship value is the same, score is 0.5
  - TODO: if relationship value is a 'single person' value and they 'match' (other scores sum to 3.0), score is 1.0
  - else, score is 0

    Parameters:
      relationship_1 (int): relationship value of node 1.
      relationship_2 (int): relationship value of node 2.

    Returns:
      float: The relationship score.
  '''

  return 0.5 if relationship_1 == relationship_2 else 0.0