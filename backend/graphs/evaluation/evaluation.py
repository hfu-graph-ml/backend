import networkx as nx
import numpy as np


def valid_table_graph(graph: nx.Graph):
    '''
    Checks wheter the graph has a valid seating plan graph structure.

    Args:
        graph: The graph to check.

    Returns:
        True if valid. False if invalid.
    '''
    return nx.is_isomorphic(graph, nx.grid_graph((2, graph.number_of_nodes()//2)))


def calculate_edge_score_matrix(nodes):
    '''
    Calculates the matrix containing the mood scores of every node pair. See `calculate_mood_score` method for how a mood score is calculated between two nodes.

    Args:
        nodes: The list of nodes with their features.

    Returns:
        The edge score matrix.
    '''
    edge_score_matrix = np.zeros((len(nodes), len(nodes)))

    for n in range(edge_score_matrix.shape[0]):
        for m in range(edge_score_matrix.shape[1]):
            edge_score_matrix[n][m] = calculate_mood_score(
                dict(
                    age=nodes[n][0],
                    country=nodes[n][1],
                    drinker=nodes[n][2],
                    relationship=nodes[n][3]
                ),
                dict(
                    age=nodes[m][0],
                    country=nodes[m][1],
                    drinker=nodes[m][2],
                    relationship=nodes[m][3]
                )
            )

    return edge_score_matrix


def calculate_mood_score(edge_1, edge_2):
    '''
    Calculates the mood score between two nodes according to it's features.

    The mood score is a sum of the individual scores for each feature:

    - Age score (see method `calculate_age_score`).
    - Country score (see method `edge_score_country`).
    - Drinker score (see method `edge_score_drinker`).
    - Relationship score (see method `edge_score_relationship`).

    Args:
        edge_1: Features of first node.
        edge_2: Features of second node.

    Returns:
        The mood score.
    '''
    edge_score_age = calculate_age_score(edge_1["age"], edge_2["age"])
    edge_score_country = calculate_country_score(edge_1["country"], edge_2["country"])
    edge_score_drinker = calculate_drinker_score(edge_1["drinker"], edge_2["drinker"])
    edge_score_relationship = calculate_relationship_score(edge_1["relationship"], edge_2["relationship"])

    return np.sum([edge_score_age, edge_score_country, edge_score_drinker, edge_score_relationship])


def calculate_age_score(age_1: int, age_2: int) -> float:
    '''
    Calculates the age score of the edge.

    The age score is calculated as followed:

    - if age group difference is 0, score is 1
    - if age group difference is 1, score is 0.5
    - if age group difference is 2 or more, score is 0

    Args:
        age_1 (int): Age group value of node 1.
        age_2 (int): Age group value of node 2.

    Returns:
        The age score.
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
    Calculates the country score of the edge.

    The country score is calculated as followed:

    - if country is the same, score is 1
    - if country is different but its language is the same, score is 0.5
    - if country and its language is different, score is 0

    Country codes and language:

    - 0: Germany (german)
    - 1: Swiss (german)
    - 2: Spain (spanish)
    - 3: Mexico (spanish)

    Args:
        country_1 (int): Country value of node 1.
        country_2 (int): Country value of node 2.

    Returns:
        The country score.
    '''

    if country_1 == country_2:
        return 1.0
    elif (country_1**2 + country_2**2) == (0**2 + 1**2) or (country_1**2 + country_2**2) == (2**2 + 3**2):
        return 0.5
    else:
        return 0.0


def calculate_drinker_score(drinker_1: int, drinker_2: int) -> float:
    '''
    Calculates the drinker score of the edge.

    The drinker score is calculated as followed:

    - if drinker value is the same, score is 1
    - else, score is 0

    Args:
        drinker_1 (int): drinker value of node 1.
        drinker_2 (int): drinker value of node 2.

    Returns:
        The drinker score.
    '''

    return 1.0 if drinker_1 == drinker_2 else 0.0


def calculate_relationship_score(relationship_1: int, relationship_2: int) -> float:
    '''
    Calculates the relationship score of the edge.

    The relationship score is calculated as followed:
    
    - if relationship value is the same, score is 0.5
    - else, score is 0

    Args:
        relationship_1 (int): relationship value of node 1.
        relationship_2 (int): relationship value of node 2.

    Returns:
        The relationship score.
    '''

    return 0.5 if relationship_1 == relationship_2 else 0.0
