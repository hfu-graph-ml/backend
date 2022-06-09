import random
import networkx as nx

from config import config

random.seed(0)

# feature 1: age
feature_list_age = {
    "child": 0,
    "young_adult": 1,
    "adult": 2,
    "old_adult": 3,
}

# feature 2: country
feature_list_country = {
    "germany": 0,
    "swiss": 1,
    "spain": 2,
    "mexico": 3,
}

# feature 3: drinker
feature_list_drinker = {
    "drinking": 0,
    "sober": 1,
}

# feature 4: relationship
feature_list_relationship = {
    "family_1": 0,
    "friends_1": 1,
    "no_relationship": None,
}

# people to draw from
all_nodes_20 = [
    [0, "Mira", "child", "germany", "sober", "family_1"],
    [1, "Stefan", "adult", "germany", "drinking", "family_1"],
    [2, "Katrin", "adult", "germany", "sober", "family_1"],
    [3, "Jorge", "young_adult", "mexico", "drinking", "friends_1"],
    [4, "Juan", "young_adult", "mexico", "drinking", "friends_1"],
    [5, "Liselotte", "old_adult", "swiss", "sober", "no_relationship"],
    # WIP
]

all_nodes = [
    [0, "Mira", "child", "germany", "sober", "family_1"],
    [1, "Stefan", "adult", "germany", "drinking", "family_1"],
    [2, "Katrin", "adult", "germany", "sober", "family_1"],
    [3, "Jorge", "young_adult", "mexico", "drinking", "friends_1"],
    [4, "Juan", "young_adult", "mexico", "drinking", "friends_1"],
    [5, "Liselotte", "old_adult", "swiss", "sober", "no_relationship"],
]

all_nodes_features = []
for node in all_nodes:
  all_nodes_features.append([
      feature_list_age[node[2]],
      feature_list_country[node[3]],
      feature_list_drinker[node[4]],
      feature_list_relationship[node[5]] if feature_list_relationship[node[5]
                                                                      ] != None else len(feature_list_relationship) + node[0]
  ])

# draw people from all
base_graph = nx.Graph()
for sample in random.sample(list(enumerate(all_nodes_features)), k=config['data']['num_nodes']):
  base_graph.add_node(
      sample[0],
      age=sample[1][0],
      country=sample[1][1],
      drinker=sample[1][2],
      relationship=sample[1][3]
  )

# print(base_graph.nodes(data=True))