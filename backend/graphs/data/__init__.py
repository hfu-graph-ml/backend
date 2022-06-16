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
    "switzerland": 1,
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

if config['data']['num_nodes'] == 20:
    feature_list_relationship = {
        "family_1": 0,
        "family_2": 1,
        "friends_1": 2,
        "friends_2": 3,
        "friends_3": 4,
        "friends_4": 5,
        "no_relationship": None,
    }

# people to draw from
all_nodes = [
    [0, "Mira", "child", "germany", "sober", "family_1"],
    [1, "Stefan", "adult", "germany", "drinking", "family_1"],
    [2, "Katrin", "adult", "germany", "sober", "family_1"],
    [3, "Jorge", "young_adult", "mexico", "drinking", "friends_1"],
    [4, "Juan", "young_adult", "mexico", "drinking", "friends_1"],
    [5, "Liselotte", "old_adult", "switzerland", "sober", "no_relationship"],
    [6, "Manuel", "young_adult", "germany", "drinking", "no_relationship"],
    [7, "Laura", "young_adult", "germany", "drinking", "no_relationship"],
    [8, "Blanca", "old_adult", "spain", "drinking", "friends_2"],
    [9, "Luís", "adult", "mexico", "sober", "friends_2"],
    [10, "Robin", "adult", "germany", "drinking", "friends_3"],
    [11, "Daniel", "adult", "germany", "drinking", "friends_3"],
    [12, "Michel", "child", "switzerland", "sober", "family_2"],
    [13, "Elias", "child", "switzerland", "sober", "family_2"],
    [14, "Noah", "young_adult", "switzerland", "drinking", "family_2"],
    [15, "Sofia", "young_adult", "switzerland", "sober", "family_2"],
    [16, "Waldemar", "old_adult", "switzerland", "drinking", "family_2"],
    [17, "Valeria", "adult", "spain", "sober", "friends_4"],
    [18, "Carmen", "adult", "spain", "drinking", "friends_4"],
    [19, "Diego", "old_adult", "spain", "drinking", "no_relationship"],
]


if config['data']['num_nodes'] == 6:
    all_nodes = all_nodes[0:6]

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
for sample in random.sample(all_nodes, k=config['data']['num_nodes']):
  node_id = sample[0]
  base_graph.add_node(
      node_id,
      age=all_nodes_features[node_id][0],
      country=all_nodes_features[node_id][1],
      drinker=all_nodes_features[node_id][2],
      relationship=all_nodes_features[node_id][3]
  )

# print(base_graph.nodes(data=True))
