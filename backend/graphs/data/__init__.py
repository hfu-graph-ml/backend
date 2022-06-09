import random
import networkx as nx
import matplotlib.pyplot as plt

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
all_nodes = {
    0: ["child", "germany", "sober", "family_1"], # mira
    1: ["adult", "germany", "drinking", "family_1"], # stefan
    2: ["adult", "germany", "sober", "family_1"], # katrin
    3: ["young_adult", "mexico", "drinking", "friends_1"], # jorge
    4: ["young_adult", "mexico", "drinking", "friends_1"], # juan
    5: ["old_adult", "swiss", "sober", "no_relationship"], # liselotte
}

# draw people from all
base_graph = nx.Graph()
for sample in random.sample(list(all_nodes.items()), k=config['data']['num_nodes']):
  base_graph.add_node(
      sample[0],
      age=feature_list_age[sample[1][0]],
      country=feature_list_country[sample[1][1]],
      drinker=feature_list_drinker[sample[1][2]],
      relationship=feature_list_relationship[sample[1][3]] if feature_list_relationship[sample[1][3]] != None else len(feature_list_relationship) + sample[0]
  )

# print(base_graph.nodes(data=True))
# nx.draw(base_graph, with_labels=True)