import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd
import numpy as np

from evaluation import calculate_edge_score_matrix
from utils.colors import score_color_map
from data import all_nodes_features, all_nodes

save_edge_score_matrix = False

# generate edge score matrix
edge_score_matrix = calculate_edge_score_matrix(all_nodes_features)
name_list = list(zip(*all_nodes))[1]

if save_edge_score_matrix:
  plt.matshow(edge_score_matrix, cmap=ListedColormap([color for color in score_color_map.values()]))
  ax = plt.gca()
  ax.set_xticks(range(len(name_list)), labels=name_list)
  ax.set_yticks(range(len(name_list)), labels=name_list)
  plt.xticks(rotation=45)
  for (i, j), z in np.ndenumerate(edge_score_matrix):
    ax.text(j, i, '{:0.1f}'.format(z), ha='center', va='center')
  fig = plt.gcf()
  fig.savefig('backend/graphs/data/edge_score_matrix.png', dpi=480)
  plt.close()

  df = pd.DataFrame()
  df[''] = name_list
  for edge_score in enumerate(edge_score_matrix):
    df[name_list[edge_score[0]]] = edge_score[1]
  df.to_excel('backend/graphs/data/edge_score_matrix.xlsx', index=False)


def get_edge_scores(edges):
  edge_scores = []

  for edge in edges:
    edge_scores.append(edge_score_matrix[edge[0]][edge[1]])

  return edge_scores
