import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from data import all_nodes
from data.edge_score_matrix import get_edge_scores
from utils.colors import score_color_map

def get_draw_graph_options(edge_scores=None):
  draw_graph_options = dict(
      labels=dict(zip(list(zip(*all_nodes))[0], list(zip(*all_nodes))[1])),
      node_color="#efefef",
      node_size=1000,
      font_weight="bold",
      width=5,
      with_labels=True,
  )

  if edge_scores:
    draw_graph_options["edge_color"] = [score_color_map[score] for score in edge_scores]

  return draw_graph_options


def draw_graph(graph, layout="spectral", show_graph=True):
  pos = None
  if layout == "spectral":
    pos = nx.spectral_layout(graph)

  mood_scores = get_edge_scores(graph.edges())
  nx.draw(graph, pos=pos, **get_draw_graph_options(edge_scores=mood_scores))
  fig = plt.gcf()
  fig.suptitle(f"Mood Score: {np.sum(mood_scores):.1f}", fontsize=12)

  if show_graph:
    plt.show()
