import networkx as nx
from data import all_nodes
from evaluation import calculate_mood_scores_from_graph

score_color_map = {
    0.0: "#f1464c",
    0.5: "#f1464c",
    1.0: "#ffb56b",
    1.5: "#ffb56b",
    2.0: "#d7dd3c",
    2.5: "#d7dd3c",
    3.0: "#60d394",
    3.5: "#60d394",
}


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


def draw_graph(graph, layout="spectral"):
  pos = None
  if layout == "spectral":
    pos = nx.spectral_layout(graph)
  nx.draw(graph, pos=pos, **get_draw_graph_options(edge_scores=calculate_mood_scores_from_graph(graph)))
