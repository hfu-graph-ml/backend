from completion import complete_graph
from data import base_graph
from utils import draw_graph

base_graph.add_edges_from([(0, 1), (1, 2), (0, 4), (4, 5), (1, 5), (3, 8), (8, 9), (9, 2)]) # 2022-06-28_16-12-03 hat probleme
# base_graph.add_edges_from([(3, 8), (8, 9), (9, 2), (5, 7)])
draw_graph(base_graph)

completed_graphs = complete_graph(base_graph, "2022-06-28_16-12-03", draw_generated_graphs="only_valid", n_samples=1000)

if completed_graphs:
  draw_graph(completed_graphs[1])