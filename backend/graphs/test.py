from completion import complete_graph
from data import base_graph
from utils import draw_graph

# base_graph.add_edges_from([(9, 2)]) # 2022-06-28_16-12-03 hat probleme
base_graph.add_edges_from([(8, 9), (9, 2)]) # 2022-06-28_16-12-03 geht trotz (9, 2)
# base_graph.add_edges_from([(3, 8), (9, 2)]) # 2022-06-28_16-12-03 geht nicht
draw_graph(base_graph)

completed_graphs = complete_graph(base_graph, "2022-06-28_16-12-03", draw_generated_graphs="only_valid")

if completed_graphs:
  draw_graph(completed_graphs[1])