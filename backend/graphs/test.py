from completion import complete_graph
from data import base_graph
from utils import draw_graph

# base_graph.add_edge(0, 3)
# base_graph.add_edge(1, 3)
# base_graph.add_edge(3, 4) # 2022-06-09_14-21-02
# draw_graph(base_graph, layout=None)

completed_graph = complete_graph(base_graph, "2022-06-09_14-21-02", draw_generated_graphs="only_valid")

if not completed_graph:
  print("Graph generation failed!")
  exit()

print("Graph completed!")
draw_graph(completed_graph[1])