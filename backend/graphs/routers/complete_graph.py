from fastapi import APIRouter

from data import base_graph
from completion import complete_graph
from utils import draw_graph

router = APIRouter()

@router.get("/complete-graph/{edge_string}", tags=["complete_graph"])
def get_completed_graph(edge_string: str):
  '''
  Completes the graph with nodes:

  - 0 - Michel
  - 1 - Elias
  - 2 - Sofia
  - 3 - Jorge
  - 4 - Juan
  - 5 - Manuel
  - 6 - Laura
  - 7 - Stefan
  - 8 - Robin
  - 9 - Daniel

  and edges given via edge_string parameter.

  The edge_string parameter should come in the format used in the following examples:

  - `/complete-graph/0-1_1-2_8-9` (translates into array of 3 edges `[(0, 1), (1, 2), (8, 9)]`)
  - `/complete-graph/-` (translates into empty array of 0 edges `[]`)
  
  '''

  if edge_string == "-":
    edges = []
  else:
    edges = [edge for edge in edge_string.split("_")]
    edges = [(int(edge.split("-")[0]), int(edge.split("-")[1])) for edge in edges]

  # use nodes from base_graph
  input_graph = base_graph
  # add given edges to achieve graph from interface
  input_graph.add_edges_from(edges)

  # generate completed graph, sorted by descending mood scores
  completed_graphs = complete_graph(input_graph, "2022-06-28_09-50-50", run_for_min_seconds=5, run_for_max_seconds=120)

  if completed_graphs:
    best_graph = completed_graphs[1]
    mood_score = completed_graphs[0]
    response = {
      "status": "success",
      "completed_graph_edge_list": list(best_graph.edges),
      "completed_graph_mood_score": mood_score,
    }
    draw_graph(best_graph)
  else:
    response = {
      "status": "failure",
      "completed_graph_edge_list": [],
      "completed_graph_mood_score": 0,
    }
  
  return response
