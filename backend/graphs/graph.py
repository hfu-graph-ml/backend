from typing import Iterable, List, Tuple

from backend.graphs.node import Node


class ValueOrError:

    value = None
    error = ""

    def __init__(self, value=None, error: str = ""):
        self.value = value
        self.error = error

    def is_error(self) -> bool:
        return self.error and self.value == None


class Graph:

    nodes: List[Node]

    def __init__(self, nodes: List[Node] = None):
        self.nodes = nodes

    def __len__(self) -> int:
        """
        Returns the number of nodes in the graph. Use: 'len(G)'
        """
        return len(self.nodes)

    def add_node(self, node: Node):
        """
        add_node adds a new node to the graph
        """
        self.nodes.insert(node.id, node)

    def add_nodes_from(self, nodes: Iterable[Node]):
        """
        add_nodes_from adds multiple nodes from an iterable of nodes (e.g. a list)
        """
        for node in nodes:
            self.nodes.insert(node.id, node)

    def remove_node(self, id: int) -> Node:
        """
        remove_node removes a node with 'id' from the graph
        """
        return self.nodes.pop(id)

    def connect_nodes(self, nodes: Tuple[Node]) -> ValueOrError:
        if len(nodes) != 2:
            return ValueError(error="invalid len of tuple")

        return ValueOrError(value="success")
