from typing import List


class Node:

    id: int = 0
    # TODO (Techassi): How can we self reference here?
    # children: List[Node]

    def __init__(self, id: int):
        self.id = id
