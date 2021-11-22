from fastapi import FastAPI

from backend.graphs.graph import Graph
from backend.graphs.node import Node

from .routers import nodes, edges

app = FastAPI()

app.include_router(nodes.router)
app.include_router(edges.router)


@app.get("/")
async def root():
    g = Graph()
    valOrErr = g.connect_nodes((Node(0), Node(1)))
    if valOrErr.is_error():
        print(valOrErr.error)
    else:
        print(valOrErr.value)

    return {"message": "Root"}
