from fastapi import FastAPI

from routers import nodes, edges, graphs, complete_graph

app = FastAPI()

app.include_router(graphs.router)
app.include_router(nodes.router)
app.include_router(edges.router)
app.include_router(complete_graph.router)

@app.get("/")
async def root():
  return {"message": "Root"}
