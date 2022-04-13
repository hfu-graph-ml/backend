from fastapi import FastAPI

from routers import nodes, edges, graphs

app = FastAPI()

app.include_router(graphs.router)
app.include_router(nodes.router)
app.include_router(edges.router)


@app.get("/")
async def root():
  return {"message": "Root"}
