from fastapi import FastAPI

from routers import complete_graph

app = FastAPI()

app.include_router(complete_graph.router)

@app.get("/")
async def root():
  return {"message": "Root"}
