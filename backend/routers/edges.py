from fastapi import APIRouter

router = APIRouter()


@router.get("/edges/", tags=["edges"])
def get_edges():
  return [{"id": 1}, {"id": 2}]


@router.get("/edges/{id}", tags=["edges"])
def get_edge(id: int):
  return [{"id": 1}, {"id": 2}]


@router.put("/edges/", tags=["edges"])
def add_edge():
  return {"message": "added"}


@router.post("/edges/{id}", tags=["edges"])
def update_edge(id: int):
  return {"message": "updated"}


@router.delete("/edges/{id}", tags=["edges"])
def delete_edge(id: int):
  return {"message": "deleted"}
