from fastapi import APIRouter

router = APIRouter()


@router.get("/nodes/", tags=["nodes"])
def get_nodes():
  return [{"id": 1}, {"id": 2}]


@router.get("/nodes/{id}", tags=["nodes"])
def get_node(id: int):
  return [{"id": 1}, {"id": 2}]


@router.put("/nodes/", tags=["nodes"])
def add_node():
  return {"message": "added"}


@router.post("/nodes/{id}", tags=["nodes"])
def update_node(id: int):
  return {"message": "updated"}


@router.delete("/nodes/{id}", tags=["nodes"])
def delete_node(id: int):
  return {"message": "deleted"}
