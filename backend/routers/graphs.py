from fastapi import APIRouter

router = APIRouter()

graph = {
    "nodes": [
        {
            "id": 0,
            "name": "Peter",
        },
        {
            "id": 1,
            "name": "Paul",
        },
        {
            "id": 2,
            "name": "Anna",
        },
        {
            "id": 3,
            "name": "Lucy",
        },
        {
            "id": 4,
            "name": "Max",
        },
        {
            "id": 5,
            "name": "Lena",
        }
    ],
    "edges": [
        {
            "connects": [0, 1],
            "score": 0.5,
        },
        {
            "connects": [0, 2],
            "score": 0.23,
        },
        {
            "connects": [2, 5],
            "score": 0.8,
        },
        {
            "connects": [1, 5],
            "score": 0.65,
        },
        {
            "connects": [1, 3],
            "score": 0.1,
        },
        {
            "connects": [5, 4],
            "score": 0.73,
        },
        {
            "connects": [3, 4],
            "score": 0.9
        }
    ]
}

response = {
    "status": "success",
    "graph": graph,
}


@router.get("/graphs/", tags=["graphs"])
def get_graphs():
  '''
  This currently only returns a hardcoded graph for testing purposes
  '''
  return response
