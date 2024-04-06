from typing import Union
from fastapi.middleware.cors import CORSMiddleware

from fastapi import FastAPI
import solver

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*']
)

@app.get("/run")
def get_all_maze_outputs():
    return solver.solver()



# @app.get("/items/{item_id}")
# def read_item(item_id: int, q: Union[str, None] = None):
#     return {"item_id": item_id, "q": q}