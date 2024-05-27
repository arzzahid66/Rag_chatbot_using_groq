from routers import fast , test
from routers import test
from typing import Union
from fastapi import FastAPI , Response , status
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from fastapi.responses import Response
import uvicorn
from fastapi.params import Body
from typing import Optional
from random import randrange
# from simple_qa import*
from fastapi import FastAPI, HTTPException

app = FastAPI()


app.include_router(test.router,
                   prefix="/scap data",
                   tags=["click here to scrap data"])

app.include_router(fast.router,
                   prefix="/qa retrieval",
                   tags=["QA Retrieval"])