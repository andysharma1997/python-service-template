import time
import uvicorn

from fastapi import FastAPI, Path, Query, Body, status, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware

from .routers import models

app = FastAPI()

app.add_middleware(
    TrustedHostMiddleware, allowed_hosts=["*"]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

@app.get("/")
async def read_root():
    return {"Hello": "World"}

subapi = FastAPI()

subapi.include_router(models.router)

app.mount("/v1", subapi)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)