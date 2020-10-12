from typing import Optional

from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from pydantic import BaseModel

import generate

class Seed(BaseModel):
    payload: Optional[str] = ""
    temp: Optional[float] = 0.5

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/play_trumpet")
async def play_trumpet(seed: Seed):
    return {"text": generate.generate(seed.payload, seed.temp)}
