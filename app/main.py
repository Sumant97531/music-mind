#%% 
"""
app/main.py
FastAPI application entry point.

Start the API:
    uvicorn app.main:app --reload

Docs at:  http://localhost:8000/docs
"""
from fastapi import FastAPI
from app.routes.recommend import router

app = FastAPI(
    title       = "Music Mind API",
    description = "Hybrid music recommendation: content + collaborative + mood",
    version     = "1.0.0",
)

app.include_router(router, prefix="/api/v1")


@app.get("/health")
def health():
    return {"status": "ok"}