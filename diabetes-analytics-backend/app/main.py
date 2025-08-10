from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routers import labels, stats, ml

app = FastAPI(title="Diabetes Analytics API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok"}

app.include_router(labels.router)
app.include_router(stats.router)
app.include_router(ml.router)

from fastapi.responses import RedirectResponse

@app.get("/")
def root():
    return RedirectResponse(url="/docs")
