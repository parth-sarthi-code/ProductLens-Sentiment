from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from productlens.engine import ProductLensEngine
from backend.api.v1.health import router as health_router
from backend.api.v1.analyze import router as analyze_router

_engine: ProductLensEngine | None = None


def get_engine() -> ProductLensEngine:
    global _engine
    if _engine is None:
        _engine = ProductLensEngine(lazy_load=True)
    return _engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize engine
    get_engine()
    yield
    # Shutdown


app = FastAPI(
    title="ProductLens Aspect-Based Sentiment Analysis Platform",
    description="High-performance ABSA API powered by a Hierarchical Mixture of Experts.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router, prefix="/api/v1", tags=["Health"])
app.include_router(analyze_router, prefix="/api/v1", tags=["Analysis"])


@app.get("/")
async def root():
    return {
        "service": "ProductLens Sentiment Analysis API",
        "version": "1.0.0",
        "docs_url": "/docs",
    }
