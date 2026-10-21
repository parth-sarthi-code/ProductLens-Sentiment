from fastapi import APIRouter
from backend.schemas.requests import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    from backend.main import get_engine
    engine = get_engine()
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        device=str(engine.pipeline.device),
        loaded_experts=list(engine.pipeline.experts.keys()),
        available_experts=list(engine.pipeline.checkpoint_paths.keys()),
    )
