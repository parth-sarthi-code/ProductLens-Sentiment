from fastapi import APIRouter, HTTPException
from backend.schemas.requests import AnalyzeReviewRequest, BatchAnalyzeRequest
from productlens.schemas import ReviewAnalysisReport, BatchAnalysisReport

router = APIRouter()


@router.post("/analyze", response_model=ReviewAnalysisReport)
async def analyze_single_review(req: AnalyzeReviewRequest):
    from backend.main import get_engine
    try:
        engine = get_engine()
        report = engine.analyze_review(
            text=req.text,
            category=req.category,
            aspects=req.aspects,
        )
        return report
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze/batch", response_model=BatchAnalysisReport)
async def analyze_batch_reviews(req: BatchAnalyzeRequest):
    from backend.main import get_engine
    try:
        engine = get_engine()
        items = [{"text": r.text, "category": r.category, "aspects": r.aspects} for r in req.reviews]
        return engine.analyze_batch(items)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
