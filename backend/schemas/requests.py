from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class AnalyzeReviewRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Customer review text to analyze")
    category: Optional[str] = Field(None, description="Optional e-commerce category (e.g. 'Electronics')")
    aspects: Optional[List[str]] = Field(None, description="Optional explicit aspects list to analyze")


class BatchAnalyzeRequest(BaseModel):
    reviews: List[AnalyzeReviewRequest] = Field(..., min_length=1, description="Batch of customer reviews")


class HealthResponse(BaseModel):
    status: str
    version: str
    device: str
    loaded_experts: List[str]
    available_experts: List[str]
