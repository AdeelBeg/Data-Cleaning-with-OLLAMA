from __future__ import annotations
from pydantic import BaseModel, Field
from typing import List, Optional

class NormalizationItem(BaseModel):
    raw: str = Field(..., description="Observed raw categorical value")
    normalized: str = Field(..., description="Canonical normalized value (prefer from canonical options)")
    confidence: float = Field(..., ge=0.0, le=1.0)
    rationale: Optional[str] = Field(None, description="Short reason for mapping decision")

class NormalizationResponse(BaseModel):
    field: str
    mappings: List[NormalizationItem]
