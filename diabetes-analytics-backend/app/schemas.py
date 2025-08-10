from pydantic import BaseModel
from typing import Dict, List

class RetinaLabelOut(BaseModel):
    image_id: str
    patient_id: int | None = None
    patient_age: int | None = None
    diabetes_time_y: float | None = None
    DR_ICDR: int | None = None
    class Config:
        from_attributes = True

class CorrelationOut(BaseModel):
    matrix: Dict[str, Dict[str, float]]

class BivariateOut(BaseModel):
    series: List[dict]
