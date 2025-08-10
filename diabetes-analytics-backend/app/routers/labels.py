from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from ..db import get_db
from ..models import RetinaLabel
from ..schemas import RetinaLabelOut

router = APIRouter(prefix="/labels", tags=["labels"])

@router.get("/", response_model=list[RetinaLabelOut])
def list_labels(limit: int = 50, offset: int = 0, db: Session = Depends(get_db)):
    return db.query(RetinaLabel).offset(offset).limit(limit).all()

@router.get("/{image_id}", response_model=RetinaLabelOut | None)
def get_label(image_id: str, db: Session = Depends(get_db)):
    return db.query(RetinaLabel).get(image_id)
