from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from ..db import get_db
from ..models import RetinaLabel
from ..schemas import CorrelationOut, BivariateOut
import pandas as pd

router = APIRouter(prefix="/stats", tags=["stats"])

@router.get("/correlation", response_model=CorrelationOut)
def correlation(cols: str, db: Session = Depends(get_db)):
    wanted = [c.strip() for c in cols.split(",") if c.strip()]
    cols_ok = [getattr(RetinaLabel, c) for c in wanted if hasattr(RetinaLabel, c)]
    if not cols_ok:
        raise HTTPException(400, "Provide valid ?cols=col1,col2")
    rows = db.query(*cols_ok).all()
    df = pd.DataFrame(rows, columns=wanted)
    if df.empty or len(df.columns) < 2:
        return {"matrix": {}}
    corr = df.corr(numeric_only=True).fillna(0.0)
    return {"matrix": corr.to_dict()}
    
@router.get("/bivariate", response_model=BivariateOut)
def bivariate(x: str, y: str, db: Session = Depends(get_db)):
    if not hasattr(RetinaLabel, x) or not hasattr(RetinaLabel, y):
        raise HTTPException(400, "Unknown column(s)")
    rows = db.query(getattr(RetinaLabel, x), getattr(RetinaLabel, y)).all()
    df = pd.DataFrame(rows, columns=["x", "y"])
    if df.empty:
        return {"series": []}
    out = df.groupby(["x", "y"]).size().reset_index(name="count").to_dict(orient="records")
    return {"series": out}
