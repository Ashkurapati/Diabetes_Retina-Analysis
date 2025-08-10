from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from ..db import get_db
from ..models import RetinaLabel
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
import numpy as np

router = APIRouter(prefix="/ml", tags=["ml"])

@router.get("/feature-importance")
def feature_importance(target: str = "DR_ICDR", db: Session = Depends(get_db)):
    rows = db.query(RetinaLabel).all()
    if not rows:
        return {"importances": []}
    df = pd.DataFrame([r.__dict__ for r in rows]).drop(columns=["_sa_instance_state"], errors="ignore")
    if target not in df.columns:
        raise HTTPException(400, f"Unknown target {target}")

    X = df.drop(columns=[target])
    y = df[target]

    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    num_imputer = SimpleImputer(strategy="median")
    X_num = num_imputer.fit_transform(X[num_cols]) if num_cols else np.empty((len(X),0))

    if cat_cols:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X_cat = enc.fit_transform(X[cat_cols].astype(str))
        X_proc = np.hstack([X_num, X_cat])
        feat_names = num_cols + cat_cols
    else:
        X_proc = X_num
        feat_names = num_cols

    if len(pd.Series(y).dropna().unique()) < 2:
        raise HTTPException(400, "Target has <2 classes after cleaning")

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_proc, y)
    imps = sorted(
        [{"feature": f, "importance": float(v)} for f, v in zip(feat_names, model.feature_importances_)],
        key=lambda d: d["importance"],
        reverse=True
    )
    return {"importances": imps[:30]}
