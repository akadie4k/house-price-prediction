"""Shared validation, inference and non-sensitive monitoring for the dashboard."""
from pathlib import Path
from datetime import datetime, timezone
import csv
import json
import logging
import math
import os

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
FEATURES = ["floor_area_m2", "bedrooms", "bathrooms", "property_age_years", "location_zone", "property_type", "parking_spaces"]
ZONES = ["Rural", "Suburban", "Town_Periphery", "Urban_Centre"]
TYPES = ["Apartment", "Detached", "Semi_Detached", "Terrace"]
INTEGER_COLUMNS = ["bedrooms", "bathrooms", "property_age_years", "parking_spaces"]
OPTIONAL = {"floor_area_m2", "bedrooms"}

def load_artifacts(root=ROOT):
    root=Path(root)
    pipeline=joblib.load(root/"models/house_price_pipeline.joblib")
    metadata=json.loads((root/"models/metadata.json").read_text())
    if metadata["feature_order"] != FEATURES:
        raise ValueError("Saved feature order does not match the application.")
    return pipeline,metadata

def validate_record(record, metadata):
    if set(record) != set(FEATURES):
        raise ValueError("Provide exactly the seven expected property features.")
    cleaned={}; warnings=[]
    for col in FEATURES:
        value=record[col]
        if col in ("location_zone","property_type"):
            choices=ZONES if col=="location_zone" else TYPES
            if value not in choices:
                raise ValueError(f"Unsupported {col}; choose a listed category.")
            cleaned[col]=value
            continue
        if value is None or (isinstance(value,(int,float,np.number)) and np.isnan(value)):
            if col not in OPTIONAL:
                raise ValueError(f"{col} is required.")
            cleaned[col]=np.nan;warnings.append(f"{col} is missing; the training median will be used.")
            continue
        if isinstance(value,(bool,str)):
            raise ValueError(f"{col} must be numeric.")
        try:
            value=float(value)
        except (TypeError,ValueError):
            raise ValueError(f"{col} must be numeric.") from None
        if not math.isfinite(value):
            raise ValueError(f"{col} must be finite.")
        if col in INTEGER_COLUMNS and not value.is_integer():
            raise ValueError(f"{col} must be a whole number.")
        if col in ("floor_area_m2","bedrooms","bathrooms") and value<=0:
            raise ValueError(f"{col} must be greater than zero.")
        if col in ("property_age_years","parking_spaces") and value<0:
            raise ValueError(f"{col} cannot be negative.")
        bounds=metadata["training_input_ranges"][col]
        if value<bounds[0] or value>bounds[1]:
            warnings.append(f"{col} is outside the training range {bounds[0]:g} to {bounds[1]:g}; reliability is unverified.")
        cleaned[col]=value
    return pd.DataFrame([cleaned],columns=FEATURES),warnings

def record_event(metadata,status,frame=None,prediction=None,warnings=None):
    event={"timestamp_utc":datetime.now(timezone.utc).isoformat(),"model":metadata["selected_model"],"status":status,
           "prediction_rm":None if prediction is None else round(float(prediction),2),"warning_count":len(warnings or [])}
    if frame is not None:
        event.update({c:None if pd.isna(frame.iloc[0][c]) else frame.iloc[0][c] for c in FEATURES})
    # JSON events are visible in Streamlit hosting logs; no personal identifiers.
    logging.getLogger("houseprice").warning(json.dumps(event,default=float))
    path=Path(os.environ.get("HOUSEPRICE_LOG_PATH",str(ROOT/"logs/predictions.csv")))
    path.parent.mkdir(parents=True,exist_ok=True)
    fields=["timestamp_utc","model","status","prediction_rm","warning_count"]+FEATURES
    exists=path.exists()
    try:
        with path.open("a",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=fields)
            if not exists:writer.writeheader()
            writer.writerow(event)
    except OSError:
        logging.getLogger("houseprice").warning("Local CSV unavailable; retain the console event instead.")

def predict_record(pipeline,metadata,record,log=False):
    try:
        frame,warnings=validate_record(record,metadata)
    except ValueError:
        if log:record_event(metadata,"rejected")
        raise
    pred=float(pipeline.predict(frame)[0])
    if not np.isfinite(pred) or pred<=0:
        raise ValueError("The model did not return a valid positive price.")
    if log:record_event(metadata,"accepted",frame,pred,warnings)
    return pred,warnings
