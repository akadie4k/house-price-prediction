"""Streamlit entry point: streamlit run app.py."""
from pathlib import Path
import json
import pandas as pd
import streamlit as st
from inference import load_artifacts, predict_record, ZONES, TYPES

st.set_page_config(page_title="House Price Prediction",page_icon="🏠",layout="centered")
st.title("House Price Prediction")
st.write("Enter property details, then select Predict price. The estimate is shown in Malaysian ringgit (RM).")
st.caption("Educational model trained on synthetic properties. Estimates are not verified market valuations.")

@st.cache_resource
def artifacts():
    return load_artifacts()

try:
    pipeline,metadata=artifacts()
except Exception:
    st.error("The saved model could not be loaded. Restore the models folder and use the documented dependency versions.")
    st.stop()

with st.form("property_form"):
    left,right=st.columns(2)
    with left:
        area=st.number_input("Floor area (m²)",min_value=0.,max_value=1000.,value=200.,step=0.1)
        area_missing=st.checkbox("Floor area is unknown",value=False)
        bedrooms=st.number_input("Bedrooms",min_value=1,max_value=15,value=3,step=1)
        bedrooms_missing=st.checkbox("Bedrooms are unknown",value=False)
        bathrooms=st.number_input("Bathrooms",min_value=1,max_value=10,value=2,step=1)
    with right:
        age=st.number_input("Property age (years)",min_value=0,max_value=100,value=10,step=1)
        zone=st.selectbox("Location zone",ZONES,index=1)
        kind=st.selectbox("Property type",TYPES,index=3)
        parking=st.number_input("Parking spaces",min_value=0,max_value=10,value=2,step=1)
    submitted=st.form_submit_button("Predict price",type="primary")

if submitted:
    record={"floor_area_m2":None if area_missing else area,"bedrooms":None if bedrooms_missing else bedrooms,
            "bathrooms":bathrooms,"property_age_years":age,"location_zone":zone,"property_type":kind,"parking_spaces":parking}
    try:
        price,warnings=predict_record(pipeline,metadata,record,log=True)
        for warning in warnings:st.warning(warning)
        st.metric("Estimated house price",f"RM {price:,.2f}")
        st.caption("This is a point estimate. The test MAE is an average error, not a confidence interval for this property.")
    except ValueError as exc:
        st.error(str(exc))

with st.expander("Model and evaluation"):
    m=metadata["test_metrics"]
    st.write(f"Selected model: {metadata['selected_model']}")
    st.write(f"Held out test R²: {m['R2']:.4f}; MAE: RM {m['MAE_RM']:,.2f}; RMSE: RM {m['RMSE_RM']:,.2f}.")
    st.write("Split: 70% train, 15% validation, 15% test. Missing area and bedrooms use training medians.")

with st.expander("Monitoring"):
    st.write("Only non-sensitive property inputs, predictions and validation outcomes are logged. No names, addresses or student IDs are collected.")
    st.write("Proposed review trigger: more than 10% of the latest 100 accepted requests have missing or out-of-range inputs. This is an operational signal, not an accuracy estimate.")
    import os
    log_path=Path(os.environ.get("HOUSEPRICE_LOG_PATH",str(Path(__file__).resolve().parent/"logs/predictions.csv")))
    if log_path.exists():
        logs=pd.read_csv(log_path)
        st.dataframe(logs.tail(10),hide_index=True)
        st.download_button("Download monitoring records",logs.to_csv(index=False),file_name="monitoring_records.csv",mime="text/csv")
    else:
        st.write("No prediction events have been recorded in this app instance.")
    st.caption("Local records may be lost when hosting restarts. Export them during the demonstration; use persistent storage for longer deployments.")
