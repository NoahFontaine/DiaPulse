import streamlit as st
from Glucose_processor import GlucoseDataProcessor

st.set_page_config(layout="wide")

Glucose_data = GlucoseDataProcessor("CGM_data.csv", "Bolus_data.csv", "Carb_data.csv", min_range=68, max_range=180, target=100, tp=80, td=280, dp=45, dd=180)

st.plotly_chart(Glucose_data.plot_data(), use_container_width=True)

