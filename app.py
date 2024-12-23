import streamlit as st
from CGM_processor import CGMDataProcessor

st.set_page_config(layout="wide")

CGM_data = CGMDataProcessor("CGM_data.csv", min_range=68, max_range=180, target=100)

st.plotly_chart(CGM_data.plot_data(), use_container_width=True)

