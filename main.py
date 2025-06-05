import pandas as pd 
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import prepro
import ollama
import joblib 
from darts import TimeSeries
from darts.models import NBEATSModel
from darts.dataprocessing.transformers import Scaler
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from darts.utils.timeseries_generation import holidays_timeseries
from darts.metrics import mape
# Set page configuration
st.set_page_config(layout="wide", page_title="Dashboard Group 15", page_icon="📊")
# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f5;
    }
    .title {
        font-size: 2.5em;
        color: #1F2A44;
