import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import plotly.graph_objects as go

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('sensor_data.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

# Train model
@st.cache_resource
def train_model(df):
    features = ['temperature', 'vibration', 'pressure']
    model = IsolationForest(contamination=0.05, random_state=42)
    model.fit(df[features])
    return model

# UI
st.title("🏭 IoT Anomaly Detection Dashboard")
st.write("Real-time sensor monitoring with ML-powered anomaly detection")

df = load_data()
model = train_model(df)

# Predict anomalies
features = ['temperature', 'vibration', 'pressure']
df['anomaly'] = model.predict(df[features])
df['anomaly'] = df['anomaly'].map({1: 'Normal', -1: 'Anomaly'})

# Stats
col1, col2, col3 = st.columns(3)
col1.metric("Total Readings", len(df))
col2.metric("Anomalies Found", len(df[df['anomaly'] == 'Anomaly']))
col3.metric("System Status", 
    "⚠️ Alert" if len(df[df['anomaly'] == 'Anomaly']) > 0 else "✅ Normal")

st.divider()

# Chart
sensor = st.selectbox("Select Sensor", ['temperature', 'vibration', 'pressure'])

normal = df[df['anomaly'] == 'Normal']
anomaly = df[df['anomaly'] == 'Anomaly']

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=normal['timestamp'], y=normal[sensor],
    mode='lines', name='Normal',
    line=dict(color='#2563eb')
))
fig.add_trace(go.Scatter(
    x=anomaly['timestamp'], y=anomaly[sensor],
    mode='markers', name='Anomaly',
    marker=dict(color='red', size=10, symbol='x')
))
fig.update_layout(
    title=f'{sensor.capitalize()} Readings Over Time',
    xaxis_title='Time',
    yaxis_title=sensor.capitalize()
)
st.plotly_chart(fig, use_container_width=True)

# Anomaly table
st.subheader("⚠️ Anomaly Records")
st.dataframe(
    df[df['anomaly'] == 'Anomaly'][['timestamp', 'temperature', 'vibration', 'pressure']],
    use_container_width=True
)