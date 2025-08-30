import streamlit as st
import pandas as pd
import numpy as np
from data_preprocessing import preprocess_data
from models import train_isolation_forest, train_autoencoder, predict_anomalies
from utils import plot_data

st.set_page_config(layout="wide")

st.title("AI-Based Structural Health Monitoring System")

st.sidebar.title("Controls")
uploaded_file = st.sidebar.file_uploader("Upload sensor data (CSV)", type="csv")
model_option = st.sidebar.selectbox("Select Anomaly Detection Model", ["Isolation Forest", "Autoencoder (LSTM)"])
sensitivity = st.sidebar.slider("Anomaly Sensitivity", 0.1, 1.0, 0.5, 0.05)

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("### Raw Sensor Data")
    st.write(data.head())

    features, preprocessed_data = preprocess_data(data)

    st.write("### Preprocessed Data")
    st.write(preprocessed_data.head())

    if model_option == "Isolation Forest":
        model = train_isolation_forest(preprocessed_data)
        anomalies = predict_anomalies(model, preprocessed_data, contamination=sensitivity)
    else:
        model = train_autoencoder(preprocessed_data)
        anomalies = predict_anomalies(model, preprocessed_data, threshold=sensitivity)

    st.write("### Anomaly Detection Results")
    results = preprocessed_data.copy()
    results['anomaly'] = anomalies

    st.write(results.head())

    st.write("### Recommendations for Interpreting Results")
    st.info("""
    - An anomaly indicates a data point that significantly deviates from the normal pattern of sensor readings.
    - Detected anomalies may correspond to potential structural issues, sensor malfunctions, or unusual events.
    - Review the time and sensor values of anomalies to assess their significance.
    - Not all anomalies are critical; further engineering analysis is recommended for flagged points.
    - Adjust the sensitivity slider to tune the strictness of anomaly detection.
    """)
    st.write("### Visualization")
    fig = plot_data(results, features, anomalies)
    st.plotly_chart(fig)

    st.write("### Alerts")
    anomaly_df = results[results['anomaly'] == -1]
    if not anomaly_df.empty:
        st.warning("Anomalies detected! Please check the highlighted points in the visualization.")
        st.write(f"Number of anomalies detected: {len(anomaly_df)}")
        st.write("#### Anomaly Details:")
        st.dataframe(anomaly_df)

        csv = anomaly_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Anomaly Report (CSV)",
            data=csv,
            file_name='anomaly_report.csv',
            mime='text/csv',
        )
    else:
        st.success("No anomalies detected.")

else:
    st.info("Please upload a CSV file to begin.")
