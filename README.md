# Structural Health Monitoring

This project provides tools for anomaly detection in structural health monitoring data using machine learning models. It features a Streamlit web app for interactive data analysis and visualization.

## Features
- Data preprocessing utilities
- Anomaly detection using Isolation Forest and LSTM Autoencoder
- Interactive visualization with Streamlit
- Support for CSV data input

## Project Structure
- `app.py` — Main Streamlit application
- `data_preprocessing.py` — Data cleaning and preprocessing functions
- `models.py` — Machine learning models for anomaly detection
- `utils.py` — Utility functions
- `sample_data.csv` — Example dataset
- `requirements.txt` — Python dependencies

## Getting Started

### 1. Clone the repository
```bash
cd structural_health_monitoring
```

### 2. Set up a virtual environment (recommended)
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit app
```bash
streamlit run app.py
```

## Usage
- Upload your structural health monitoring data in CSV format.
- Select the anomaly detection model (Isolation Forest or LSTM Autoencoder).
- Adjust model parameters as needed.
- View detected anomalies and visualizations.

## Requirements
See `requirements.txt` for exact package versions.
