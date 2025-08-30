import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import MinMaxScaler

def noise_reduction(data):
    """
    Apply a basic low-pass Butterworth filter to reduce noise in a 1D signal.
    Args:
        data (array-like): Input signal data.
    Returns:
        np.ndarray: Filtered signal.
    """
    b, a = butter(3, 0.05, btype='low', analog=False)
    return filtfilt(b, a, data)

def preprocess_data(data):
    """
    Preprocess the input DataFrame by applying noise reduction, normalization, and feature extraction.
    Args:
        data (pd.DataFrame): Raw sensor data. Assumes 'time' column if present.
    Returns:
        tuple: (list of feature names, preprocessed DataFrame)
    """
    if 'time' in data.columns:
        features = data.columns.drop('time')
    else:
        features = data.columns
    
    preprocessed_data = data.copy()

    for feature in features:
        # Noise reduction
        preprocessed_data[feature] = noise_reduction(preprocessed_data[feature])

    # Normalization
    scaler = MinMaxScaler()
    preprocessed_data[features] = scaler.fit_transform(preprocessed_data[features])

    # Feature extraction (example: rolling mean and std)
    for feature in features:
        preprocessed_data[f'{feature}_rolling_mean'] = preprocessed_data[feature].rolling(window=10).mean()
        preprocessed_data[f'{feature}_rolling_std'] = preprocessed_data[feature].rolling(window=10).std()

    preprocessed_data = preprocessed_data.dropna().reset_index(drop=True)
    
    # Return the names of the features used for modeling
    final_features = [col for col in preprocessed_data.columns if col not in ['time']]

    return final_features, preprocessed_data[final_features]
