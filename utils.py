"""
Utility functions for plotting and visualization.
"""
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

def plot_data(data, features, anomalies):
    """
    Plots sensor data and highlights anomalies.
    """
    fig = make_subplots(rows=len(features), cols=1, shared_xaxes=True, subplot_titles=features)

    anomaly_indices = data[anomalies == -1].index

    for i, feature in enumerate(features):
        # Plot main sensor data
        fig.add_trace(
            go.Scatter(
                x=data.index, 
                y=data[feature], 
                name=feature,
                mode='lines'
            ),
            row=i+1,
            col=1
        )
        
        # Highlight anomalies
        fig.add_trace(
            go.Scatter(
                x=anomaly_indices,
                y=data.loc[anomaly_indices, feature],
                mode='markers',
                marker=dict(color='red', size=8),
                name='Anomaly'
            ),
            row=i+1,
            col=1
        )

    fig.update_layout(
        height=200 * len(features), 
        title_text="Sensor Data with Anomalies",
        showlegend=False
    )
    return fig
