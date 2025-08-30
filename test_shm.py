import unittest
import pandas as pd
import numpy as np
from data_preprocessing import preprocess_data
from models import train_isolation_forest, train_autoencoder

class TestPreprocessing(unittest.TestCase):
    def setUp(self):
        # Create a small sample DataFrame
        self.df = pd.DataFrame({
            'time': np.arange(20),
            'sensor1': np.random.normal(0, 1, 20),
            'sensor2': np.random.normal(5, 2, 20)
        })

    def test_preprocess_data_output(self):
        features, processed = preprocess_data(self.df)
        self.assertIsInstance(features, list)
        self.assertIsInstance(processed, pd.DataFrame)
        self.assertGreater(len(features), 0)
        self.assertEqual(processed.shape[0], self.df.shape[0] - 9)  # due to rolling window

class TestModelShapes(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            'sensor1': np.random.normal(0, 1, 30),
            'sensor2': np.random.normal(5, 2, 30)
        })

    def test_isolation_forest_shape(self):
        model = train_isolation_forest(self.df)
        preds = model.predict(self.df)
        self.assertEqual(preds.shape[0], self.df.shape[0])

    def test_autoencoder_shape(self):
        model = train_autoencoder(self.df, timesteps=10)
        # Prepare input for prediction
        n_samples = self.df.shape[0] - 10
        n_features = self.df.shape[1]
        X = np.zeros((n_samples, 10, n_features))
        for i in range(n_samples):
            X[i] = self.df.iloc[i:i+10].values
        preds = model.predict(X)
        self.assertEqual(preds.shape, X.shape)

if __name__ == '__main__':
    unittest.main()
