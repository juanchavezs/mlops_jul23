import pytest
import pandas as pd
import os
import sys

import os
import sys

current_dir = os.path.abspath(os.path.dirname(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(project_root)

print("Actual Directory:", current_dir)

from itesm_mlops.preprocess.preprocess_data import DataPreprocessor


def test_model_existence():
    """
    Test whether the trained models exists.
    """

    for n_clusters in range(2, 8):
        # Train the pipeline and save the model

        model_filename = f"./itesm_mlops/models/kmeans_{n_clusters}_clusters_model.pkl"
        
        # Verify the trained model's existence for the current number of clusters
        assert os.path.exists(model_filename)
