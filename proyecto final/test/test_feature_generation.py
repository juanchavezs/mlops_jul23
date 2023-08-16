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

def test_feature_generation():
    """
    Test the feature generation process of the DataPreprocessor class.
    """
    # Create an instance of DataPreprocessor
    data_preprocessor = DataPreprocessor()

    # Prepare test data for transformation
    test_data = pd.DataFrame()  # Your test data here

    # Perform the transformation
    transformed_data = data_preprocessor.feature_generation(test_data)

    # Verify that the transformation is done correctly
    assert 'Age' in transformed_data.columns
    assert 'Years_Since_Registration' in transformed_data.columns
    # Add more assertions for other generated features


