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

def test_remove_outliers():
    """
    Test the remove_outliers method of the DataPreprocessor class.
    """
    # Create an instance of DataPreprocessor
    data_preprocessor = DataPreprocessor()

    # Prepare test data for transformation
    test_data = pd.DataFrame()  # Your test data here

    # Perform the transformation
    transformed_data = data_preprocessor.remove_outliers(test_data, 'Income')

    # Verify that outliers are removed correctly
    lower_bound, upper_bound = data_preprocessor.calculate_bounds(test_data, 'Income')
    assert all(transformed_data['Income'] > lower_bound)
    assert all(transformed_data['Income'] < upper_bound)
