import pytest
import DataPreprocessor()
import pandas as pd
import os

def test_data_existence():
    """
    Test whether training data exists.
    """
    assert 'df' in globals(), "No training data available"


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


def test_model_existence():
    """
    Test whether the trained models exists.
    """

    for n_clusters in range(2, 8):
        # Train the pipeline and save the model

        model_filename = f"./itesm_mlops/models/kmeans_{n_clusters}_clusters_model.pkl"
        
        # Verify the trained model's existence for the current number of clusters
        assert os.path.exists(model_filename)