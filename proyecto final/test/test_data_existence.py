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


def test_data_existence():
    """
    Test whether training data exists.
    """
    assert 'df' in globals(), "No training data available"