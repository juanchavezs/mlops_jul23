"""Main module."""
from load.load_data import WebDataRetriever
from preprocess.preprocess_data import DataPreprocessor
from train.train_data import Clustering
from sklearn.pipeline import Pipeline

import pandas as pd

DATASETS_DIR = './data/'
URL = 'https://raw.githubusercontent.com/juanchavezs/mlops_jpcs_proyectofinal/master/marketing_campaign.csv'
RETRIEVED_DATA = 'retrieved_data.csv'
DELIMITER = '\t'

TRAINED_MODEL_DIR = './models/'

if __name__ == "__main__":
    
    # Retrieve data
    data_retriever = WebDataRetriever(URL, DATASETS_DIR,DELIMITER)
    result = data_retriever.retrieve_data()
    print(result)
    
    data_transformer = DataPreprocessor()  

    pipeline = Pipeline([  ('data_transformer',
                            data_transformer),
       ])

    # Read data
    df = pd.read_csv(DATASETS_DIR + RETRIEVED_DATA)
    
    data_transformer = DataPreprocessor()
    
    pipeline = Pipeline([
        ('data_transformer', data_transformer),
    ])

    df_transform = pipeline.fit_transform(df)

    print('Ends preprocessing')

    clustering = Clustering()
    clustering.make_models(df_transform)