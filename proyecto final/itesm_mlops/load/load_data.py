import pandas as pd
import numpy as np
import re
import os

class WebDataRetriever:
    """
    A class for retrieving data from a given URL.

    Parameters:
        url (str): The URL from which the data will be loaded.

    Attributes:
        url (str): The URL from which the data will be loaded.

    Example usage:
    ```
    URL = 'https://www.openml.org/data/get_csv/16826755/phpMYEkMl'
    data_retriever = WebDataRetriever(URL, DATASETS_DIR)
    result = data_retriever.retrieve_data()
    print(result)
    ```
    """

    DATASETS_DIR = './data/'  # Directory where data will be saved.
    RETRIEVED_DATA = 'retrieved_data.csv'  # File name for the retrieved data.

    def __init__(self, url, data_path, delimiter_url):
        self.url = url
        self.DATASETS_DIR = data_path
        self.delimiter_url = delimiter_url

    def retrieve_data(self):
        """
        Retrieves data from the specified URL, processes it, and stores it in a CSV file.

        Returns:
            str: A message indicating the location of the stored data.
        """
        # Loading data from specific URL
        data = pd.read_csv(self.url , delimiter = self.delimiter_url)

        # Create directory if it does not exist
        if not os.path.exists(self.DATASETS_DIR):
            os.makedirs(self.DATASETS_DIR)
            print(f"Directory '{self.DATASETS_DIR}' created successfully.")
        else:
            print(f"Directory '{self.DATASETS_DIR}' already exists.")

        # Save data to CSV file
        data.to_csv(self.DATASETS_DIR + self.RETRIEVED_DATA, index=False)
        
        return f'Data stored in {self.DATASETS_DIR + self.RETRIEVED_DATA}'
    
# Example
# URL = 'https://raw.githubusercontent.com/juanchavezs/mlops_jpcs_proyectofinal/master/marketing_campaign.csv'
# DELIMITER = '\t'
# data_retriever = WebDataRetriever(url= URL, delimiter_url= DELIMITER , data_path= './data/')
# result = data_retriever.retrieve_data()
# print(result)