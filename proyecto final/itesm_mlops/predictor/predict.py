import argparse
import joblib
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
from datetime import datetime
from sklearn.decomposition import PCA  # Principal Component Analysis for dimensionality reduction
import numpy as np
from sklearn.preprocessing import MinMaxScaler # Data preprocessing

class DataPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
  
    def transform(self, X):
        df_transformed = self.feature_generation(X)
        df_transformed = self.remove_outliers(df_transformed, 'Income')
        df_transformed = self.scaling_func(df_transformed)
        df_transformed = self.dim_reduction(df_transformed)
        return df_transformed

    def feature_generation(self,data: pd.DataFrame) -> pd.DataFrame:        
        """
        Generates and adds new features to the provided DataFrame.

        Parameters:
        data (pd.DataFrame): DataFrame containing the original data.

        Returns:
        pd.DataFrame: DataFrame with the generated and added features.
        """
        # Calculate client age
        data['Age'] = datetime.now().year - data['Year_Birth']  
        
        # Calculate number of years since customer registration
        registration_year = pd.to_datetime(data['Dt_Customer'], format='%d-%m-%Y').apply(lambda x: x.year)
        current_year = datetime.now().year
        data['Years_Since_Registration'] = current_year - registration_year 

        # Encode Education
        data["Education"] = data["Education"].replace({"Basic": 0, "Graduation": 1, "2n Cycle": 2, "Master": 2, "PhD": 3})
        
        # Calculate total amount spent on products
        mnt_cols = ['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
        data['Sum_Mnt'] = data[mnt_cols].sum(axis=1)
        
        # Calculate number of companies in which the client accepted the offer
        accepted_cmp_cols = ['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'Response']
        data['Num_Accepted_Cmp'] = data[accepted_cmp_cols].sum(axis=1)

        # Calculate total number of purchases
        total_purchases = ['NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases']
        data['Num_Total_Purchases'] = data[total_purchases].sum(axis=1)

        # Drop missing values and unnecessary columns
        data.dropna(inplace=True)
        data.drop(['Year_Birth', 'Z_CostContact', 'Z_Revenue', 'Dt_Customer', 'Marital_Status'], axis=1, inplace=True)
    
        # Apply one-hot encoding for remaining categorical features
        data = pd.get_dummies(data)

        return data

    def remove_outliers(self,data: pd.DataFrame, column: str) -> pd.DataFrame:
        """
        Removes outliers from the specified column of the provided DataFrame.

        Parameters:
        data (pd.DataFrame): DataFrame containing the data.
        column (str): Name of the column to remove outliers from.

        Returns:
        pd.DataFrame: DataFrame with outliers removed from the specified column.

        Description:
        This function identifies and removes outliers from the specified column using the Interquartile Range (IQR) method.
        """
        q3, q1 = np.nanpercentile(data[column], [75, 25])
        iqr = q3 - q1
        upper_bound = q3 + 1.5 * iqr
        lower_bound = q1 - 1.5 * iqr
        data = data[(data[column] > lower_bound) & (data[column] < upper_bound)]
        return data

    def scaling_func(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Applies Min-Max scaling to the provided DataFrame.

        Parameters:
        data (pd.DataFrame): DataFrame containing the data to be scaled.

        Returns:
        pd.DataFrame: DataFrame with Min-Max scaled values.

        Description:
        This function performs Min-Max scaling on the provided DataFrame to scale the features within a specified range.
        """
        mms = MinMaxScaler()
        return pd.DataFrame(data=mms.fit_transform(data), columns=data.columns)

    @staticmethod
    def dim_reduction(data: pd.DataFrame) -> pd.DataFrame:
        """
        Performs dimensionality reduction using Principal Component Analysis (PCA) on the provided DataFrame.

        Parameters:
        data (pd.DataFrame): DataFrame containing the data for dimensionality reduction.

        Returns:
        pd.DataFrame: DataFrame with reduced dimensions using PCA.

        Description:
        This function applies Principal Component Analysis (PCA) to reduce the dimensionality of the provided DataFrame.
        """
        n_components = 8
        pca = PCA(n_components=n_components, random_state=42)
        data = pd.DataFrame(pca.fit_transform(data), columns=[f'PC{i}' for i in range(1, n_components + 1)])
        return data

class LoadAndPredict:
    """
    A class for loading KMeans clustering models and predicting clusters.

    Attributes:
    models_dir (str): Directory path for saving and loading models.

    Methods:
    load_kmeans_model(self, n_clusters): Load a KMeans clustering model.
    predict_clusters(self, data, n_clusters): Predict clusters using a loaded KMeans model.
    """
    models_dir = '../models/'

    def load_kmeans_model(self, n_clusters):
        """
        Load a KMeans clustering model from a file.

        Parameters:
        n_clusters (int): Number of clusters associated with the model.

        Returns:
        KMeans: Loaded KMeans model.

        Description:
        This method loads a previously saved KMeans clustering model from a file.

        - 'n_clusters': Number of clusters associated with the model.
        - 'model_filename': Generate the filename for the model based on the number of clusters.
        - Load the KMeans model using joblib.load and return it.
        """
        model_filename = f"{self.models_dir}kmeans_{n_clusters}_clusters_model.pkl"
        return joblib.load(model_filename)

    def predict_clusters(self, data, n_clusters):
        """
        Predict clusters for new data using a loaded KMeans model.

        Parameters:
        data: Data for which clusters are to be predicted.
        n_clusters (int): Number of clusters associated with the model.

        Returns:
        np.ndarray: Predicted cluster labels.

        Description:
        This method predicts clusters for new data using a loaded KMeans model.

        - 'data': New data for cluster prediction.
        - 'n_clusters': Number of clusters associated with the model.
        - Load the KMeans model using 'load_kmeans_model' and predict clusters for the data.
        - Return the array of predicted cluster labels.
        """
        kmeans_model = self.load_kmeans_model(n_clusters)
        return kmeans_model.predict(data)

# Usage
if __name__ == "__main__":
    """
    Demonstration of using the Clustering class for prediction and preprocessing.

    Description:
    This part of the code showcases the usage of the Clustering class to predict clusters for new data
    and save the preprocessed data along with the predicted clusters to a new CSV file.

    - Create an instance of the Clustering class.
    - Load new data for prediction from a CSV file.
    - Perform data preprocessing steps using various functions, including data reduction and scaling.
    - Specify the number of clusters to predict.
    - Predict clusters for the new data using the 'predict_clusters' method.
    - Add the predicted cluster labels to the original DataFrame.
    - Save the preprocessed data with predicted clusters to a new CSV file.
    """
    LoadAndPredict = LoadAndPredict()

    # Load new data for prediction
    df = pd.read_csv("../data/retrieved_data.csv")  # Replace with your new data file

    # Prepare data
    df = DataPreprocessor.feature_generation('/',df)
    df = DataPreprocessor.remove_outliers('/', df,'Income')
    
    df_scaled = DataPreprocessor.scaling_func('/',df)
    df_scaled.index = df.index 
    
    df_transform = DataPreprocessor.dim_reduction(df_scaled)  # df: scaling + dimensionality reduction
    df_transform.index = df_scaled.index

    # Choose the number of clusters to predict
    n_clusters_to_predict = 4

    # Predict clusters for the new data
    df['Predicted_Cluster'] = LoadAndPredict.predict_clusters(df_transform, n_clusters_to_predict)
     
    # Save preprocessed data with predictions to a new CSV
    df.to_csv("data_with_predictions.csv", index=False)