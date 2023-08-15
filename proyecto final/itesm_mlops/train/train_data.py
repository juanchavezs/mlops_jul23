import os
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score  # Clustering evaluation metrics

import joblib  # For saving and loading machine learning models

class Clustering:
    """
    A class for performing KMeans clustering analysis and saving models.

    Attributes:
    min_n_clusters (int): Minimum number of clusters for analysis.
    max_n_clusters (int): Maximum number of clusters for analysis.
    kmeans_distortions (list): List to store model distortions for different hyperparameters (n_clusters).
    models_dir (str): Directory to save trained models.

    Methods:
    load_kmeans_model(self, n_clusters): Load a KMeans clustering model from a file.
    predict_clusters(self, data, n_clusters): Predict clusters using a loaded KMeans model.
    make_models(self, df_transform): Build KMeans clustering models and save them.
    save_kmeans_model(self, model, n_clusters): Save a trained KMeans model to a file.
    """
    min_n_clusters = 2
    max_n_clusters = 8
    kmeans_distortions = []  # Model distortions for different hyperparameters(n_clusters)
    models_dir = '../models/'  # Directory to save models

    def __init__(self):
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)

    def fit(self, data):
        return self 
    
    def save_kmeans_model(self, model, n_clusters) -> None:
        """
        Save the trained KMeans clustering model to a file.

        Parameters:
        model: Trained KMeans model to be saved.
        n_clusters (int): Number of clusters in the model.

        Returns:
        None

        Description:
        This method saves the trained KMeans clustering model to a file in the specified models directory.

        - 'model': Trained KMeans model to be saved.
        - 'n_clusters': Number of clusters associated with the model.
        - 'model_filename': Generate a filename for the model based on the number of clusters.
        - Save the KMeans model using the joblib.dump function.
        - Print a confirmation message indicating the successful saving of the model.
        """
        model_filename = f"{self.models_dir}kmeans_{n_clusters}_clusters_model.pkl"
        joblib.dump(model, model_filename)
        print(f"Saved KMeans model with {n_clusters} clusters to {model_filename}")
       
    def make_models(self, df_transform: pd.DataFrame) -> None:
        """
        Build KMeans clustering models for various numbers of clusters and save the models.

        Parameters:
        df_transform (pd.DataFrame): DataFrame with transformed data for clustering.

        Returns:
        None

        Description:
        This method constructs KMeans clustering models with varying numbers of clusters
        and evaluates the models using the silhouette score. It also saves the KMeans models
        for future use.

        - 'df_transform': DataFrame containing the transformed data for clustering.
        - 'metrics': A list to store silhouette scores for different numbers of clusters.
        - Iterate through a range of cluster numbers from 'min_n_clusters' to 'max_n_clusters'.
        - Create a KMeans model with the specified number of clusters and fit it to the transformed data.
        - Save the trained KMeans model using the 'save_kmeans_model' method.
        """
        metrics = []  # metrics: silhouette score
        for n_clusters in range(Clustering.min_n_clusters, Clustering.max_n_clusters + 1):
            kmeans = KMeans(n_clusters=n_clusters, init="k-means++", n_init=10, max_iter=280, random_state=42)
            pred = kmeans.fit_predict(df_transform)
            labels = pd.DataFrame(pred, columns=['Labels'], index=df_transform.index)
            self.save_kmeans_model(kmeans, n_clusters)  # Save KMeans model
            print('Model {} saved'.format(n_clusters))