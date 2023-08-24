import joblib

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
    
# # Usage
# if __name__ == "__main__":
#     """
#     Demonstration of using the Clustering class for prediction and preprocessing.

#     Description:
#     This part of the code showcases the usage of the Clustering class to predict clusters for new data
#     and save the preprocessed data along with the predicted clusters to a new CSV file.

#     - Create an instance of the Clustering class.
#     - Load new data for prediction from a CSV file.
#     - Perform data preprocessing steps using various functions, including data reduction and scaling.
#     - Specify the number of clusters to predict.
#     - Predict clusters for the new data using the 'predict_clusters' method.
#     - Add the predicted cluster labels to the original DataFrame.
#     - Save the preprocessed data with predicted clusters to a new CSV file.
#     """
#     LoadAndPredict = LoadAndPredict()

#     # Load new data for prediction
#     df = pd.read_csv("../data/retrieved_data.csv")  # Replace with your new data file

#     # Prepare data
#     df = DataPreprocessor.feature_generation('/',df)    
#     df_transform = DataPreprocessor.scaling_func('/',df)
#     df_transform.index = df.index

#     # Choose the number of clusters to predict
#     n_clusters_to_predict = 4

#     # Predict clusters for the new data
#     df['Predicted_Cluster'] = LoadAndPredict.predict_clusters(df_transform, n_clusters_to_predict)
     
#     # Save preprocessed data with predictions to a new CSV
#     df.to_csv("data_with_predictions.csv", index=False)