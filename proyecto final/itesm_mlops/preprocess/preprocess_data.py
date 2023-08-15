from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array
import pandas as pd
import re
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder  # Data preprocessing
from datetime import datetime  # For working with dates and times
from sklearn.decomposition import PCA  # Principal Component Analysis for dimensionality reduction


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
    
# Example Use
# data_transformer = DataPreprocessor()
# clustering = Clustering()

# pipeline = Pipeline([
#     ('data_transformer', data_transformer),
# ])

# df_transform = pipeline.fit_transform(df)
