from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
from datetime import datetime  # For working with dates and times

class DataPreprocessor(BaseEstimator, TransformerMixin):
  
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_transformed = self.feature_generation(X.copy())
        return X_transformed

    def feature_generation(self, data: pd.DataFrame) -> pd.DataFrame:
        data_copy = data.copy()

        data_copy['Age'] = datetime.now().year - data_copy['Year_Birth']  
        
        registration_year = pd.to_datetime(data_copy['Dt_Customer'], format='%d-%m-%Y').apply(lambda x: x.year)
        current_year = datetime.now().year
        data_copy['Years_Since_Registration'] = current_year - registration_year 

        data_copy["Education"] = data_copy["Education"].replace({"Basic": 0, "Graduation": 1, "2n Cycle": 2, "Master": 2, "PhD": 3})
        
        mnt_cols = ['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
        data_copy['Sum_Mnt'] = data_copy[mnt_cols].sum(axis=1)
        
        accepted_cmp_cols = ['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'Response']
        data_copy['Num_Accepted_Cmp'] = data_copy[accepted_cmp_cols].sum(axis=1)

        total_purchases = ['NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases']
        data_copy['Num_Total_Purchases'] = data_copy[total_purchases].sum(axis=1)

        data_copy.dropna(inplace=True)
        data_copy.drop(['Year_Birth', 'Z_CostContact', 'Z_Revenue', 'Dt_Customer', 'Marital_Status'], axis=1, inplace=True)
    
        data_copy = pd.get_dummies(data_copy)

        return data_copy

def transformacion( data: pd.DataFrame) -> pd.DataFrame:
    data_copy = data.copy()
    data_copy['Age'] = datetime.now().year - data_copy['Year_Birth']  
    registration_year = pd.to_datetime(data_copy['Dt_Customer'], format='%d-%m-%Y').apply(lambda x: x.year)
    current_year = datetime.now().year
    data_copy['Years_Since_Registration'] = current_year - registration_year 
    data_copy["Education"] = data_copy["Education"].replace({"Basic": 0, "Graduation": 1, "2n Cycle": 2, "Master": 2, "PhD": 3})
    mnt_cols = ['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
    data_copy['Sum_Mnt'] = data_copy[mnt_cols].sum(axis=1)
    accepted_cmp_cols = ['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'Response']
    data_copy['Num_Accepted_Cmp'] = data_copy[accepted_cmp_cols].sum(axis=1)

    total_purchases = ['NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases']
    data_copy['Num_Total_Purchases'] = data_copy[total_purchases].sum(axis=1)

    data_copy.dropna(inplace=True)
    data_copy.drop(['Year_Birth', 'Z_CostContact', 'Z_Revenue', 'Dt_Customer', 'Marital_Status'], axis=1, inplace=True)
    data_copy = pd.get_dummies(data_copy)

    return data_copy    
# Example Use
# data_transformer = DataPreprocessor()
# clustering = Clustering()

# pipeline = Pipeline([
#     ('data_transformer', data_transformer),
# ])

# df_transform = pipeline.fit_transform(df)
