from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
from datetime import datetime

class DataPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
  
    def transform(self, X):
        df_transformed = self.feature_generation(X)
        return df_transformed

    def preprocess_input(data):
        if isinstance(data, pd.DataFrame):
            return data
        elif isinstance(data, dict):
            df = pd.DataFrame([data])
            return df
        else:
            raise ValueError("Los datos proporcionados no son compatibles.")
    
    def feature_generation(self,data: pd.DataFrame) -> pd.DataFrame:        

        # Calculate client age
        data['Age'] = 2023 - data['Year_Birth']
        
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