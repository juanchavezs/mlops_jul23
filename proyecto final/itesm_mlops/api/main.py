import os
import sys
from models.models import data_market
import pandas as pd
from fastapi import FastAPI
from starlette.responses import JSONResponse
from datetime import datetime  # For working with dates and times


current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# from preprocess.preprocess_data import transformacion

from predictor.predict_data import LoadAndPredict
predictor =LoadAndPredict()

app = FastAPI()


def data_transform( data: pd.DataFrame) -> pd.DataFrame:
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

    print('ya se finalizó')

    return data_copy   

@app.get('/', status_code=200)
async def healthcheck():
    return 'Cluster Classifiers is ready!'

@app.post('/predict')
def extract_name(market_features: data_market):
    predictor.load_kmeans_model(4)
    df = pd.DataFrame([market_features.ID,
    market_features.Education,
    market_features.Year_Birth,
    market_features.Marital_Status,
    market_features.Income,
    market_features.Kidhome,
    market_features.Teenhome,
    market_features.Dt_Customer, 
    market_features.Recency,
    market_features.MntWines,
    market_features.MntFruits,
    market_features.MntMeatProducts,
    market_features.MntFishProducts,
    market_features.MntSweetProducts,
    market_features.MntGoldProds,
    market_features.NumDealsPurchases,
    market_features.NumWebPurchases,
    market_features.NumCatalogPurchases,
    market_features.NumStorePurchases,
    market_features.NumWebVisitsMonth,
    market_features.AcceptedCmp3,
    market_features.AcceptedCmp4,
    market_features.AcceptedCmp5,
    market_features.AcceptedCmp1,
    market_features.AcceptedCmp2,
    market_features.Complain,
    market_features.Z_CostContact, 
    market_features.Z_Revenue,
    market_features.Response])
    
    df_transform = data_transform(data = df)

    prediction = predictor.predict_clusters(df_transform,4)

    print('Aquí ya se corrió la predicción')

    return JSONResponse(f"Resultado predicción: {prediction}")


