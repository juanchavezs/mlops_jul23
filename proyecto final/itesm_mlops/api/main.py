import os
import sys

# Add the parent directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)

from fastapi import FastAPI
from starlette.responses import JSONResponse

from predictor.predict import LoadAndPredict
from .models.models import data_market

app = FastAPI()

@app.get('/', status_code=200)
async def healthcheck():
    return 'Cluster Classifiers is ready!'

@app.post('/predict')
def extract_name(market_features: data_market):
    predictor = predict_clusters("/Users/Usuario 1/Desktop/repos/mlops_jul23/proyecto final/itesm_mlops/models/kmeans_4_clusters_model.pkl")
    X = [market_features.ID,
    market_features.Educationduation,
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
    market_features.Response]
    prediction = predictor.predict([X])
    return JSONResponse(f"Resultado predicción: {prediction}")


