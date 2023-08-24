from pydantic import BaseModel

class data_market(BaseModel):
    """
    - Year_Birth: Customer's birth year
    - Education: Customer's education level
    - Marital_Status: Customer's marital status
    - Income: Customer's yearly household income
    - Kidhome: Number of children in customer's household
    - Teenhome: Number of teenagers in customer's household
    - Dt_Customer: Date of customer's enrollment with the company
    - Recency: Number of days since customer's last purchase
    - Complain: 1 if the customer complained in the last 2 years, 0 otherwise

##### **Products**

    - MntWines: Amount spent on wine in last 2 years
    - MntFruits: Amount spent on fruits in last 2 years
    - MntMeatProducts: Amount spent on meat in last 2 years
    - MntFishProducts: Amount spent on fish in last 2 years
    - MntSweetProducts: Amount spent on sweets in last 2 years
    - MntGoldProds: Amount spent on gold in last 2 years

##### **Promotion**

    -  NumDealsPurchases: Number of purchases made with a discount
    - AcceptedCmp1: 1 if customer accepted the offer in the 1st campaign, 0 otherwise
    - AcceptedCmp2: 1 if customer accepted the offer in the 2nd campaign, 0 otherwise
    - AcceptedCmp3: 1 if customer accepted the offer in the 3rd campaign, 0 otherwise
    - AcceptedCmp4: 1 if customer accepted the offer in the 4th campaign, 0 otherwise
    - AcceptedCmp5: 1 if customer accepted the offer in the 5th campaign, 0 otherwise
    - Response: 1 if customer accepted the offer in the last campaign, 0 otherwise

##### **Place**

    - NumWebPurchases: Number of purchases made through the company’s website
    - NumCatalogPurchases: Number of purchases made using a catalogue
    - NumStorePurchases: Number of purchases made directly in stores
    - NumWebVisitsMonth: Number of visits to company’s website in the last month    """
    
    Education: int = 1
    Income: float = 50000.0
    Kidhome: int = 0
    Teenhome: int = 0
    Recency: int = 30
    MntWines: int = 100
    MntFruits: int = 10
    MntMeatProducts: int = 200
    MntFishProducts: int = 20
    MntSweetProducts: int = 5
    MntGoldProds: int = 3
    NumDealsPurchases: int = 2
    NumWebPurchases: int = 5
    NumCatalogPurchases: int = 3
    NumStorePurchases: int = 2
    NumWebVisitsMonth: int = 7
    AcceptedCmp3: int = 0
    AcceptedCmp4: int = 0
    AcceptedCmp5: int = 0
    AcceptedCmp1: int = 0
    AcceptedCmp2: int = 0
    Complain: int = 0
    Response: int = 0
    Age: int = 40
    Years_Since_Registration: int = 5
    Sum_Mnt: int = 500
    Num_Accepted_Cmp: int = 1
    Num_Total_Purchases: int = 10 
