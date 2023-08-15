from pydantic import BaseModel

class data_market(BaseModel):
    """
    - ID: Customer's unique identifier
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
    Response: 1 if customer accepted the offer in the last campaign, 0 otherwise

##### **Place**

    - NumWebPurchases: Number of purchases made through the company’s website
    - NumCatalogPurchases: Number of purchases made using a catalogue
    - NumStorePurchases: Number of purchases made directly in stores
    - NumWebVisitsMonth: Number of visits to company’s website in the last month    """
    
    ID: int  = 1
    Education: str = 'Graduation'
    Year_Birth:  int = 1957
    Marital_Status: str = 'Single'
    Income: float
    Kidhome: int  
    Teenhome: int  
    Dt_Customer: str = '04-09-2012' 
    Recency: float  = 50
    MntWines: float  = 635
    MntFruits: float  = 88
    MntMeatProducts: float =  546
    MntFishProducts: float  = 172
    MntSweetProducts: float  = 88
    MntGoldProds: float  =88
    NumDealsPurchases: float = 3
    NumWebPurchases: float  = 8
    NumCatalogPurchases: float = 10
    NumStorePurchases: float  = 4
    NumWebVisitsMonth: float  = 7
    AcceptedCmp3: float  = 0
    AcceptedCmp4: float  = 0
    AcceptedCmp5: float  = 0
    AcceptedCmp1: float  = 0
    AcceptedCmp2: float  = 0
    Complain: float  = 0
    Z_CostContact: float = 3 
    Z_Revenue: float  = 11
    Response: float = 1
    
