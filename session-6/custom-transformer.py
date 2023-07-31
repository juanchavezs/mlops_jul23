import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# TODO: Modify this list to include the numerical columns
NUMERICAL_VARS = ['pclass', 'age', 'sibsp', 'parch', 'fare']


# Crear custom transformer

class MissingIndicator(BaseEstimator, TransformerMixin):

    def __init__(self, variables):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X, y):
        return self

    def transform(self, X):
        X_copy = X.copy()
        numeric_columns = ['pclass', 'age', 'sibsp', 'parch', 'fare']

        for col in numeric_columns:
            col_name_with_suffix = col + '_isNull'
            X_copy[col_name_with_suffix] = X_copy[col].isnull().astype(int)
            X_copy[col].fillna(1, inplace=True)

        return X_copy


# Leer el csv sin aplicar transformaciones
df = pd.read_csv("raw-data.csv")

mi = MissingIndicator(variables=NUMERICAL_VARS)
# Aplicar las transformaciones
df_mi = mi.transform(df)

# Imprimir resultados despues de las transformaciones
df_mi.head(200)
