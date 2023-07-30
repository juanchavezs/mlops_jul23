import os # This is an unused import
import json # This is an unused import
from sklearn.datasets import load_iris # Import in incorrect order

import numpy as np # Import in incorrect order
import json # This is an unused import

from my_library import test # This is an unused import
from sklearn.linear_model import LogisticRegression # Import in incorrect order

# Load data from sklearn
X, y = load_iris(return_X_y=True)

# Train the model using regresion logistic
clf = LogisticRegression(solver='lbfgs',max_iter=1000,multi_class='multinomial').fit(X, y)
# Define iris types
iris_type = {0: 'setosa',1: 'versicolor',2: 'virginica'}


# Define dummy values
sepal_length, sepal_width, petal_length, petal_width = 2, 3, 4, 6

X = [sepal_length, sepal_width, petal_length, petal_width]


# Make a prediction

prediction = clf.predict_proba([X])
print({'class': iris_type[np.argmax(prediction)],'probability':round(max(prediction[0]), 2)})