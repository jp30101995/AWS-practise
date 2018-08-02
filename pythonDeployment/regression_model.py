from sklearn.externals import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
from sklearn.model_selection import train_test_split

# Load the diabetes dataset
df=pd.read_csv("https://s3-ap-southeast-1.amazonaws.com/data-for-models-python/data/trainingtestingdata.csv")
# Use only one feature
y=df.model
X=df.drop('model',axis=1)
# Split the data into training/testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2)
# Create linear regression object
regr = linear_model.LinearRegression()
# Train the model using the training sets
regr.fit(X_train, y_train)

#store model in file
joblib.dump(regr, 'regmodel.pkl')
