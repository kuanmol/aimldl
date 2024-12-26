import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values


imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:, 1:3])  #calc mean
X[:,1:3] = imputer.transform(X[:,1:3])

#Independent Variable encoding
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
#print(X)

#Dependenet Variable encoding
from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()
Y=le.fit_transform(Y)
print(Y)