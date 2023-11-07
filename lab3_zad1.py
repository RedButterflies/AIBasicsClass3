# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 08:10:18 2023

@author: szyns
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier as kNN
from sklearn.svm import SVC as SVM
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier as kNN
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.tree import plot_tree
from matplotlib import pyplot as plt

#ZADANIE 3.1
#from sklearn.metrics import mean_absolute_error, mean_squared_error,mean_absolute_percentage_error

models = [kNN(),SVM()]

data = pd.read_excel('practice_lab_3.xlsx')
columns= data.columns.to_list()
data_arr= data.values
X,y= data_arr[:,:], data_arr[:,:]

cat_feature = pd.Categorical(data.Property_Area)
one_hot = pd.get_dummies(cat_feature)
data = pd.concat([data,one_hot],axis=1)
data = data.drop(columns = ['Property_Area'])

binary_columns = ['Gender', 'Married', 'Education', 'Self_Employed', 'Loan_Status', 'Urban', 'Semiurban', 'Rural']

for col in binary_columns:
    data[col] = data[col].map({'Female':1,'Male':0,'No': 0, 'Yes': 1, 'Graduate': 1, 'Not Graduate': 0, 'Y': 1, 'N': 0, True: 1, False: 0})

features = data.columns
vals = data.values.astype(float)
y=data['Loan_Status'].values
X = data.drop(columns= ['Loan_Status']).values
X_train, X_test, y_train, y_test  = train_test_split(X,y,test_size = 0.2,shuffle=True)

for model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    cm= confusion_matrix(y_test, y_pred,labels = model.classes_)
    print(cm)
    disp = ConfusionMatrixDisplay(cm,display_labels=model.classes_)
    disp.plot()
    plt.show()
    
scaler = StandardScaler()
scaler.fit(X_train)
X_train=scaler.transform(X_train)
X_text=scaler.transform(X_test)
new_models = [kNN(),SVC()]

for model in new_models:
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    cm=confusion_matrix(y_test, y_pred)
    print(cm)
    dispx = ConfusionMatrixDisplay(cm,display_labels=model.classes_)
    dispx.plot()
    plt.show()
 
    
model= DT(max_depth = 3)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
cm=confusion_matrix(y_test, y_pred)
print(cm)
plt.figure(figsize=(30,10))
tree_vis = plot_tree(model,feature_names=data.columns[:-1].to_list(),
   
                     class_names = ['N','Y'], fontsize = 20)



features = data.columns
vals = data.values.astype(float)
y=data['Loan_Status'].values
X = data.drop(columns= ['Loan_Status']).values
X_train, X_test, y_train, y_test  = train_test_split(X,y,test_size = 0.2,shuffle=True)

scaler = StandardScaler()
scaler.fit(X_train)
X_train=scaler.transform(X_train)
X_test=scaler.transform(X_test)
new_models = [kNN(),SVC()]
for model in models:
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    cm=confusion_matrix(y_test, y_pred)
    print(cm)
    dispx = ConfusionMatrixDisplay(cm,display_labels=model.classes_)
    dispx.plot()
    plt.show()
    
    
 