# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 11:34:12 2023

@author: szyns
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier as kNN
from sklearn.svm import SVC as SVM
from sklearn.preprocessing import StandardScaler,MinMaxScaler, RobustScaler
from sklearn.neighbors import KNeighborsClassifier as kNN
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.tree import plot_tree
from matplotlib import pyplot as plt

#ZADANIE 3.2
#from sklearn.metrics import mean_absolute_error, mean_squared_error,mean_absolute_percentage_error

models = [kNN(),SVM()]
from sklearn.datasets import load_breast_cancer
dane = load_breast_cancer()

# Create a DataFrame from the dataset
data = pd.DataFrame(dane.data, columns=dane.feature_names)

# Add the target variable to the DataFrame
data['target'] = dane.target


columns= data.columns.to_list()
X = data.drop(columns=['target']).values
y = data['target'].values

#cat_feature = pd.Categorical(data.Property_Area)
#one_hot = pd.get_dummies(cat_feature)
#data = pd.concat([data,one_hot],axis=1)
#data = data.drop(columns = ['Property_Area'])
'''
def qualitative_to_0_1(data, column, value_to_be_1):
    data[column] = data[column].apply(lambda x: 1 if x == value_to_be_1 else 0)
    return data


data = qualitative_to_0_1(data, 'Gender', 'Female')
data = qualitative_to_0_1(data, 'Married', 'Yes')
data = qualitative_to_0_1(data, 'Education', 'Graduate')
data = qualitative_to_0_1(data, 'Self_Employed', 'Yes')
data = qualitative_to_0_1(data, 'Loan_Status', 'Y')
data = qualitative_to_0_1(data, 'Urban', True)
data = qualitative_to_0_1(data, 'Semiurban', True)
data = qualitative_to_0_1(data, 'Rural', True)

vals = data.values.astype(float)
'''

features = data.columns
X_train, X_test, y_train, y_test  = train_test_split(X,y,test_size = 0.2,shuffle=True)
scaler = StandardScaler()
scaler.fit(X_train)
X_train=scaler.transform(X_train)
X_test=scaler.transform(X_test)
new_models = [kNN(),SVC()]

for model in new_models:
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    cm=confusion_matrix(y_test, y_pred)
    print(cm)
    dispx = ConfusionMatrixDisplay(cm,display_labels=model.classes_)
    dispx.plot()
    plt.show()
      
    print("Wartosci dla matrycy pomylek wygenerowanej w programie, skaler StandardScaler: ")
    
    sensitivity = cm[0][0]/(cm[0][0]+cm[0][1])
    print("Czulsc: ",sensitivity)
    
    precision = cm[0][0]/(cm[0][0]+cm[1][0])
    print("Precyzja:",precision)
    
    specificity = cm[1][1]/(cm[1][1]+cm[1][0])
    print("Specyficznosc: ",specificity)
    
    
    accuracy = (cm[0][0] + cm[1][1])/(cm[0][0] + cm[0][1]+ cm[1][0] + cm[1][1])
    print("Dokladnosc: ",accuracy)
    
    
    F1 = 2*(sensitivity*precision)/(sensitivity+precision)
    print("F1: ",F1)


model= DT(max_depth = 3)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
cm=confusion_matrix(y_test, y_pred)
print(cm)
plt.figure(figsize=(30,10))
tree_vis = plot_tree(model,feature_names=data.columns[:-1].to_list(), class_names = ['N','Y'], fontsize = 20)

scaler = MinMaxScaler()
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
       
    print("Wartosci dla matrycy pomylek wygenerowanej w programie, skaler MinMaxScaler: ")
    
    sensitivity = cm[0][0]/(cm[0][0]+cm[0][1])
    print("Czulsc: ",sensitivity)
    
    precision = cm[0][0]/(cm[0][0]+cm[1][0])
    print("Precyzja:",precision)
    
    specificity = cm[1][1]/(cm[1][1]+cm[1][0])
    print("Specyficznosc: ",specificity)
    
    
    accuracy = (cm[0][0] + cm[1][1])/(cm[0][0] + cm[0][1]+ cm[1][0] + cm[1][1])
    print("Dokladnosc: ",accuracy)
    
    
    F1 = 2*(sensitivity*precision)/(sensitivity+precision)
    print("F1: ",F1)
 
    
model= DT(max_depth = 3)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
cm=confusion_matrix(y_test, y_pred)
print(cm)
plt.figure(figsize=(30,10))
tree_vis = plot_tree(model,feature_names=data.columns[:-1].to_list(), class_names = ['N','Y'], fontsize = 20)

scaler = RobustScaler()
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
       
    print("Wartosci dla matrycy pomylek wygenerowanej w programie, skaler RobustScaler: ")
    
    sensitivity = cm[0][0]/(cm[0][0]+cm[0][1])
    print("Czulsc: ",sensitivity)
    
    precision = cm[0][0]/(cm[0][0]+cm[1][0])
    print("Precyzja:",precision)
    
    specificity = cm[1][1]/(cm[1][1]+cm[1][0])
    print("Specyficznosc: ",specificity)
    
    
    accuracy = (cm[0][0] + cm[1][1])/(cm[0][0] + cm[0][1]+ cm[1][0] + cm[1][1])
    print("Dokladnosc: ",accuracy)
    
    
    F1 = 2*(sensitivity*precision)/(sensitivity+precision)
    print("F1: ",F1)
 
    
model= DT(max_depth = 3)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
cm=confusion_matrix(y_test, y_pred)
print(cm)
plt.figure(figsize=(30,10))
tree_vis = plot_tree(model,feature_names=data.columns[:-1].to_list(), class_names = ['N','Y'], fontsize = 20)
