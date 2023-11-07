# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 11:22:16 2023

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

#ZADANIE 3.2
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
       
    print("Wartosci dla matrycy pomylek wygenerowanej w programie: ")
    
    sensitivity = cm[0][0]/cm[0][0]+cm[0][1]
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



print("Wartosci dla macierzy pomylek kNN z tabeli 3.4: ")
tabela = [[7, 26],[17 ,23]]
print(tabela)
sensitivity = tabela[0][0]/tabela[0][0]+tabela[0][1]
print("Czulsc: ",sensitivity)
 
precision = tabela[0][0]/(tabela[0][0]+tabela[1][0])
print("Precyzja:",precision)
 
specificity = tabela[1][1]/(tabela[1][1]+tabela[1][0])
print("Specyficznosc: ",specificity)
 
 
accuracy = (tabela[0][0] + tabela[1][1])/(tabela[0][0] + tabela[0][1]+ tabela[1][0] + tabela[1][1])
print("Dokladnosc: ",accuracy)
 
 
F1 = 2*(sensitivity*precision)/(sensitivity+precision)
print("F1: ",F1)


print("Wartosci dla macierzy pomylek SVM z tabeli 3.4: ")
tabela1 = [[0, 33],[0 ,90]]
print(tabela1)
#sensitivity = tabela1[0][0]/tabela1[0][0]+tabela1[0][1]
#print("Czulsc: ",sensitivity) Dzielnie przez 0
 
#precision = tabela1[0][0]/(tabela1[0][0]+tabela1[1][0])
#print("Precyzja:",precision) Dzielenie przez 0
 
specificity = tabela1[1][1]/(tabela1[1][1]+tabela1[1][0])
print("Specyficznosc: ",specificity)
 
 
accuracy = (tabela[0][0] + tabela1[1][1])/(tabela1[0][0] + tabela1[0][1]+ tabela1[1][0] + tabela1[1][1])
print("Dokladnosc: ",accuracy)
 
 
#F1 = 2*(sensitivity*precision)/(sensitivity+precision)
#print("F1: ",F1) Dzielenie prze 0


# Tworzenie modeli kNN i SVM z określonymi parametrami
knn_model = kNN(n_neighbors=5, weights='uniform')
svm_model = SVC(kernel='rbf')

# Trenowanie modeli
knn_model.fit(X_train, y_train)
svm_model.fit(X_train, y_train)

# Przewidywanie klas na zbiorze testowym
knn_y_pred = knn_model.predict(X_test)
svm_y_pred = svm_model.predict(X_test)

# Obliczenie macierzy pomyłek dla kNN i SVM
knn_cm = confusion_matrix(y_test, knn_y_pred, labels=knn_model.classes_)
svm_cm = confusion_matrix(y_test, svm_y_pred, labels=svm_model.classes_)

# Wyświetlenie macierzy pomyłek
print("Macierz pomyłek kNN:")
print(knn_cm)
disp = ConfusionMatrixDisplay(knn_cm,display_labels=model.classes_)
disp.plot()
plt.show()

print("Macierz pomyłek SVM:")
print(svm_cm)
disp = ConfusionMatrixDisplay(svm_cm,display_labels=model.classes_)
disp.plot()
plt.show()
