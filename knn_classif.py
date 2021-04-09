# -*- coding: utf-8 -*-
# Created on Thu Mar 18 09:23:58 2021

# k-NN classification
# dataset: wheat

# import libraries
import pandas as pd
import numpy as np

# to convert data into std form (minmax)
from sklearn import preprocessing

# cross_val_score -> cross validation score
from sklearn.model_selection import train_test_split,cross_val_score

from sklearn import neighbors

from sklearn.metrics import accuracy_score,classification_report

import matplotlib.pyplot as plt

# read data
path="F:/aegis/4 ml/dataset/supervised/classification/wheat/wheat.csv"
wheat=pd.read_csv(path)

wheat.head()

# check the Y-distribution
wheat.type.value_counts()


# standardize the data - minmax function

wheat_std = wheat.copy()

# create an instance of minmax scaler
minmax=preprocessing.MinMaxScaler()

# standardization to be done only on the features
scaledvals=minmax.fit_transform(wheat_std.iloc[:,:])
wheat_std.iloc[:,:]=scaledvals

wheat_std.head(10)

# replace the Y with the actual data
wheat_std.type = wheat.type

# shuffle dataset
wheat_std = wheat_std.sample(frac=1)

wheat_std.shape

# split data into train and test
trainx,testx,trainy,testy = train_test_split(wheat_std.drop('type',axis=1),
                                             wheat_std.type,
                                             test_size=0.2) 

# cross-validation to get the optimal neighbours
# for classification, take the number of neighbours (k) as an Odd number to avoid ties during predictions

nn = range(3,12,2)

# store the accuracy of CV for every value of K
cv_score = []

for k in nn:
    model=neighbors.KNeighborsClassifier(n_neighbors=k)
    acc=cross_val_score(model,trainx,trainy,cv=5,scoring='accuracy')
    acc = np.round(np.mean(acc),3)
    cv_score.append(acc)
    
# print the accuracy for each K
print(cv_score)        

# optimal K 
bestk = nn[cv_score.index(max(cv_score))]
print("best K =",bestk)

# plot the chart to determine the best K
plt.plot(nn, cv_score,color='blue')
plt.xlabel('Neighbors')
plt.ylabel('Accuracy')
plt.title('Best Value of K based on CV accuracy')

# knn model and prediction using the best K
m1 = neighbors.KNeighborsClassifier(n_neighbors=bestk,
                                    metric='manhattan').fit(trainx,trainy)

p1 = m1.predict(testx)

# accuracy
accuracy_score(testy,p1)

# confusion matrix
cm=pd.DataFrame({'actual':testy,'predicted':p1})
pd.crosstab(cm.actual,cm.predicted,margins=True)
print(classification_report(testy,p1))














 





