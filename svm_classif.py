# -*- coding: utf-8 -*-
# Created on Thu Mar 18 09:23:58 2021

# SVM classification
# dataset: diabetes

import pandas as pd
import numpy as np

from sklearn import preprocessing, svm 

from sklearn.model_selection import train_test_split,cross_val_score 

from sklearn.metrics import accuracy_score,classification_report

# read data
path="F:/aegis/4 ml/dataset/supervised/classification/diabetes/diab.csv"
diab=pd.read_csv(path)

# since the y is repeating, remove the feature 'class val'
diab = diab.drop('class_val',axis=1)

# do the EDA to ensure there are no missing / bad data

# standardize the dataset
diab_std = diab.copy()

minmax = preprocessing.MinMaxScaler()
scaledvals = minmax.fit_transform(diab_std.iloc[:,:])
diab_std.iloc[:,:] = scaledvals

# replace the Y-variable with the original values
diab_std['class'] = diab['class']

diab_std.shape

# split the dataset into train and test
trainx,testx,trainy,testy=train_test_split(diab_std.drop('class',axis=1),
                                           diab_std['class'],
                                           test_size=0.25)


# kernels = linear,linearSVC,polynomial,sigmoid,rbf
 
# determine the best C for linear and linearSVC kernels
lov_C = range(1,11)

cv_scores = []

for c in lov_C:
    model = svm.SVC(C=c,kernel='linear')
    acc = cross_val_score(model,trainx,trainy,cv=5,scoring='accuracy')
    acc = np.round(np.mean(acc),3)
    cv_scores.append(acc)

bestC = lov_C[cv_scores.index(max(cv_scores))] 
print("Best C = ", bestC)

# build the model using the best C
# 1) kernel = 'linear'
m1 = svm.SVC(C=bestC,kernel='linear').fit(trainx,trainy)
p1 = m1.predict(testx)
accuracy_score(testy,p1)
df=pd.DataFrame({'actual':testy,'predicted':p1})
ct1=pd.crosstab(df.actual,df.predicted,margins=True)
print(classification_report(testy,p1))

# 2) kernel = 'svcLinear', C = bestC
m2=svm.LinearSVC(C=bestC,max_iter=1200).fit(trainx,trainy)
p2=m2.predict(testx)
accuracy_score(testy,p2)
df=pd.DataFrame({'actual':testy,'predicted':p2})
ct2=pd.crosstab(df.actual,df.predicted,margins=True)
ct2
print(classification_report(testy,p2))


# determine the best C and Gamma for poly,sigmoid,rbf
lov_C = range(1,11)
lov_G = np.linspace(0.1,0.99,10)

cv_scores = []
cg = []

for c in lov_C:
    for g in lov_G:
        model = svm.SVC(kernel='rbf',C=c,gamma=g)
        acc = cross_val_score(model,trainx,trainy,cv=5,scoring='accuracy')
        acc = np.round(np.mean(acc),3)
        cv_scores.append(acc)
        cg.append(str(c)+":"+str(g))

print(cv_scores,end=" ")
print(cg,end= " ")

# highest accuracy and the best C and Gamma
best = cg[cv_scores.index(max(cv_scores))]

bestC = int(best.split(":")[0])
bestG = float(best.split(":")[1])

# build the models where kernel is ('rbf','sigmoid','poly')

# kernel = 'rbf'
m3 = svm.SVC(kernel='rbf',C=bestC,gamma=bestG).fit(trainx,trainy)
p3 = m3.predict(testx)
accuracy_score(testy,p3)
df=pd.DataFrame({'actual':testy,'predicted':p3})
ct3=pd.crosstab(df.actual,df.predicted,margins=True)
ct3
print(classification_report(testy,p3))

# kernel = 'sigmoid'
m4 = svm.SVC(kernel='sigmoid',C=bestC,gamma=bestG).fit(trainx,trainy)
p4 = m4.predict(testx)
accuracy_score(testy,p4)
df=pd.DataFrame({'actual':testy,'predicted':p4})
ct4=pd.crosstab(df.actual,df.predicted,margins=True)
ct4
print(classification_report(testy,p4))

# kernel = 'poly'
m5 = svm.SVC(kernel='poly',C=bestC,gamma=bestG).fit(trainx,trainy)
p5 = m5.predict(testx)
accuracy_score(testy,p5)
df=pd.DataFrame({'actual':testy,'predicted':p5})
ct5=pd.crosstab(df.actual,df.predicted,margins=True)
ct5
print(classification_report(testy,p5))









































