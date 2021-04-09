# -*- coding: utf-8 -*-
# Created on Thu Mar 18 09:23:58 2021

# random forest classification
# dataset: CTG (cardiotocography)

# import libraries
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# read dataset
path="F:/aegis/4 ml/dataset/supervised/classification/cardiotacography/ctg_1.csv"
ctg = pd.read_csv(path)

ctg.head(3)
ctg.columns
ctg.dtypes

# EDA -> null checks and collinearity checks
cols = list(ctg.columns)
cols.remove('NSP')

# build the correlation matrix (lower triangle)
cor = ctg.corr()
cor = np.tril(cor)
sns.set(font_scale=0.52)
sns.heatmap(cor,vmin=-1,vmax=1,
            xticklabels=cols, yticklabels=cols,
            annot=True,square=False)

# there exists correlation among features. they need to be analysed and removed to avoid overfitting

ctg.shape

# split data into train and test
trainx,testx,trainy,testy=train_test_split(ctg.drop('NSP',axis=1),
                                           ctg.NSP,
                                           test_size=0.3)

print('trainx={},trainy={},testx={},testy={}'.format(trainx.shape,trainy.shape,testx.shape,testy.shape))

# random forest model and prediction
m1 =  RandomForestClassifier().fit(trainx,trainy)
p1 = m1.predict(testx)

# accuracy
accuracy_score(testy,p1)

# confusion matrix
# confusion_matrix(testy,p1) # not easy to read and interpret

# ii) use crosstab for a detailed CM
df1=pd.DataFrame({'actual':testy,'predicted':p1})
pd.crosstab(df1.actual,df1.predicted,margins=True)

# pd.crosstab(df1.predicted,df1.actual,margins=True)


dir(m1)

# important features
scores=pd.DataFrame({'features':cols,
                     'score':m1.feature_importances_})
scores.sort_values('score',ascending=False,inplace=True)

# plot the significant features
sns.barplot(x=scores.score,y=scores.features,color='green')
plt.title('Random Forest - Significant Features')
plt.xlabel('Score')
plt.ylabel('Features')


# with hyperparameters
m2=RandomForestClassifier(n_estimators=20,criterion='entropy',
                          max_depth=6,min_samples_split=3,
                          min_samples_leaf=2).fit(trainx,trainy)

p2=m2.predict(testx)

accuracy_score(testy,p2)

# ii) use crosstab for a detailed CM
df1=pd.DataFrame({'actual':testy,'predicted':p2})
pd.crosstab(df1.actual,df1.predicted,margins=True)


