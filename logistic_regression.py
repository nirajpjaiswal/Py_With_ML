# -*- coding: utf-8 -*-
# Created on Tue Mar 16 09:40:04 2021

# import libraries
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
import numpy as np
from sklearn.preprocessing import LabelEncoder
import statsmodels.api as sm
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import f_classif

# read data
path="F:/aegis/4 ml/dataset/supervised/classification/heart/heart_v1.csv"

data = pd.read_csv(path)

data.shape
data.dtypes

# get the count of Y
data.target.value_counts()

# i) to convert the Y-variable into numbers 
# use labelencoder
le = LabelEncoder()
data['Y'] = le.fit_transform(data.target)
data.head(5)
# drop the old Y-variable
data = data.drop('target',axis=1)
# rename Y to old Y-value
data=data.rename(columns={'Y':'target'})

data.dtypes

# split the columns into nc and fc
nc=data.select_dtypes(exclude='object').columns.values
fc=data.select_dtypes(include='object').columns.values

# EDA
# check for collinearity, distribution etc for numeric data

# NULL check
data.isnull().sum()

# 0 check
data[nc][data[nc]==0].count()

# analyse the factor cols
for f in fc:
    print("Factor Column = ", f)
    print(data[f].unique())
    print("\n")
    
# 'thal' column has invalid levels that have to be imputed

data.thal.value_counts()

# merge '1' and '2' into the level 'fixed'
data.thal[data.thal.isin(['1','2'])] = 'fixed'     

# verify the change
data.thal.value_counts()

# count plot to check the y-distribution
sns.countplot(x='target',data=data)
plt.title('Y-value distribution')

# heatmap
cor = data[nc].corr()
cor=np.tril(cor)
sns.set(font_scale=0.85)
sns.heatmap(cor,xticklabels=nc,yticklabels=nc,
            vmin=-1,vmax=1,square=False,
            annot=True)

# dummy variables
pd.get_dummies(data.gender).head(20)
pd.get_dummies(data.gender,drop_first=True).head(10)

# convert factor to dummies
new_data = data.copy()

for f in fc:
    dummy = pd.get_dummies(data[f],drop_first=True,prefix=f)
    new_data = new_data.join(dummy)

new_data.head(3)

new_cols = new_data.columns

# remove the original factor variables
new_cols = list(set(new_cols) - set(fc))

print(new_cols)

# refresh the new dataset with the new columns
new_data = new_data[new_cols]

# split the data into train and test
trainx,testx,trainy,testy=train_test_split(
    new_data.drop('target',axis=1),
    new_data['target'],
    test_size=0.25)

print('trainx={},trainy={},testx={},testy={}'.format(trainx.shape,trainy.shape,testx.shape,testy.shape))

# build the logistic regression model using Logit()
m1 = sm.Logit(trainy,trainx).fit()

# summarise the model
m1.summary()

# beta values for individual features
m1.params


# k-Fold Cross-Validation    

folds=5    
cv_acc = []

X=trainx.values
Y=trainy.values

kf= KFold(folds)

for train_index,test_index in kf.split(X):
    cv_trainx,cv_trainy = X[train_index],Y[train_index]
    cv_testx, cv_testy =  X[test_index],Y[test_index]
    
    # build model on cv_train and predict on cv_test
    m = sm.Logit(cv_trainy,cv_trainx).fit()
    p = m.predict(cv_testx)
    
    # covert predictions into classes
    p_Y = p.copy()
    
    # take the cutoff as 0.5
    p_Y[p_Y < 0.5] = 0
    p_Y[p_Y > 0.5] = 1
    
    # calculate the accuracy and append to list
    cv_acc.append(np.round(accuracy_score(cv_testy,p_Y),3))
    
# print the CV accuracies
print(cv_acc)        

# train accuracy
np.round(np.mean(cv_acc),2)


# prediction on the test data
p1 = m1.predict(testx)

# convert the probabilities into classes
p1_Y = p1.copy()
p1_Y[p1_Y < 0.5] = 0
p1_Y[p1_Y > 0.5] = 1

# confusion matrix
# method 1
confusion_matrix(testy,p1_Y)

# method 2
df=pd.DataFrame({'actual':testy,'predicted':p1_Y})
pd.crosstab(df.actual,df.predicted,margins=True)

# classification_report
print(classification_report(testy,p1_Y))

# accuracy score
print(accuracy_score(testy,p1_Y))

# AUC / ROC
from sklearn import metrics
fpr,tpr,threshold = metrics.roc_curve(testy,p1_Y)

# auc score
auc_score = metrics.auc(fpr,tpr) 
print("AUC for model = ", auc_score)

# plot the ROC
plt.plot(fpr,tpr,'b')
plt.plot([0,1],[0,1],'r--')
plt.xlim([0,1])
plt.ylim([0,1])
plt.title('ROC Curve. AUC = ' + str(round(auc_score,2)))
plt.xlabel('FPR')
plt.ylabel('TPR')


# feature selection
features = trainx.columns
scores, pval = f_classif(trainx,trainy)

# store the feature's scores in a dataframe
df_scores = pd.DataFrame({'feature':features,
                          'score':scores,
                          'pvalue':pval})

# sort the dataset in the descending order of Scores
df_scores.sort_values('score',ascending=False,inplace=True)
df_scores

# how to make this model better
# -----------------------------
# try changing the cutoff
# check collinearity
# feature selection
# try to get more records of +ve class
# try to change the train/test ratio











































    
    
    
    
    
    
    
















