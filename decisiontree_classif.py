# -*- coding: utf-8 -*-
# Created on Wed Mar 17 09:32:44 2021

# Decision Tree classification
# dataset: ecoli

# import libraries

import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix,classification_report,accuracy_score

# plot the DT
from sklearn import tree
from IPython.display import Image
from subprocess import check_call

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# RFE (Recusrive Feature Elimination) - feature selection
from sklearn.feature_selection import RFE

# read the data
path="F:/aegis/4 ml/dataset/supervised/classification/ecoli/ecoli.csv"
ecoli=pd.read_csv(path)

ecoli.shape
ecoli.head(3)

# drop column 'sequence name'
ecoli = ecoli.drop('sequence_name',axis=1)

ecoli.head(3)

# check for singularities
ecoli.lip.value_counts()
326/len(ecoli)

ecoli.chg.value_counts()
335/len(ecoli)

# check the distribution of Y-classes
ecoli.lsp.value_counts()

# shuffle the data since Y is grouped
ecoli = ecoli.sample(frac=1)

# perform EDA

# split the data into train and test
trainx,testx,trainy,testy=train_test_split(ecoli.drop('lsp',axis=1),
                                           ecoli.lsp,
                                           test_size=0.2)
print("trainx={},trainy={},testx={},testy={}".format(trainx.shape,trainy.shape,testx.shape,testy.shape))

# There are 2 DT models
# i) Entropy model
# ii) Gini model

# Entropy model - without HPT
m1=DecisionTreeClassifier(criterion="entropy").fit(trainx,trainy)

# plot the decision tree
features=list(ecoli.columns)
features.remove('lsp')
classes = ecoli.lsp.unique()

# create the tree
tree.export_graphviz(m1,'m1tree.dot',filled=True,rounded=True,feature_names=features,class_names=classes)

# convert dot to image file
check_call(['dot', '-Tpng', 'm1tree.dot','-o','m1tree.png'])
Image(filename='m1tree.png')

# predictions
p1 = m1.predict(testx)

# confusion matrix / classification report / accuracy score

# accuracy score
accuracy_score(testy,p1)

# confusion matrix
# confusion_matrix(testy,p1)
df=pd.DataFrame({'actual':testy,'predicted':p1})
pd.crosstab(df.actual,df.predicted,margins=True)
print(classification_report(testy,p1))

# important features
m1.feature_importances_

# create dataframe to store the feature name and their scores
# higher score = high significance
impf=pd.DataFrame({'features':trainx.columns,
                   'score':m1.feature_importances_})

# sort the data by scores in decreasing order 
impf.sort_values('score',ascending=False,inplace=True)

# plot the significant features
sns.barplot(x=impf.score,y=impf.features)
plt.title('Decision Tree - Significant Features')
plt.xlabel('Score')
plt.ylabel('Features')

# Decision Tree pruning
dt_path = m1.cost_complexity_pruning_path(trainx,trainy)

# cost complexity parameter values
ccp_alphas = dt_path.ccp_alphas

# find the best ccp_alpha value
results = []
for cp in ccp_alphas:
    m = DecisionTreeClassifier(ccp_alpha=cp).fit(trainx,trainy)
    results.append(m)

# calculate the Accuracy scores for train and test data
trg_score = [r.score(trainx,trainy) for r in results]
test_score = [r.score(testx,testy) for r in results]

# plot the scores
fig,ax = plt.subplots()
ax.plot(ccp_alphas,trg_score,marker='o',label='train',drawstyle='steps-post')
ax.plot(ccp_alphas,test_score,marker='o',label='test',drawstyle='steps-post')
ax.set_xlabel("CCP alpha")
ax.set_ylabel("Accuracy")
ax.set_title("CCP Alpha vs Accuracy")
ax.legend()

# based on the graph, the best ccp_alpha = 0.023
# build model with this ccp_alpha value
# for better results, experiment with the ccp_alpha values

m1_1 = DecisionTreeClassifier(criterion="entropy",
                              ccp_alpha=0.007).fit(trainx,trainy)

p1_1 = m1_1.predict(testx)

df1_1=pd.DataFrame({'actual':testy,'predicted':p1_1})
pd.crosstab(df1_1.actual,df1_1.predicted,margins=True)
print(classification_report(testy,p1_1))


# 2) Entropy model with Hyper-parameter tuning
m2=DecisionTreeClassifier(criterion="entropy",
                          max_depth=4,
                          min_samples_leaf=2).fit(trainx,trainy)
p2=m2.predict(testx)
df2=pd.DataFrame({'actual':testy,'predicted':p2})
pd.crosstab(df2.actual,df2.predicted,margins=True)
print(classification_report(testy,p2))

# feature selection - Method 2 - RFE (Recusrive Feature Elimination)
cols = list(testx.columns)

# specify the number of significant features you want from the model
features=3
rfe = RFE(m1,features).fit(testx,testy)
support = rfe.support_
ranking = rfe.ranking_

# store the results in dataframe
df_rfe = pd.DataFrame({'feature':cols,
                       'support':support,
                       'rank':ranking})

# sort the dataframe by 'rank'
df_rfe.sort_values('rank',ascending=True,inplace=True)

print(df_rfe)


# assignment
# --------------------
# feature selection
# next models: m3 and m4
# criterion = "gini"
# follow the same steps as above


# ......... end of DT  ...... # 

