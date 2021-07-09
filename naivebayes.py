# -*- coding: utf-8 -*-
# naive bayes classifier
# dataset: social network ads


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,classification_report

path="F:/aegis/4 ml/dataset/supervised/classification/socialnetworkad/Social_Network_Ads.csv"

data=pd.read_csv(path)

print(data)

# remove user Id from dataset
data = data.drop('User ID',axis=1)
data

# change 'gender' into a numeric format
data['gender'] = 0
data['gender'][data.Gender == "Female"] = 1
data = data.drop('Gender',axis=1)
data

# perform EDA


# convert the data into a standard scale (you can also convert to a minmax scale)
data_std = data.copy()
sc = StandardScaler()
vals = sc.fit_transform(data_std.iloc[:,:])
data_std.iloc[:,:] = vals

# restore the actual Y-value
data_std.Purchased = data.Purchased
data_std

# split the data into train and test
trainx,testx,trainy,testy=train_test_split(
    data_std.drop('Purchased',axis=1),
    data_std.Purchased,
    test_size=0.2)

print(trainx.shape,trainy.shape,testx.shape,testy.shape)


# build the model and predict
m1 = GaussianNB().fit(trainx,trainy)
p1 = m1.predict(testx)

# accuracy score
accuracy_score(testy,p1)

# confusion matrix
df=pd.DataFrame({'actual':testy,'pred':p1})
pd.crosstab(df.actual,df.pred,margins=True)

# classification report
print(classification_report(testy,p1))

# Exercise: Build a NB model on the actual train data 
# compare both the results













