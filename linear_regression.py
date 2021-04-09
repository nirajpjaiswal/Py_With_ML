# -*- coding: utf-8 -*-
# Created on Thu Mar 11 09:24:38 2021

# import libraries
import pandas as pd
import numpy as np

# split data and cross-validation
from sklearn.model_selection import train_test_split, KFold,cross_val_score

# OLS library for linear regression
import statsmodels.api as sm

# visualisation
import pylab
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import mean_squared_error

import scipy.stats as stats 

# VIF
from statsmodels.stats.outliers_influence import variance_inflation_factor

# feature selection for regression
from sklearn.feature_selection import f_regression


# read the dataset
path="F:/aegis/4 ml/dataset/supervised/regression/protein/prot.csv"
prot=pd.read_csv(path)

prot.head(3)

# to change the display settings (to get full view)
pd.set_option('display.expand_frame_repr',False)

# remove the unwanted columns
prot = prot.drop(['Unnamed: 0','X','X.1','s','t','u'],axis=1)
prot.columns

# dimension (R,C)
prot.shape
prot.shape[0] # total rows
prot.shape[1] # total columns

# head/tail
prot.head(2)
prot.tail(2)

# data summary
prot.describe()

# from summary, we see column 'ssp' has all the measures as 0. so we can remove it from the dataset
prot = prot.drop('ssp',axis=1)

# check for NULLs
prot.isnull().sum()

print(prot)

prot[['tot_surf_area','npe_area']][prot.npe_area.isnull()]

# impute 'npe_area' as tot_surf_area/3 where npe_area = NULL
ndx = prot.npe_area[prot.npe_area.isnull()].index

prot.npe_area[prot.npe_area.isnull()] = prot.tot_surf_area/3

# verify the changes and check for Nulls again
prot[['tot_surf_area','npe_area']][prot.index.isin(ndx)] 
prot.isnull().sum()

prot.dtypes

# check for 0
prot[prot==0].count()

# impute 'tot_surf_area' = npe_area * 3 where tot_surf_area = 0

prot[['tot_surf_area','npe_area']][prot.tot_surf_area==0]
ndx=prot[['tot_surf_area','npe_area']][prot.tot_surf_area==0].index

# update
prot.tot_surf_area[prot.tot_surf_area==0] = prot.npe_area*3

# verify the change
prot[['tot_surf_area','npe_area']][prot.index.isin(ndx)]

# check for 0
prot[prot==0].count()

prot[['RMSD','fa_enppr']][prot.RMSD > 0].head(50).sort_values('RMSD')

# since the Y-var has 0, drop these records
ndx = prot[prot.RMSD==0].index
len(ndx)

print("before dropping rows, shape = ", prot.shape)
prot = prot.drop(ndx,axis=0)
print("after dropping rows, shape = ", prot.shape)

prot[prot==0].count()


# check the distibution and outliers of features
cols = list(prot.columns)
cols.remove('RMSD')

len(cols)

# outliers
prot.boxplot('tot_surf_area',vert=False)

nrow=4; ncol=2; npos=1

# whis=<n> indicates the IQR formula. can override the default 1.5 to any value to suppress the outliers from appearing on the boxplot
fig = plt.figure()
for c in cols:
    fig.add_subplot(nrow,ncol,npos)
    prot.boxplot(c,vert=False,whis=5)
    npos+=1

# check distribution
nrow=4; ncol=2; npos=1
fig=plt.figure()
for c in cols:
    fig.add_subplot(nrow,ncol,npos)
    sns.distplot(prot[c])
    npos+=1

# Agistino-Pearson test for normality
# H0: normal distribution
# H1: not a normal distribution

from scipy.stats import normaltest

# create a k-v pair to store column names and its corresponding distribution type (Normal/Not Normal)
aptest = {}

for c in cols:
    tstat,pval = normaltest(prot[c])
    if pval < 0.05:
        aptest[c] = "Not Normal"
    else:
        aptest[c] = "Normal"

print(aptest)

# correlation matrix - take only the lower triangle. 
# then plot the heatmap to check collinearity
cor = prot[cols].corr()
cor = np.tril(cor)
sns.heatmap(cor,xticklabels=cols,yticklabels=cols,
            vmin=-1,vmax=1,annot=True,square=False)
plt.title("Correlation Matrix")

# based on the matrix, there are some correlated variables that needs to be removed

# for linear regression, data types have to be numeric
prot.dtypes

# split the data into train/test
# trainx/trainy, testx/testy

trainx,testx,trainy,testy = train_test_split(prot.drop('RMSD',axis=1),
                                             prot['RMSD'],
                                             test_size=0.3 )

print("trainx={},trainy={},testx={},testy={}".format(trainx.shape,trainy.shape,testx.shape,testy.shape))

'''
train,test = train_test_split(prot,test_size=0.3)
train.shape
test.shape
trainx = train.drop('RMSD',axis=1)
trainy = train['RMSD']
testx = test.drop('RMSD',axis=1)
testy = test['RMSD']
'''

# Add a constant term to the trainx and testx
# this will ensure that the model summary has the 'intercept' term displayed

trainx = sm.add_constant(trainx)
testx = sm.add_constant(testx)

# Linear Regression model building
# OLS -> Ordinary Least Square method
m1=sm.OLS(trainy,trainx).fit()

# summarise the model
m1.summary()


# validation of LR assumptions
# i) mean of reisduals = 0
print(m1.resid.mean())

'''
m1.rsquared
m1.rsquared_adj
dir(m1)
'''

# ii) residuals have constant variance (homoscedasticity)
# plot the graph
# lowess->locally weighted scatterplot smoothing
yhat = m1.predict(trainx)
sns.set(style="whitegrid")
sns.residplot(m1.resid,yhat,lowess=True,color='g')

# based on the graph, the model is heteroscedastic

# bruesch-pagan test against heteroscedasticty
import statsmodels.stats.api as sms

# H0: homoscedasticity
# H1: heteroscedasticity

# return value of breusch pagan test
# lagrange_multiplier, pvalue, fscore, fp-value

# parameters: [residuals, x-array]
pval = sms.het_breuschpagan(m1.resid,m1.model.exog)[1]

if pval < 0.05:
    print("Reject H0. Model is Heteroscedastic")
else:
    print("FTR H0. Model is Homoscedastic")
    

# iii) Reasiduals have a normal distribution
stats.probplot(m1.resid, dist='norm', plot=pylab)
pylab.show()
    
# iv) rows > columns
prot.shape    


# k-Fold Cross-Validation    

folds=5    
cv_mse = []

X=trainx.values
Y=trainy.values

kf= KFold(folds)
# kf.get_n_splits(X)

for train_index,test_index in kf.split(X):
    cv_trainx,cv_trainy = X[train_index],Y[train_index]
    cv_testx, cv_testy =  X[test_index],Y[test_index]
    
    # build model on cv_train and predict on cv_test
    m = sm.OLS(cv_trainy,cv_trainx).fit()
    p = m.predict(cv_testx)
    
    # store MSE in the list for each model
    cv_mse.append(np.round(mean_squared_error(cv_testy,p),3))
    
cv_mse

# mean MSE of k-Fold CV
np.mean(cv_mse) 


# prediction on the test data
p1 = m1.predict(testx)

# MSE of model 1
mse1 = round(mean_squared_error(testy,p1),3)

# compare the train and test errors
print("Training MSE = {}, Testing MSE = {}".format(np.mean(cv_mse),mse1))

# store the actual and predicted data for comparison
df = pd.DataFrame({'actual':testy,'predicted':round(p1,3)})
df.head(30)

# plot the actual and predicted values
ax1=sns.distplot(testy,hist=False,color='blue',label='Actual')
sns.distplot(p1,hist=False,color='red',label='Predicted',ax=ax1)


# other considerations
# VIF (Variance Inflation Factor)
vif = pd.DataFrame()

vif["inflation"] = [variance_inflation_factor(trainx.values,i) 
for i in range(trainx.shape[1])]

vif['features'] = list(trainx.columns)

# consider VIF > 10 to remove collinearity
# sometimes VIF > 6 is also taken as the cutoff

# build the next model with the significant features and compare the models for the RMSE


# Box Cox transformation
from scipy.stats import boxcox

# transform data into BoxCox transformed data
d = np.round(np.random.uniform(10,200,100),2)
bc_values,lamda = boxcox(d)
bc_values
lamda

bc1 = bc_values[0]
bc1
# compare the original and transformed data
print('actual={},BoxCox={}'.format(d[0],round(bc1,2)))

# transform BoxCox data into its original form
orig = np.exp(np.log(bc1 * lamda+1)/lamda)
orig

################################################################################

# 2) Decision Tree Regression
from sklearn.tree import DecisionTreeRegressor

m2 = DecisionTreeRegressor(criterion="mse",max_depth=5).fit(trainx,trainy)
p2 = m2.predict(testx)
mse2 = round(mean_squared_error(testy,p2),3)
mse2

# 3) Random Forest Regression
from sklearn.ensemble import RandomForestRegressor

# without any hyper-parameter tuning. Add, whatever is required
m3 = RandomForestRegressor().fit(trainx,trainy)
p3 = m3.predict(testx)
mse3 = round(mean_squared_error(testy,p3),3)
mse3

# 4) kNN Regression
from sklearn import neighbors,preprocessing

# standardize the data
trainx_std = trainx.copy()
testx_std = testx.copy()

minmax=preprocessing.MinMaxScaler()

# scale the train data
sc=minmax.fit_transform(trainx_std.iloc[:,:])
trainx_std.iloc[:,:] = sc

# scale the test data
sc=minmax.fit_transform(testx_std.iloc[:,:])
testx_std.iloc[:,:] = sc


trainx_std.head(3)

# for regression, take NN between 3-11 (include even numbers)
nn = range(3,12)
list(nn)

mse_cv = []

for k in nn:
    m=neighbors.KNeighborsRegressor(n_neighbors=k).fit(trainx_std,trainy)
    err=cross_val_score(m,trainx_std,trainy,cv=5,scoring='neg_mean_squared_error')
    err = np.round(np.mean(err),3)
    mse_cv.append(err)

# the MSE scores are all in -ve numbers. Convert to +ve
print(mse_cv)

# convert all -ve to +ve numbers using lambda()
mse_cv = list(map(lambda x:abs(x),mse_cv))

# select the lowest MSE score and its corresponding K-value
bestk = nn[mse_cv.index(min(mse_cv))]

# Build the kNN model with the bestK
m4 = neighbors.KNeighborsRegressor(n_neighbors=bestk).fit(trainx_std,trainy)
p4 = m4.predict(testx_std)
mse4 = mean_squared_error(testy,p4)
mse4

# SVM Regression
from sklearn import svm,preprocessing

# since data is already scaled, we can directly use the kernels on the scaled data

kernels = ['linear', 'rbf', 'poly', 'sigmoid']
kernels

# determine the R-square for each regression model, for each Kernel
for k in kernels:
    m = svm.SVR(kernel = k).fit(trainx_std,trainy)
    rsq = m.score(testx_std,testy)
    print("Kernel = {}, R-Square = {}".format(k, rsq))

# based on the R-square, it is clear that Sigmoid kernel does not fit the data properly. 


# SVM regression
def svmRegression(ker,trainx,trainy,testx,testy,bestc=1,bestg='scale'):
    model=svm.SVR(kernel=ker,C=bestc,gamma=bestg).fit(trainx,trainy)
    pred = model.predict(testx)
    mse = mean_squared_error(testy,pred)
    return(pred, mse)

m_mse=[]; p5=[]

# run the regression model for each Kernel
for k in kernels:
    pred,mseval=svmRegression(k,trainx_std,trainy,testx_std,testy)  
    p5.append(pred)
    m_mse.append(round(mseval,3))



# create a dataframe to store the MSE's
df_mse=pd.DataFrame({'LR':[mse1], 'DT':[mse2],
                      'RF':[mse3], 'kNN':[mse4],
                     'SVM-linear': [m_mse[0]],
                     'SVM-rbf': [m_mse[1]],
                     'SVM-poly': [m_mse[2]],
                     'SVM-sig': [m_mse[3]] 
                     })

print(df_mse)

# transpose the data
df_mse.T

    
# create a dataframe to store the Actual / Predicted values

df_vals = pd.DataFrame({'actual':testy,
                        'LR':p1,
                       'DT':p2,
                       'RF':p3,
                       'kNN':p4,
                     'SVM-linear': p5[0],
                     'SVM-rbf':  p5[1],
                     'SVM-poly': p5[2],
                     'SVM-sig': p5[3] } )

df_vals
    
# visualise the Actual vs Predicted Data
def showPlot(act,pred,model):
    ax1=sns.distplot(act,hist=False,color='r',label='Actual')
    sns.distplot(pred,hist=False,color='b',label='Predicted',ax=ax1)
    plt.title('Actual vs Predicted. Model = ' + model)
    

# function call to display the chart
showPlot(df_vals.actual,df_vals['RF'],'Random Forest')  
    
    
    
    
    
    
    
    
    




    
    
    
    
    
    
    
    
    
    
    

    
    







 



























































