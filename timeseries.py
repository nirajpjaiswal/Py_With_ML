# -*- coding: utf-8 -*-
# Created on Thu Mar 25 10:04:23 2021

# Time Series
# dataset: mrf

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ADF (augmented dickey-fuller test)
from statsmodels.tsa.stattools import adfuller,acf,pacf

# ARIMA model
from statsmodels.tsa.arima_model import ARIMA

# LJung-Box test
import statsmodels.api as sm

# read the input file
path="F:/work/2 mypresentation/2 ml/2 algorithms/4 time series/r/mrf/mrf.csv"
stock=pd.read_csv(path)
stock

# take the column to predict
mystock = stock[['Close Price']]
mystock


# function to check data stationarity
def checkStationarity(data):
    
    # adfuller() returns the following:
    # test statistic, pvalue, lags, total observations
    
    pval = adfuller(data)[1]
    
    if pval > 0.05:
        msg="pvalue = {}. Data not stationary".format(pval)
    else:
        msg="pvalue = {}. Data stationary".format(pval)
    
    return(msg)

# check for data stationarity
checkStationarity(mystock)

# since data is not stationary, it has to be made stationary by taking a difference

diff_mystock = mystock - mystock.shift()

print("Before deleting NA. count = ", len(diff_mystock))

# drop NA from the dataset
diff_mystock.dropna(inplace=True)

print("After deleting NA. count = ", len(diff_mystock))

# verify the drop
print(diff_mystock)

# check for data stationairty
checkStationarity(diff_mystock)

# data has now become stationary

# plot the actual and stationary data
plt.subplot(121)
plt.plot(mystock,color='red')
plt.title('Actual Close Price')

plt.subplot(122)
plt.plot(diff_mystock,color='green')
plt.title('Differenced Close Price')

# Plot the Correlogram to identify the p and q (lags for AR and MA process)

# get the PACF and ACF Lag values
lags_pacf = pacf(diff_mystock, nlags=20)
lags_acf = acf(diff_mystock, nlags=20)

# PACF -> to identify p
plt.subplot(121)
plt.plot(lags_pacf)
plt.axhline(y=0,linestyle="-",color="gray")
plt.axhline(y=-1.96/np.sqrt(len(diff_mystock)),linestyle="--",color='red')
plt.axhline(y=1.96/np.sqrt(len(diff_mystock)),linestyle="--",color='red')
plt.title("PACF")
plt.xlabel("Lags")
plt.ylabel("Correlation")

# ACF -> to identify q
plt.subplot(122)
plt.plot(lags_acf)
plt.axhline(y=0,linestyle="-",color="gray")
plt.axhline(y=-1.96/np.sqrt(len(diff_mystock)),linestyle="--",color='red')
plt.axhline(y=1.96/np.sqrt(len(diff_mystock)),linestyle="--",color='red')
plt.title("ACF")
plt.xlabel("Lags")
plt.ylabel("Correlation")

p=0; q=0; d=0

# Build the ARIMA model
m1 = ARIMA(diff_mystock,order=(p,d,q)).fit(disp=0)
m1.summary()

plt.hist(m1.resid)
plt.title("ARIMA model residuals")
# LJung-Box test to check the model goodness
# H0: residuals are independently distributed
# H1: residuals are not independently distributed
pvalue = sm.stats.acorr_ljungbox(m1.resid,lags=[1])[1]
if pvalue > 0.05:
    print("FTR H0. Residuals are independently distributed")
else:
    print("Reject H0. Residuals are not independently distributed")

# forecast for the next 12 months
f1 = m1.forecast(steps=12)

# the actual forecasted values are the first set of values in the output
forecasts = f1[0]
print(forecasts)
len(forecasts)















































































