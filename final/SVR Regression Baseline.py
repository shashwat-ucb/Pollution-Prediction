#!/usr/bin/env python
# coding: utf-8

# ### Nonlinear Regression - Persistent Model Baseline
# Predicts a single time step into the future using past time steps of various sequence lengths. An example is using 4 past hours to future 1 future hour.

# Imports

# In[1]:


from math import sqrt
from numpy import concatenate
from matplotlib import pyplot as plt
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import numpy as np
import pandas as pd
import time


# In[25]:


# The following data prep is based on a tutorial by Dr. Jason Brownlee
# found here: https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/


# In[26]:


# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# ### Initial Preprocessing

# In[27]:


from pandas import read_csv
#data = pd.read_csv('../data/UCI_2010_2014.csv')
#data = data.drop('No', axis=1, inplace=False)


# In[28]:


#data = data.drop(['year','month','day','hour'], axis=1, inplace=False)


# In[29]:


#data.columns = ['pollution', 'dew', 'temp', 'press', 'wnd_dir', 'wnd_spd', 'snow', 'rain']


# In[30]:


# mark all NA values with 0
#data['pollution'].fillna(0, inplace=True)
# drop the first 24 hours
#data = data[24:]
# summarize first 5 rows
#print(data.head(5))
# save to file
#data.to_csv('pollution.csv')


# In[31]:


# load dataset
dataset = pd.read_csv('pollution.csv', header=0, index_col=0)
values = dataset.values
# integer encode direction
encoder = LabelEncoder()
values[:,4] = encoder.fit_transform(values[:,4])
# ensure all data is float
values = values.astype('float32')
# frame as supervised learning -
####### Can change t_input timesteps here ##########  ### I changed it to time lag = 4
reframed = series_to_supervised(values, 1, 1)
# drop columns we don't want to predict
reframed.drop(reframed.columns[[-7,-6,-5,-4,-3,-2,-1]], axis=1, inplace=True)
print(reframed.head())

# split into train and test sets
values = reframed.values
#values = scaled
scaler = MinMaxScaler(feature_range=(0, 1), copy=True)
scaled_features = scaler.fit_transform(values[:,:-1])
scaled_label = scaler.fit_transform(values[:,-1].reshape(-1,1))
values = np.column_stack((scaled_features, scaled_label))

n_train_hours = 365 * 24 + (365 * 48)
train = values[:n_train_hours, :]
test = values[n_train_hours:-365*24, :]
# split into input and outputs
# features take all values except the var1
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]


# ### Nonlinear Regression Analysis
# Support Vector Regression (SVR)

# #### Fit to train set and predict on train, calculate errors

# In[32]:


from sklearn.svm import SVR

x = train_X
y = train_y

regr = SVR(C = 2.0, epsilon = 0.1, kernel = 'rbf', gamma = 0.5, 
           tol = 0.001, verbose=False, shrinking=True, max_iter = 10000)

regr.fit(x, y)
data_pred = regr.predict(x)
y_pred = scaler.inverse_transform(data_pred.reshape(-1,1))
y_inv = scaler.inverse_transform(y.reshape(-1,1))

mse = mean_squared_error(y_inv, y_pred)
rmse = np.sqrt(mse)
print('Mean Squared Error: {:.4f}'.format(mse))
print('Root Mean Squared Error: {:.4f}'.format(rmse))

print('Variance score: {:2f}'.format(r2_score(y_inv, y_pred)))


# ### Plot Predictions vs. Actual

# In[33]:


def plot_preds_actual(preds, actual):
    fig, ax = plt.subplots(figsize=(17,8))
    ax.plot(preds, color='red', label='Predicted data')
    ax.plot(actual, color='green', label='True data')
    ax.set_xlabel('Hourly Timestep in First Month of Predicted Year', fontsize=16)
    ax.set_ylabel('Pollution [pm2.5]', fontsize=16)
    ax.set_title('Nonlinear Regression using SVR on Test set', fontsize=16)
    ax.legend()
    plt.show()


# In[34]:


plot_preds_actual(y_pred[:24*31*1,], y_inv[:24*31*1,])


# ### Predict on test/dev sets and Calculate errors

# In[35]:


def run_test_nonlinear_reg(x, y):
    data_pred = regr.predict(x)
    y_pred = scaler.inverse_transform(data_pred.reshape(-1,1))
    y_inv = scaler.inverse_transform(y.reshape(-1,1))

    mse = mean_squared_error(y_inv, y_pred)
    rmse = np.sqrt(mse)
    print('Mean Squared Error: {:.4f}'.format(mse))
    print('Root Mean Squared Error: {:.4f}'.format(rmse))

    #Calculate R^2 (regression score function)
    print('Variance score: {:2f}'.format(r2_score(y_inv, y_pred)))
    return y_pred, y_inv


# In[36]:


y_pred, y_inv = run_test_nonlinear_reg(test_X, test_y)


# In[37]:


plot_preds_actual(y_pred[:24*31*1,], y_inv[:24*31*1,])


# In[38]:


print('Root Mean Squared Error: {:.4f}'.format(rmse))

#Calculate R^2 (regression score function)
print('Variance score: {:2f}'.format(r2_score(y_inv, y_pred)))

