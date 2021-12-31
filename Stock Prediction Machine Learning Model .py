#!/usr/bin/env python
# coding: utf-8

# In[60]:


import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader as web
import pandas as pd
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout
import pandas_datareader as pdr

import datetime


# In[ ]:



company = "FB"
start = dt.datetime(2012, 1, 1)
end = dt.datetime(2020, 1, 1)
data = web.DataReader(company, "yahoo", start, end)

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))
prediction_days = 60
xtrain = []
ytrain = []
for x in range(prediction_days, len(scaled_data)):
    xtrain.append(scaled_data[x - prediction_days: x, 0])
    ytrain.append(scaled_data[x, 0])
xtrain, ytrain = np.array(xtrain), np.array(ytrain)
xtrain = np.reshape(xtrain, (xtrain.shape[0], xtrain.shape[1], 1))

model = Sequential()
model.add(LSTM(units = 50, return_sequences= True, input_shape = (xtrain.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))  # next price prediction
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(xtrain, ytrain, epochs=25, batch_size=32)
test_start = dt.datetime(2020, 1, 1)
test_end = dt.datetime.now()

test_data = web.DataReader(company, 'yahoo', test_start)
print(test_data)
actual_prices = test_data['Close'].values
total_dataset = pd.concat((data['Close'], test_data['Close']), axis=0)
model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values
model_inputs = model_inputs.reshape(-1,1)
model_inputs = scaler.transform(model_inputs)

xtest =[]
for x in range(prediction_days, len(model_inputs)):
    xtest.append(model_inputs[x-prediction_days:x, 0])
xtest = np.array(xtest)
xtest = np.reshape(xtest, (xtest.shape[0], xtest.shape[1], 1))
predicted_prices = model.predict(xtest)
predicted_prices = scaler.inverse_transform(predicted_prices)


# In[ ]:


plt.plot(actual_prices, color="black", label = f"Actual {company} Price")
plt.plot(predicted_prices, color="green", label =f"Predicted {company} Price")
plt.title(f"Actual {company} Share Price")
plt.xlabel("Time")
plt.ylabel(f"Predicted {company} Share Price")
plt.legend()
plt.show()


# In[ ]:


real_data = [model_inputs[len(model_inputs) + 1 - prediction_days: len(model_inputs+1),0]]
real_data = np.array(real_data)
real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1],1))
prediction = model.predict(real_data)
prediction = scaler.inverse.transform(prediction)
print(f"Prediction: {prediction}")


# In[ ]:





# In[ ]:





# In[ ]:




