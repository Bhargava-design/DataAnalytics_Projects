#!/usr/bin/env python
# coding: utf-8

# In[125]:


# Install the tiingo package if you haven't already
# pip install tiingo

# Import necessary libraries
get_ipython().system('pip install tensorflow')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from tiingo import TiingoClient


# In[126]:


# Set your Tiingo API key
api_key = "fcd1dda20ddf992ad55de36f2b3fe86ce71539ea"


# In[128]:


# Initialize the Tiingo client
config = {
    'session': True,
    'api_key': api_key
}
client = TiingoClient(config)


# In[129]:


# Fetch TSLA stock data
ticker = 'TSLA'
start_date = '2021-01-01'
end_date = '2021-12-31'
df = client.get_dataframe(tickers=ticker, startDate=start_date, endDate=end_date)


# In[130]:


# Save the data to a CSV file
df.to_csv(f'{ticker}.csv')


# In[131]:


# Display the first few rows of the dataset
print(df.head())


# In[132]:


# Extract the 'adjClose' column and store it in df1
df1 = df['adjClose']


# In[133]:


# Plot the data
plt.plot(df1)
plt.show()


# In[134]:


# Reshape and scale the data using MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
df1 = scaler.fit_transform(np.array(df1).reshape(-1, 1))
print(df1)


# In[135]:


# Split the dataset into training and test sets
training_size = int(len(df1) * 0.65)
train_data, test_data = df1[0:training_size, :], df1[training_size:len(df1), :]


# In[136]:


# Display the sizes of the training and test sets
print("Training data size:", len(train_data))
print("Test data size:", len(test_data))


# In[137]:


# Create a dataset for training and testing
time_step = 100  # You can adjust this value
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)


# In[138]:


print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)


# In[139]:


# Reshape input to be [samples, time steps, features] which is required for LSTM
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

# Ensure that X_test has the same number of time steps (columns) as X_train
# You can adjust it according to your requirements
X_test = X_test.reshape(X_test.shape[0], X_train.shape[1], 1)


# In[140]:


get_ipython().system('pip install tensorflow')
from tensorflow.keras.models import Sequential


# In[148]:


from tensorflow.keras.layers import LSTM

# Create the Stacked LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
model.add(LSTM(50, return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.summary()


# In[150]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# Generate random time series data
data = [x for x in range(1, 301)]
data = np.array(data).reshape(-1, 1)

# Normalize the data
scaler = MinMaxScaler()
data = scaler.fit_transform(data)

# Split the data into training and test sets
train_size = int(len(data) * 0.65)
test_size = len(data) - train_size
train_data, test_data = data[0:train_size, :], data[train_size:len(data), :]

# Create a dataset function
def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - time_step):
        a = dataset[i:(i + time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)

time_step = 100  # You can adjust this value
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Create and compile the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
model.add(LSTM(50, return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

# Train the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=64, verbose=1)

# Make predictions
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Inverse transform predictions
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)

# Calculate RMSE
train_rmse = np.sqrt(mean_squared_error(scaler.inverse_transform(y_train), train_predict))
test_rmse = np.sqrt(mean_squared_error(scaler.inverse_transform(y_test), test_predict))
print("Train RMSE:", train_rmse)
print("Test RMSE:", test_rmse)

# Plot the results
look_back = time_step
trainPredictPlot = np.empty_like(data)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_predict) + look_back, :] = train_predict

testPredictPlot = np.empty_like(data)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(train_predict) + (look_back * 2) + 1:len(data) - 1, :] = test_predict

plt.plot(scaler.inverse_transform(data))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()


# In[151]:


import tensorflow as tf
tf.__version__

# Perform the prediction and check performance metrics
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Transform back to the original form
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)

# Calculate RMSE performance metrics
import math
from sklearn.metrics import mean_squared_error

train_rmse = math.sqrt(mean_squared_error(scaler.inverse_transform(y_train), train_predict))
test_rmse = math.sqrt(mean_squared_error(scaler.inverse_transform(y_test), test_predict))

print("Train RMSE:", train_rmse)
print("Test RMSE:", test_rmse)

# Plotting 
# Shift train predictions for plotting
look_back = 100
trainPredictPlot = numpy.empty_like(df1)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[look_back:len(train_predict) + look_back, :] = train_predict

# Shift test predictions for plotting
testPredictPlot = numpy.empty_like(df1)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(train_predict) + (look_back * 2) + 1:len(df1) - 1, :] = test_predict

# Plot baseline and predictions
plt.plot(scaler.inverse_transform(df1))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()


# In[152]:


y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)


# In[157]:


import tensorflow as tf
tf.__version__

# Perform the prediction and check performance metrics
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Transform back to the original form
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)

# Reshape y_train and y_test to be 2D arrays
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

# Calculate RMSE performance metrics
import math
from sklearn.metrics import mean_squared_error

train_rmse = math.sqrt(mean_squared_error(y_train, train_predict))
test_rmse = math.sqrt(mean_squared_error(y_test, test_predict))

print("Train RMSE:", train_rmse)
print("Test RMSE:", test_rmse)

# Plotting 
# Shift train predictions for plotting
look_back = 100
trainPredictPlot = numpy.empty_like(df1)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[look_back:len(train_predict) + look_back, :] = train_predict

# Shift test predictions for plotting
testPredictPlot = numpy.empty_like(df1)
testPredictPlot[:, :] = numpy.nan

# Calculate the start and end indices for plotting
start_index = len(train_predict) + look_back  # Adjusted for time step
end_index = start_index + len(test_predict)

testPredictPlot[start_index:end_index, :] = test_predict

# Plot baseline and predictions
plt.plot(scaler.inverse_transform(df1))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()



# In[158]:


len(test_data)


# In[159]:


x_input=test_data[341:].reshape(1,-1)
x_input.shape


# In[160]:


temp_input=list(x_input)
temp_input=temp_input[0].tolist()


# In[161]:


temp_input


# In[164]:


# demonstrate prediction for the next 10 days
lst_output = []
n_steps = 100
i = 0
while i < 10:  # Predict the next 10 days
    if len(temp_input) > 100:
        x_input = np.array(temp_input[1:])
        x_input = x_input.reshape(1, -1)
        x_input = x_input.reshape((1, n_steps, 1))
        yhat = model.predict(x_input, verbose=0)
        temp_input.extend(yhat[0].tolist())
        temp_input = temp_input[1:]
        lst_output.extend(yhat.tolist())
        i += 1
    else:
        if len(temp_input) > 0:  # Ensure that temp_input is not empty
            x_input = np.array(temp_input).reshape(1, -1)
            x_input = x_input.reshape((1, n_steps, 1))
            yhat = model.predict(x_input, verbose=0)
            temp_input.extend(yhat[0].tolist())
            lst_output.extend(yhat.tolist())
        else:
            break  # Exit the loop if temp_input is empty

# Print the predicted values for the next 10 days
print("Predicted Values for the Next 10 Days:")
print(lst_output)


# In[165]:


day_new=np.arange(1,101)
day_pred=np.arange(101,131)


# In[166]:


import matplotlib.pyplot as plt


# In[167]:


len(df1)


# In[174]:


# Define the indices for slicing df1
start_index = 1158
end_index = start_index + len(lst_output)

# Check if the slicing indices are valid
if start_index < len(df1) and end_index <= len(df1):
    # Inverse transform and plot the original data
    plt.plot(day_new, scaler.inverse_transform(df1[start_index:end_index]))
    # Inverse transform and plot the predicted data
    plt.plot(day_pred, scaler.inverse_transform(lst_output))
    plt.show()
else:
    # Sample data
    x = np.linspace(0, 10, 100)  # Generate 100 points from 0 to 10
    y = np.sin(x)  # Calculate the sine of each point for demonstration

    # Create the plot
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, label="Sine Wave", color="blue")
    plt.title("Sample Sine Wave")
    plt.xlabel("X-Axis")
    plt.ylabel("Y-Axis")
    plt.legend()
    plt.grid(True)

# Display the plot
plt.show()


# In[177]:


df3=df1.tolist()
df3.extend(lst_output)
plt.plot(df3[12:])


# In[178]:


df3=scaler.inverse_transform(df3).tolist()


# In[179]:


plt.plot(df3)


# In[ ]:





# In[ ]:




