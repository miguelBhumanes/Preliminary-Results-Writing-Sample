###########################################################################
### 1 IMPORTING PACKAGES
import os 
os.chdir("/Users/miguelds/Desktop/Topic-MIDAS/MIDAS models")

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error as mse

import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import GRU, Dense, LSTM, Input, Layer, Flatten, Dropout, MultiHeadAttention
from tensorflow.keras import backend as K
from tensorflow.keras import regularizers

import ssl
from fredapi import Fred

###########################################################################
### 2 IMPORTING AND HANDLING DATA

# Economic data
econ_vars = pd.read_csv("econ_vars.csv", index_col="date", parse_dates=["date"])
econ_vars.drop(columns="Unnamed: 0", inplace=True)
econ_vars.sort_index(ascending=True, inplace=True) # Revert order so first entry is the oldest
econ_vars = econ_vars[econ_vars.index <= "2023-01-01"] # Only dates until last quarterly data 2022
econ_vars.shape # 44 columns, 276 months = 23 years
monthly_vars = econ_vars # renaming, since it is later going to be used together with a "daily_vars" dataframe

# Financial data
fin_vars = pd.read_csv("financial_vars.csv", index_col="date", parse_dates=["date"])
fin_vars.drop(columns="Unnamed: 0", inplace=True)
fin_vars.sort_index(ascending=True, inplace=True) # Revert order so first entry is the oldest
fin_vars = fin_vars[fin_vars.index <= "2023-01-01"] # Only dates until last quarterly data 2022
fin_vars = fin_vars[fin_vars.index >= "2000-01-01"] # Only dates after first quarterly data 2000
fin_vars.shape # 37 columns, 6000 days. Approximately 261 days per year

# Text data
text_vars = pd.read_csv("text_vars.csv", index_col="date", parse_dates=["date"])
text_vars.drop(columns="Unnamed: 0", inplace=True)
text_vars.sort_index(ascending=True, inplace=True) # Revert order so first entry is the oldest
text_vars = text_vars[text_vars.index <= "2023-01-01"] # Only dates until last quarterly data 2022
text_vars = text_vars[text_vars.index >= "2000-01-01"] # Only dates after first quarterly data 2000
text_vars.shape # 37 columns, 5999 days. Approximately 261 days per year

# Creating a single dataframe for the daily variables
daily_vars = text_vars.merge(fin_vars, left_index=True, right_index=True, how='inner')

###########################################################################
### 3 CREATING TARGET VARIABLES 

# Importing GDP and Inflation data from January 1999 to December 2022
ssl._create_default_https_context = ssl._create_unverified_context
fred = Fred(api_key="ac7fd08d9f2c9ad01c28fb92fc5b3c85")
real_gdp_data = fred.get_series('GDPC1')
real_gdp_data = real_gdp_data['1999':'2022']
cpi_data = fred.get_series('CPIAUCSL')
cpi_data = cpi_data['1999':'2022']

# Computing the yearly rates of growth, which are the target variables
gdp_growth_y = ((real_gdp_data - real_gdp_data.shift(4)) / real_gdp_data.shift(4)).dropna()
inflation_data = ((cpi_data - cpi_data.shift(12)) / cpi_data.shift(12)).dropna()

# Moving to last day of the month
gdp_growth_y.index = gdp_growth_y.index + pd.DateOffset(months=3, days=-1)
inflation_data.index = inflation_data.index + pd.DateOffset(months=1, days=-1)

# Scaling 
gdp_g_scaled = (gdp_growth_y-min(gdp_growth_y))/(max(gdp_growth_y) - min(gdp_growth_y))
inflation_scaled = (inflation_data-min(inflation_data))/(max(inflation_data) - min(inflation_data))

# Getting all the targets: GDP Growth
y_1q = gdp_g_scaled[gdp_growth_y.index>="2001-03-31"] # For 1q ahead forecasting
y_2q = gdp_g_scaled[gdp_growth_y.index>="2001-06-30"] # For 2q ahead forecasting
y_3q = gdp_g_scaled[gdp_growth_y.index>="2001-09-30"] # For 3q ahead forecasting
y_4q = gdp_g_scaled[gdp_growth_y.index>="2001-12-31"] # For 4q ahead forecasting

# Getting all the targets: CPI Inflation
p_1q = inflation_scaled[inflation_data.index>="2001-03-31"] # For 1q ahead forecasting
p_2q = inflation_scaled[inflation_data.index>="2001-06-30"] # For 2q ahead forecasting
p_3q = inflation_scaled[inflation_data.index>="2001-09-30"] # For 3q ahead forecasting
p_4q = inflation_scaled[inflation_data.index>="2001-12-31"] # For 4q ahead forecasting

###########################################################################
### 4 CREATING FEATURE MATRIX

# Scaling data, but converting it back to dataframe
scaler_xd = scaler_xm = MinMaxScaler()
daily_vars_scaled = scaler_xd.fit_transform(daily_vars)
monthly_vars_scaled = scaler_xm.fit_transform(monthly_vars)
daily_vars_scaled = pd.DataFrame(daily_vars)
daily_vars_scaled.index = daily_vars.index
monthly_vars_scaled = pd.DataFrame(monthly_vars)
monthly_vars_scaled.index = monthly_vars.index

# Creating the windows of observations (1 year. 12 observations for monthly, 259 for daily)
X_samples_monthly_forgdp = []
X_samples_daily_forgdp = []
X_samples_monthly_forinf = []
X_samples_daily_forinf = []

for index_date in (y_1q.index + pd.DateOffset(months=-3, days=0)):
    regressors_daily = daily_vars[daily_vars.index <= index_date]
    regressors_daily = regressors_daily.tail(259)
    regressors_monthly = monthly_vars[monthly_vars.index <= index_date]
    regressors_monthly = regressors_monthly.tail(12)
    X_samples_daily_forgdp.append(regressors_daily)
    X_samples_monthly_forgdp.append(regressors_monthly)

X_samples_daily_np_gdp = np.stack([df.values for df in X_samples_daily_forgdp], axis=0)
X_samples_monthly_np_gdp = np.stack([df.values for df in X_samples_monthly_forgdp], axis=0)

for index_date in (p_1q.index + pd.DateOffset(months=-3, days=0)):
    regressors_daily = daily_vars[daily_vars.index <= index_date]
    regressors_daily = regressors_daily.tail(259)
    regressors_monthly = monthly_vars[monthly_vars.index <= index_date]
    regressors_monthly = regressors_monthly.tail(12)
    X_samples_daily_forinf.append(regressors_daily)
    X_samples_monthly_forinf.append(regressors_monthly)

X_samples_daily_np_inf = np.stack([df.values for df in X_samples_daily_forinf], axis=0)
X_samples_monthly_np_inf = np.stack([df.values for df in X_samples_monthly_forinf], axis=0)

# Create the different windows for the different models 

len_gdp = X_samples_daily_np_gdp.shape[0]
len_inflation = X_samples_daily_np_inf.shape[0]

X_samples_daily_np_y1q = X_samples_daily_np_gdp
X_samples_monthly_np_y1q = X_samples_monthly_np_gdp
X_samples_daily_np_y2q = X_samples_daily_np_gdp[:(len_gdp-1)]
X_samples_monthly_np_y2q = X_samples_monthly_np_gdp[:(len_gdp)-1]
X_samples_daily_np_y3q = X_samples_daily_np_gdp[:(len_gdp-2)]
X_samples_monthly_np_y3q = X_samples_monthly_np_gdp[:(len_gdp)-2]
X_samples_daily_np_y4q = X_samples_daily_np_gdp[:(len_gdp-3)]
X_samples_monthly_np_y4q = X_samples_monthly_np_gdp[:(len_gdp)-3]

X_samples_daily_np_p1q = X_samples_daily_np_inf
X_samples_monthly_np_p1q = X_samples_monthly_np_inf
X_samples_daily_np_p2q = X_samples_daily_np_inf[:(len_inflation-3)]
X_samples_monthly_np_p2q = X_samples_monthly_np_inf[:(len_inflation)-3]
X_samples_daily_np_p3q = X_samples_daily_np_inf[:(len_inflation-6)]
X_samples_monthly_np_p3q = X_samples_monthly_np_inf[:(len_inflation)-6]
X_samples_daily_np_p4q = X_samples_daily_np_inf[:(len_inflation-9)]
X_samples_monthly_np_p4q = X_samples_monthly_np_inf[:(len_inflation)-9]

# Store the different datasets on an appropriate list

# Define a function to split the 3D numpy array data into training and test sets
def split_data_80_20(data):
    split_idx = int(0.8 * data.shape[0])
    train_data = data[:split_idx]
    test_data = data[split_idx:]
    return train_data, test_data

# Split the X_gdp daily data
X_gdp_train_d_q1, X_gdp_test_d_q1 = split_data_80_20(X_samples_daily_np_y1q)
X_gdp_train_d_q2, X_gdp_test_d_q2 = split_data_80_20(X_samples_daily_np_y2q)
X_gdp_train_d_q3, X_gdp_test_d_q3 = split_data_80_20(X_samples_daily_np_y3q)
X_gdp_train_d_q4, X_gdp_test_d_q4 = split_data_80_20(X_samples_daily_np_y4q)

# Split the X_gdp monthly data
X_gdp_train_m_q1, X_gdp_test_m_q1 = split_data_80_20(X_samples_monthly_np_y1q)
X_gdp_train_m_q2, X_gdp_test_m_q2 = split_data_80_20(X_samples_monthly_np_y2q)
X_gdp_train_m_q3, X_gdp_test_m_q3 = split_data_80_20(X_samples_monthly_np_y3q)
X_gdp_train_m_q4, X_gdp_test_m_q4 = split_data_80_20(X_samples_monthly_np_y4q)

# Split the X_inf daily data
X_inf_train_d_q1, X_inf_test_d_q1 = split_data_80_20(X_samples_daily_np_p1q)
X_inf_train_d_q2, X_inf_test_d_q2 = split_data_80_20(X_samples_daily_np_p2q)
X_inf_train_d_q3, X_inf_test_d_q3 = split_data_80_20(X_samples_daily_np_p3q)
X_inf_train_d_q4, X_inf_test_d_q4 = split_data_80_20(X_samples_daily_np_p4q)

# Split the X_inf monthly data
X_inf_train_m_q1, X_inf_test_m_q1 = split_data_80_20(X_samples_monthly_np_p1q)
X_inf_train_m_q2, X_inf_test_m_q2 = split_data_80_20(X_samples_monthly_np_p2q)
X_inf_train_m_q3, X_inf_test_m_q3 = split_data_80_20(X_samples_monthly_np_p3q)
X_inf_train_m_q4, X_inf_test_m_q4 = split_data_80_20(X_samples_monthly_np_p4q)

# Split the target variables
y1_train, y1_test = split_data_80_20(y_1q)
y2_train, y2_test = split_data_80_20(y_2q)
y3_train, y3_test = split_data_80_20(y_3q)
y4_train, y4_test = split_data_80_20(y_4q)

p1_train, p1_test = split_data_80_20(p_1q)
p2_train, p2_test = split_data_80_20(p_2q)
p3_train, p3_test = split_data_80_20(p_3q)
p4_train, p4_test = split_data_80_20(p_4q)

# Create the list of lists
data_lists = [
    [X_gdp_train_d_q1, X_gdp_train_m_q1, y1_train, X_gdp_test_d_q1, X_gdp_test_m_q1, y1_test],
    [X_gdp_train_d_q2, X_gdp_train_m_q2, y2_train, X_gdp_test_d_q2, X_gdp_test_m_q2, y2_test],
    [X_gdp_train_d_q3, X_gdp_train_m_q3, y3_train, X_gdp_test_d_q3, X_gdp_test_m_q3, y3_test],
    [X_gdp_train_d_q4, X_gdp_train_m_q4, y4_train, X_gdp_test_d_q4, X_gdp_test_m_q4, y4_test],
    [X_inf_train_d_q1, X_inf_train_m_q1, p1_train, X_inf_test_d_q1, X_inf_test_m_q1, p1_test],
    [X_inf_train_d_q2, X_inf_train_m_q2, p2_train, X_inf_test_d_q2, X_inf_test_m_q2, p2_test],
    [X_inf_train_d_q3, X_inf_train_m_q3, p3_train, X_inf_test_d_q3, X_inf_test_m_q3, p3_test],
    [X_inf_train_d_q4, X_inf_train_m_q4, p4_train, X_inf_test_d_q4, X_inf_test_m_q4, p4_test]
]



###########################################################################
### 5 DEFINING THE MIDAS CONVOLUTION

# Define the nealmon functions for computing the weights
def nealmon_m(param1, param2):
    i_values = tf.range(1, 4, dtype=tf.float32) # Will condense 3 months into a quarter
    ll = (param1 * i_values + param2 * i_values**2) - tf.math.reduce_logsumexp((param1 * i_values + param2 * i_values**2))
    nealmon_weights = tf.exp(ll)
    return nealmon_weights

def nealmon_d(param3, param4):
    i_values = tf.range(1, 22, dtype=tf.float32) # Will condense 21 days into a month
    ll = (param3 * i_values + param4 * i_values**2) - tf.math.reduce_logsumexp((param3 * i_values + param4 * i_values**2))
    nealmon_weights = tf.exp(ll)
    return nealmon_weights

# Define a custom layer for the monthly data (converts monthly to quarterly)
class CustomConv1D_m(Layer):
    def __init__(self, **kwargs):
        super(CustomConv1D_m, self).__init__(**kwargs)

    def build(self, input_shape):
        self.param1 = self.add_weight(name='param1', shape=(), initializer=tf.constant_initializer(0), trainable=True)
        self.param2 = self.add_weight(name='param2', shape=(), initializer=tf.constant_initializer(0), trainable=True)

    def call(self, inputs):
        nealmon_weights = nealmon_m(self.param1, self.param2)
        w = tf.reshape(nealmon_weights, (3, 1, 1))
        kernel = w * tf.ones((3, 281, 281), dtype=tf.float32)
        return K.conv1d(x=inputs, kernel=kernel, strides=3) # Strides = jump 3 months
    
# Define a custom layer for the daily data (Converts daily to monthly)
class CustomConv1D_d(Layer):
    def __init__(self, **kwargs):
        super(CustomConv1D_d, self).__init__(**kwargs)

    def build(self, input_shape):
        self.param3 = self.add_weight(name='param3', shape=(), initializer=tf.constant_initializer(0), trainable=True)
        self.param4 = self.add_weight(name='param4', shape=(), initializer=tf.constant_initializer(0), trainable=True)
        
    def call(self, inputs):
        nealmon_weights = nealmon_d(self.param3, self.param4)
        w = tf.reshape(nealmon_weights, (21, 1, 1))
        kernel = w * tf.ones((21, 237, 237), dtype=tf.float32)
        return K.conv1d(x=inputs, kernel=kernel, strides=21)

# Define a custom layer to aggregate the monthly data using the average
class CustomConv1D_d_simple(Layer):
    def __init__(self, **kwargs):
        super(CustomConv1D_d_simple, self).__init__(**kwargs)

    def call(self, inputs):
        nealmon_weights = nealmon_d(0,0)
        w = tf.reshape(nealmon_weights, (21, 1, 1))
        kernel = w * tf.ones((21, 237, 237), dtype=tf.float32)
        return K.conv1d(x=inputs, kernel=kernel, strides=21)
    

############################################################################
### 10 MODELS WITH NO DIMENSIONALITY REDUCTION, SPARSITY

### Train-test split for inflation and GDP 1Q ahead prediction (used to hyperparametrize the models)

# 80% train+validation (68% train 12% validation split) - 20% test

X_gdp_train_d = X_samples_daily_np_y1q[0:70]
X_gdp_train_m = X_samples_monthly_np_y1q[0:70]
X_gdp_test_d = X_samples_daily_np_y1q[70:89] 
X_gdp_test_m = X_samples_monthly_np_y1q[70:89]
y_train = y_1q[0:70]
y_test = y_1q[70:89]

X_inf_train_d = X_samples_daily_np_p1q[0:210]
X_inf_train_m = X_samples_monthly_np_p1q[0:210]
X_inf_test_d = X_samples_daily_np_p1q[210:263]
X_inf_test_m = X_samples_monthly_np_p1q[210:263]
p_train = p_1q[0:210]
p_test = p_1q[210:263]

### Defining a callback to avoid overfitting
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                  patience=3,restore_best_weights=True)

#### 10.1 ANN NAIVE

# Custom optimizer and learning rate
custom_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# Grid Search for GDP
last_val_loss = []

for hlayers in [1,2,3,4]:
    mse_sublist = []
    val_loss_sublist = []
    for nodes in [10,50,100,250]:
        mse_sub_sublist = []
        val_loss_sub_sublist = []
        for dropout_rate in [0,0.1,0.2]:
            Input_daily = Input(shape=(259,237))
            Input_monthly = Input(shape=(12,44))
            Flat1 = Flatten()(Input_daily)
            Flat2 = Flatten()(Input_monthly)
            Flat3 = tf.keras.layers.Concatenate(axis=-1)([Flat1, Flat2])
            x = Flat3
            for _ in range(hlayers):
                x = Dense(units=nodes, activation='relu')(x)
                x = Dropout(rate=dropout_rate)(x)
            Output = Dense(1, activation='sigmoid')(x)
            model = Model([Input_daily,Input_monthly], Output)

            # Model fit 
            model.compile(optimizer=custom_optimizer, loss='mean_squared_error') 
            random.seed(99)
            np.random.seed(99)
            tf.random.set_seed(99)
            history = model.fit([X_gdp_train_d,X_gdp_train_m], y_train, 
                                epochs=200, batch_size=16, validation_split=0.15,
                                callbacks=[early_stopping])
            val_loss_sub_sublist.append(history.history['val_loss'][-1])
        val_loss_sublist.append(val_loss_sub_sublist)
    last_val_loss.append(val_loss_sublist)

valloss = np.array(last_val_loss)
position_ann_naive_gdp = np.array(np.unravel_index(np.argmin(valloss),shape=valloss.shape))+1
# Attending to validation loss, best tuning is 4 layers, 10 nodes and 0 dropout
def create_ann_naive_gdp():
    Input_daily = Input(shape=(259,237))
    Input_monthly = Input(shape=(12,44))
    Flat1 = Flatten()(Input_daily)
    Flat2 = Flatten()(Input_monthly)
    Flat3 = tf.keras.layers.Concatenate(axis=-1)([Flat1, Flat2])
    x = Flat3
    for _ in range(4):
        x = Dense(units=10, activation='relu')(x)
        x = Dropout(rate=0)(x)
    Output = Dense(1, activation='sigmoid')(x)
    model = Model([Input_daily,Input_monthly], Output)
    return model

# Grid Search for Inflation
last_val_loss = []

for hlayers in [1,2,3,4]:
    mse_sublist = []
    val_loss_sublist = []
    for nodes in [10,50,100,250]:
        mse_sub_sublist = []
        val_loss_sub_sublist = []
        for dropout_rate in [0,0.1,0.2]:
            Input_daily = Input(shape=(259,237))
            Input_monthly = Input(shape=(12,44))
            Flat1 = Flatten()(Input_daily)
            Flat2 = Flatten()(Input_monthly)
            Flat3 = tf.keras.layers.Concatenate(axis=-1)([Flat1, Flat2])
            x = Flat3
            for _ in range(hlayers):
                x = Dense(units=nodes, activation='relu')(x)
                x = Dropout(rate=dropout_rate)(x)
            Output = Dense(1, activation='sigmoid')(x)
            model = Model([Input_daily,Input_monthly], Output)

            # Model fit 
            model.compile(optimizer=custom_optimizer, loss='mean_squared_error') 
            random.seed(99)
            np.random.seed(99)
            tf.random.set_seed(99)
            history = model.fit([X_inf_train_d,X_inf_train_m], p_train, 
                                epochs=200, batch_size=16, validation_split=0.15,
                                callbacks=[early_stopping])
            val_loss_sub_sublist.append(history.history['val_loss'][-1])
        val_loss_sublist.append(val_loss_sub_sublist)
    last_val_loss.append(val_loss_sublist)

valloss = np.array(last_val_loss)
position_ann_naive_inf = np.array(np.unravel_index(np.argmin(valloss),shape=valloss.shape))+1
# Attending to validation loss, best tuning is 2 layers, 10 nodes and 0% dropout
def create_ann_naive_inf():
    Input_daily = Input(shape=(259,237))
    Input_monthly = Input(shape=(12,44))
    Flat1 = Flatten()(Input_daily)
    Flat2 = Flatten()(Input_monthly)
    Flat3 = tf.keras.layers.Concatenate(axis=-1)([Flat1, Flat2])
    x = Flat3
    for _ in range(2):
        x = Dense(units=10, activation='relu')(x)
        x = Dropout(rate=0)(x)
    Output = Dense(1, activation='sigmoid')(x)
    model = Model([Input_daily,Input_monthly], Output)
    return model

#### 10.2 LSTM NAIVE

# Custom optimizer and learning rate
custom_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# Grid Search for GDP
last_val_loss = []

for hlayers in [1,2,3,4]:
    mse_sublist = []
    val_loss_sublist = []
    for nodes in [10,50,100,250]:
        mse_sub_sublist = []
        val_loss_sub_sublist = []
        for dropout_rate in [0,0.1,0.2]:

            Input_daily = Input(shape=(259,237))
            Input_monthly = Input(shape=(12,44))
            Daily_to_monthly = CustomConv1D_d_simple()(Input_daily)
            Concat = tf.keras.layers.Concatenate(axis=-1)([Input_monthly, Daily_to_monthly])
            x = Concat
            for _ in range(hlayers):
                x = LSTM(units=nodes, activation='relu', return_sequences=True)(x)
                x = Dropout(rate=dropout_rate)(x)
            x = Flatten()(x)
            x = Dense(units=nodes, activation='relu')(x)
            Output = Dense(1, activation='sigmoid')(x)
            model = Model([Input_daily,Input_monthly], Output)

            # Model fit 
            model.compile(optimizer=custom_optimizer, loss='mean_squared_error') 
            random.seed(99)
            np.random.seed(99)
            tf.random.set_seed(99)
            history = model.fit([X_gdp_train_d,X_gdp_train_m], y_train, 
                                epochs=200, batch_size=16, validation_split=0.15,
                                callbacks=[early_stopping])
            val_loss_sub_sublist.append(history.history['val_loss'][-1])
        val_loss_sublist.append(val_loss_sub_sublist)
    last_val_loss.append(val_loss_sublist)

valloss = np.array(last_val_loss)
position_lstm_naive_gdp = np.array(np.unravel_index(np.argmin(valloss),shape=valloss.shape))+1
# Attending to validation loss, best tuning is 4 layers, 10 nodes and 0 dropout
def create_lstm_naive_gdp():
    Input_daily = Input(shape=(259,237))
    Input_monthly = Input(shape=(12,44))
    Daily_to_monthly = CustomConv1D_d_simple()(Input_daily)
    Concat = tf.keras.layers.Concatenate(axis=-1)([Input_monthly, Daily_to_monthly])
    x = Concat
    for _ in range(4):
        x = LSTM(units=10, activation='relu', return_sequences=True)(x)
        x = Dropout(rate=0)(x)
    x = Flatten()(x)
    x = Dense(units=10, activation='relu')(x)
    Output = Dense(1, activation='sigmoid')(x)
    model = Model([Input_daily, Input_monthly], Output)
    return model

# Grid Search for Inflation
last_val_loss = []

for hlayers in [1,2,3,4]:
    mse_sublist = []
    val_loss_sublist = []
    for nodes in [10,50,100,250]:
        mse_sub_sublist = []
        val_loss_sub_sublist = []
        for dropout_rate in [0,0.1,0.2]:
            Input_daily = Input(shape=(259,237))
            Input_monthly = Input(shape=(12,44))
            Daily_to_monthly = CustomConv1D_d_simple()(Input_daily)
            Concat = tf.keras.layers.Concatenate(axis=-1)([Input_monthly, Daily_to_monthly])
            x = Concat
            for _ in range(hlayers):
                x = LSTM(units=nodes, activation='relu', return_sequences=True)(x)
                x = Dropout(rate=dropout_rate)(x)
            x = Flatten()(x)
            x = Dense(units=nodes, activation='relu')(x)
            Output = Dense(1, activation='sigmoid')(x)
            model = Model([Input_daily,Input_monthly], Output)

            # Model fit 
            model.compile(optimizer=custom_optimizer, loss='mean_squared_error') 
            random.seed(99)
            np.random.seed(99)
            tf.random.set_seed(99)
            history = model.fit([X_inf_train_d,X_inf_train_m], p_train, 
                                epochs=200, batch_size=16, validation_split=0.15,
                                callbacks=[early_stopping])
            val_loss_sub_sublist.append(history.history['val_loss'][-1])
        val_loss_sublist.append(val_loss_sub_sublist)
    last_val_loss.append(val_loss_sublist)

valloss = np.array(last_val_loss)
position_lstm_naive_inf = np.array(np.unravel_index(np.argmin(valloss),shape=valloss.shape))+1
# Attending to validation loss, best tuning is 3 layers, 50 nodes and 0% dropout
def create_lstm_naive_inf():
    Input_daily = Input(shape=(259,237))
    Input_monthly = Input(shape=(12,44))
    Daily_to_monthly = CustomConv1D_d_simple()(Input_daily)
    Concat = tf.keras.layers.Concatenate(axis=-1)([Input_monthly, Daily_to_monthly])
    x = Concat
    for _ in range(3):
        x = LSTM(units=50, activation='relu', return_sequences=True)(x)
        x = Dropout(rate=0)(x)
    x = Flatten()(x)
    x = Dense(units=50, activation='relu')(x)
    Output = Dense(1, activation='sigmoid')(x)
    model = Model([Input_daily, Input_monthly], Output)
    return model

#### 10.3 GRU NAIVE

# Custom optimizer and learning rate
custom_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# Grid Search for GDP
last_val_loss = []

for hlayers in [1,2,3,4]:
    mse_sublist = []
    val_loss_sublist = []
    for nodes in [10,50,100,250]:
        mse_sub_sublist = []
        val_loss_sub_sublist = []
        for dropout_rate in [0,0.1,0.2]:

            Input_daily = Input(shape=(259,237))
            Input_monthly = Input(shape=(12,44))
            Daily_to_monthly = CustomConv1D_d_simple()(Input_daily)
            Concat = tf.keras.layers.Concatenate(axis=-1)([Input_monthly, Daily_to_monthly])
            x = Concat
            for _ in range(hlayers):
                x = GRU(units=nodes, activation='relu', return_sequences=True)(x)
                x = Dropout(rate=dropout_rate)(x)
            x = Flatten()(x)
            x = Dense(units=nodes, activation='relu')(x)
            Output = Dense(1, activation='sigmoid')(x)
            model = Model([Input_daily,Input_monthly], Output)

            # Model fit 
            model.compile(optimizer=custom_optimizer, loss='mean_squared_error') 
            random.seed(99)
            np.random.seed(99)
            tf.random.set_seed(99)
            history = model.fit([X_gdp_train_d,X_gdp_train_m], y_train, 
                                epochs=200, batch_size=16, validation_split=0.15,
                                callbacks=[early_stopping])
            val_loss_sub_sublist.append(history.history['val_loss'][-1])
        val_loss_sublist.append(val_loss_sub_sublist)
    last_val_loss.append(val_loss_sublist)

valloss = np.array(last_val_loss)
position_gru_naive_gdp = np.array(np.unravel_index(np.argmin(valloss),shape=valloss.shape))+1
# Attending to validation loss, best tuning is 4 layers, 100 nodes and 0.1 dropout
def create_gru_naive_gdp():
    Input_daily = Input(shape=(259,237))
    Input_monthly = Input(shape=(12,44))
    Daily_to_monthly = CustomConv1D_d_simple()(Input_daily)
    Concat = tf.keras.layers.Concatenate(axis=-1)([Input_monthly, Daily_to_monthly])
    x = Concat
    for _ in range(4):
        x = LSTM(units=100, activation='relu', return_sequences=True)(x)
        x = Dropout(rate=0.1)(x)
    x = Flatten()(x)
    x = Dense(units=100, activation='relu')(x)
    Output = Dense(1, activation='sigmoid')(x)
    model = Model([Input_daily, Input_monthly], Output)
    return model

# Grid Search for Inflation
last_val_loss = []

for hlayers in [1,2,3,4]:
    mse_sublist = []
    val_loss_sublist = []
    for nodes in [10,50,100,250]:
        mse_sub_sublist = []
        val_loss_sub_sublist = []
        for dropout_rate in [0,0.1,0.2]:
            Input_daily = Input(shape=(259,237))
            Input_monthly = Input(shape=(12,44))
            Daily_to_monthly = CustomConv1D_d_simple()(Input_daily)
            Concat = tf.keras.layers.Concatenate(axis=-1)([Input_monthly, Daily_to_monthly])
            x = Concat
            for _ in range(hlayers):
                x = GRU(units=nodes, activation='relu', return_sequences=True)(x)
                x = Dropout(rate=dropout_rate)(x)
            x = Flatten()(x)
            x = Dense(units=nodes, activation='relu')(x)
            Output = Dense(1, activation='sigmoid')(x)
            model = Model([Input_daily,Input_monthly], Output)

            # Model fit 
            model.compile(optimizer=custom_optimizer, loss='mean_squared_error') 
            random.seed(99)
            np.random.seed(99)
            tf.random.set_seed(99)
            history = model.fit([X_inf_train_d,X_inf_train_m], p_train, 
                                epochs=200, batch_size=16, validation_split=0.15,
                                callbacks=[early_stopping])
            val_loss_sub_sublist.append(history.history['val_loss'][-1])
        val_loss_sublist.append(val_loss_sub_sublist)
    last_val_loss.append(val_loss_sublist)

valloss = np.array(last_val_loss)
position_gru_naive_inf = np.array(np.unravel_index(np.argmin(valloss),shape=valloss.shape))+1
# Attending to validation loss, best tuning is 3 layers, 10 nodes and 20% dropout
def create_gru_naive_inf():
    Input_daily = Input(shape=(259,237))
    Input_monthly = Input(shape=(12,44))
    Daily_to_monthly = CustomConv1D_d_simple()(Input_daily)
    Concat = tf.keras.layers.Concatenate(axis=-1)([Input_monthly, Daily_to_monthly])
    x = Concat
    for _ in range(3):
        x = LSTM(units=10, activation='relu', return_sequences=True)(x)
        x = Dropout(rate=0.2)(x)
    x = Flatten()(x)
    x = Dense(units=10, activation='relu')(x)
    Output = Dense(1, activation='sigmoid')(x)
    model = Model([Input_daily, Input_monthly], Output)
    return model

#### 10.4 SA NAIVE

# Custom optimizer and learning rate
custom_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# Grid Search for GDP
last_val_loss = []

for hlayers in [1,2,3,4]:
    mse_sublist = []
    val_loss_sublist = []
    for nodes in [10,50,100,250]:
        mse_sub_sublist = []
        val_loss_sub_sublist = []
        for dropout_rate in [0,0.1,0.2]:

            Input_daily = Input(shape=(259,237))
            Input_monthly = Input(shape=(12,44))
            Daily_to_monthly = CustomConv1D_d_simple()(Input_daily)
            Concat = tf.keras.layers.Concatenate(axis=-1)([Input_monthly, Daily_to_monthly])
            x = Concat
            for _ in range(hlayers):
                x = MultiHeadAttention(num_heads=8, key_dim=nodes)(x, x)
                x = Dropout(rate=dropout_rate)(x)
            x = Flatten()(x)
            x = Dense(units=nodes, activation='relu')(x)
            Output = Dense(1, activation='sigmoid')(x)
            model = Model([Input_daily,Input_monthly], Output)

            # Model fit 
            model.compile(optimizer=custom_optimizer, loss='mean_squared_error') 
            random.seed(99)
            np.random.seed(99)
            tf.random.set_seed(99)
            history = model.fit([X_gdp_train_d,X_gdp_train_m], y_train, 
                                epochs=200, batch_size=16, validation_split=0.15,
                                callbacks=[early_stopping])
            val_loss_sub_sublist.append(history.history['val_loss'][-1])
        val_loss_sublist.append(val_loss_sub_sublist)
    last_val_loss.append(val_loss_sublist)

valloss = np.array(last_val_loss)
position_sa_naive_gdp = np.array(np.unravel_index(np.argmin(valloss),shape=valloss.shape))+1
# Attending to validation loss, best tuning is 2 layers, 10 nodes and 0 dropout
def create_sa_naive_gdp():
    Input_daily = Input(shape=(259,237))
    Input_monthly = Input(shape=(12,44))
    Daily_to_monthly = CustomConv1D_d_simple()(Input_daily)
    Concat = tf.keras.layers.Concatenate(axis=-1)([Input_monthly, Daily_to_monthly])
    x = Concat
    for _ in range(2):
        x = MultiHeadAttention(num_heads=8, key_dim=10)(x, x)
        x = Dropout(rate=0)(x)
    x = Flatten()(x)
    x = Dense(units=10, activation='relu')(x)
    Output = Dense(1, activation='sigmoid')(x)
    model = Model([Input_daily, Input_monthly], Output)
    return model

# Grid Search for Inflation
last_val_loss = []

for hlayers in [1,2,3,4]:
    mse_sublist = []
    val_loss_sublist = []
    for nodes in [10,50,100,250]:
        mse_sub_sublist = []
        val_loss_sub_sublist = []
        for dropout_rate in [0,0.1,0.2]:
            Input_daily = Input(shape=(259,237))
            Input_monthly = Input(shape=(12,44))
            Daily_to_monthly = CustomConv1D_d_simple()(Input_daily)
            Concat = tf.keras.layers.Concatenate(axis=-1)([Input_monthly, Daily_to_monthly])
            x = Concat
            for _ in range(hlayers):
                x = MultiHeadAttention(num_heads=8, key_dim=nodes)(x, x)
                x = Dropout(rate=dropout_rate)(x)
            x = Flatten()(x)
            x = Dense(units=nodes, activation='relu')(x)
            Output = Dense(1, activation='sigmoid')(x)
            model = Model([Input_daily,Input_monthly], Output)

            # Model fit 
            model.compile(optimizer=custom_optimizer, loss='mean_squared_error') 
            random.seed(99)
            np.random.seed(99)
            tf.random.set_seed(99)
            history = model.fit([X_inf_train_d,X_inf_train_m], p_train, 
                                epochs=200, batch_size=16, validation_split=0.15,
                                callbacks=[early_stopping])
            val_loss_sub_sublist.append(history.history['val_loss'][-1])
        val_loss_sublist.append(val_loss_sub_sublist)
    last_val_loss.append(val_loss_sublist)

valloss = np.array(last_val_loss)
position_sa_naive_inf = np.array(np.unravel_index(np.argmin(valloss),shape=valloss.shape))+1
# Attending to validation loss, best tuning is 2 layers, 10 nodes and 10% dropout
def create_sa_naive_inf():
    Input_daily = Input(shape=(259,237))
    Input_monthly = Input(shape=(12,44))
    Daily_to_monthly = CustomConv1D_d_simple()(Input_daily)
    Concat = tf.keras.layers.Concatenate(axis=-1)([Input_monthly, Daily_to_monthly])
    x = Concat
    for _ in range(2):
        x = MultiHeadAttention(num_heads=8, key_dim=10)(x, x)
        x = Dropout(rate=0.1)(x)
    x = Flatten()(x)
    x = Dense(units=10, activation='relu')(x)
    Output = Dense(1, activation='sigmoid')(x)
    model = Model([Input_daily, Input_monthly], Output)
    return model

#### 10.5 ANN MIDAS

# Custom optimizer and learning rate
custom_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# Grid Search for GDP
last_val_loss = []

for hlayers in [1,2,3,4]:
    mse_sublist = []
    val_loss_sublist = []
    for nodes in [10,50,100,250]:
        mse_sub_sublist = []
        val_loss_sub_sublist = []
        for dropout_rate in [0,0.1,0.2]:

            Input_daily = Input(shape=(259,237))
            Input_monthly = Input(shape=(12,44))
            midas_d = CustomConv1D_d()(Input_daily)
            Concat = tf.keras.layers.Concatenate(axis=-1)([Input_monthly, midas_d])
            midas_m = CustomConv1D_m()(Concat)
            Flat = Flatten()(midas_m)
            x = Flat
            for _ in range(hlayers):
                x = Dense(units=nodes, activation='relu')(x)
                x = Dropout(rate=dropout_rate)(x)
            Output = Dense(1, activation='sigmoid')(x)
            model = Model([Input_daily,Input_monthly], Output)

            # Model fit 
            model.compile(optimizer=custom_optimizer, loss='mean_squared_error') 
            random.seed(99)
            np.random.seed(99)
            tf.random.set_seed(99)
            history = model.fit([X_gdp_train_d,X_gdp_train_m], y_train, 
                                epochs=200, batch_size=16, validation_split=0.15,
                                callbacks=[early_stopping])
            val_loss_sub_sublist.append(history.history['val_loss'][-1])
        val_loss_sublist.append(val_loss_sub_sublist)
    last_val_loss.append(val_loss_sublist)

valloss = np.array(last_val_loss)
position_ann_midas_gdp = np.array(np.unravel_index(np.argmin(valloss),shape=valloss.shape))+1
# Attending to validation loss, best tuning is 1 layer, 10 nodes and 0 dropout
def create_ann_midas_gdp():
    Input_daily = Input(shape=(259,237))
    Input_monthly = Input(shape=(12,44))
    midas_d = CustomConv1D_d()(Input_daily)
    Concat = tf.keras.layers.Concatenate(axis=-1)([Input_monthly, midas_d])
    midas_m = CustomConv1D_m()(Concat)
    Flat = Flatten()(midas_m)
    x = Flat
    for _ in range(1):
        x = Dense(units=10, activation='relu')(x)
        x = Dropout(rate=0)(x)
    Output = Dense(1, activation='sigmoid')(x)
    model = Model([Input_daily, Input_monthly], Output)
    return model

# Grid Search for Inflation
last_val_loss = []

for hlayers in [1,2,3,4]:
    mse_sublist = []
    val_loss_sublist = []
    for nodes in [10,50,100,250]:
        mse_sub_sublist = []
        val_loss_sub_sublist = []
        for dropout_rate in [0,0.1,0.2]:
            Input_daily = Input(shape=(259,237))
            Input_monthly = Input(shape=(12,44))
            midas_d = CustomConv1D_d()(Input_daily)
            Concat = tf.keras.layers.Concatenate(axis=-1)([Input_monthly, midas_d])
            midas_m = CustomConv1D_m()(Concat)
            Flat = Flatten()(midas_m)
            x = Flat
            for _ in range(hlayers):
                x = Dense(units=nodes, activation='relu')(x)
                x = Dropout(rate=dropout_rate)(x)
            Output = Dense(1, activation='sigmoid')(x)
            model = Model([Input_daily,Input_monthly], Output)

            # Model fit 
            model.compile(optimizer=custom_optimizer, loss='mean_squared_error') 
            random.seed(99)
            np.random.seed(99)
            tf.random.set_seed(99)
            history = model.fit([X_inf_train_d,X_inf_train_m], p_train, 
                                epochs=200, batch_size=16, validation_split=0.15,
                                callbacks=[early_stopping])
            val_loss_sub_sublist.append(history.history['val_loss'][-1])
        val_loss_sublist.append(val_loss_sub_sublist)
    last_val_loss.append(val_loss_sublist)

valloss = np.array(last_val_loss)
position_ann_midas_inf = np.array(np.unravel_index(np.argmin(valloss),shape=valloss.shape))+1
# Attending to validation loss, best tuning is 1 layers, 10 nodes and 20% dropout
def create_ann_midas_inf():
    Input_daily = Input(shape=(259,237))
    Input_monthly = Input(shape=(12,44))
    midas_d = CustomConv1D_d()(Input_daily)
    Concat = tf.keras.layers.Concatenate(axis=-1)([Input_monthly, midas_d])
    midas_m = CustomConv1D_m()(Concat)
    Flat = Flatten()(midas_m)
    x = Flat
    for _ in range(1):
        x = Dense(units=10, activation='relu')(x)
        x = Dropout(rate=0.2)(x)
    Output = Dense(1, activation='sigmoid')(x)
    model = Model([Input_daily, Input_monthly], Output)
    return model

#### 10.6 LSTM MIDAS

# Custom optimizer and learning rate
custom_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# Grid Search for GDP
last_val_loss = []

for hlayers in [1,2,3,4]:
    mse_sublist = []
    val_loss_sublist = []
    for nodes in [10,50,100,250]:
        mse_sub_sublist = []
        val_loss_sub_sublist = []
        for dropout_rate in [0,0.1,0.2]:

            Input_daily = Input(shape=(259,237))
            Input_monthly = Input(shape=(12,44))
            midas_d = CustomConv1D_d()(Input_daily)
            Concat = tf.keras.layers.Concatenate(axis=-1)([Input_monthly, midas_d])
            midas_m = CustomConv1D_m()(Concat)
            x = midas_m
            for _ in range(hlayers):
                x = LSTM(units=nodes, activation='relu', return_sequences=True)(x)
                x = Dropout(rate=dropout_rate)(x)
            x = Flatten()(x)
            x = Dense(units=nodes, activation='relu')(x)
            Output = Dense(1, activation='sigmoid')(x)
            model = Model([Input_daily,Input_monthly], Output)

            # Model fit 
            model.compile(optimizer=custom_optimizer, loss='mean_squared_error') 
            random.seed(99)
            np.random.seed(99)
            tf.random.set_seed(99)
            history = model.fit([X_gdp_train_d,X_gdp_train_m], y_train, 
                                epochs=200, batch_size=16, validation_split=0.15,
                                callbacks=[early_stopping])
            val_loss_sub_sublist.append(history.history['val_loss'][-1])
        val_loss_sublist.append(val_loss_sub_sublist)
    last_val_loss.append(val_loss_sublist)

valloss = np.array(last_val_loss)
position_lstm_midas_gdp = np.array(np.unravel_index(np.argmin(valloss),shape=valloss.shape))+1
# Attending to validation loss, best tuning is 1 layers, 10 nodes and 0.2 dropout
def create_lstm_midas_gdp():
    Input_daily = Input(shape=(259,237))
    Input_monthly = Input(shape=(12,44))
    midas_d = CustomConv1D_d()(Input_daily)
    Concat = tf.keras.layers.Concatenate(axis=-1)([Input_monthly, midas_d])
    midas_m = CustomConv1D_m()(Concat)
    x = midas_m
    for _ in range(1):
        x = LSTM(units=10, activation='relu', return_sequences=True)(x)
        x = Dropout(rate=0.2)(x)
    x = Flatten()(x)
    x = Dense(units=10, activation='relu')(x)
    Output = Dense(1, activation='sigmoid')(x)
    model = Model([Input_daily, Input_monthly], Output)
    return model

# Grid Search for Inflation
last_val_loss = []

for hlayers in [1,2,3,4]:
    mse_sublist = []
    val_loss_sublist = []
    for nodes in [10,50,100,250]:
        mse_sub_sublist = []
        val_loss_sub_sublist = []
        for dropout_rate in [0,0.1,0.2]:
            Input_daily = Input(shape=(259,237))
            Input_monthly = Input(shape=(12,44))
            midas_d = CustomConv1D_d()(Input_daily)
            Concat = tf.keras.layers.Concatenate(axis=-1)([Input_monthly, midas_d])
            midas_m = CustomConv1D_m()(Concat)
            x = midas_m
            for _ in range(hlayers):
                x = LSTM(units=nodes, activation='relu', return_sequences=True)(x)
                x = Dropout(rate=dropout_rate)(x)
            x = Flatten()(x)
            x = Dense(units=nodes, activation='relu')(x)
            Output = Dense(1, activation='sigmoid')(x)
            model = Model([Input_daily,Input_monthly], Output)

            # Model fit 
            model.compile(optimizer=custom_optimizer, loss='mean_squared_error') 
            random.seed(99)
            np.random.seed(99)
            tf.random.set_seed(99)
            history = model.fit([X_inf_train_d,X_inf_train_m], p_train, 
                                epochs=200, batch_size=16, validation_split=0.15,
                                callbacks=[early_stopping])
            val_loss_sub_sublist.append(history.history['val_loss'][-1])
        val_loss_sublist.append(val_loss_sub_sublist)
    last_val_loss.append(val_loss_sublist)

valloss = np.array(last_val_loss)
position_lstm_midas_inf = np.array(np.unravel_index(np.argmin(valloss),shape=valloss.shape))+1
# Attending to validation loss, best tuning is 3 layers, 10 nodes and 20% dropout
def create_lstm_midas_inf():
    Input_daily = Input(shape=(259,237))
    Input_monthly = Input(shape=(12,44))
    midas_d = CustomConv1D_d()(Input_daily)
    Concat = tf.keras.layers.Concatenate(axis=-1)([Input_monthly, midas_d])
    midas_m = CustomConv1D_m()(Concat)
    x = midas_m
    for _ in range(3):
        x = LSTM(units=10, activation='relu', return_sequences=True)(x)
        x = Dropout(rate=0.2)(x)
    x = Flatten()(x)
    x = Dense(units=10, activation='relu')(x)
    Output = Dense(1, activation='sigmoid')(x)
    model = Model([Input_daily, Input_monthly], Output)
    return model

#### 10.7 GRU MIDAS

# Custom optimizer and learning rate
custom_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# Grid Search for GDP
last_val_loss = []

for hlayers in [1,2,3,4]:
    mse_sublist = []
    val_loss_sublist = []
    for nodes in [10,50,100,250]:
        mse_sub_sublist = []
        val_loss_sub_sublist = []
        for dropout_rate in [0,0.1,0.2]:

            Input_daily = Input(shape=(259,237))
            Input_monthly = Input(shape=(12,44))
            midas_d = CustomConv1D_d()(Input_daily)
            Concat = tf.keras.layers.Concatenate(axis=-1)([Input_monthly, midas_d])
            midas_m = CustomConv1D_m()(Concat)
            x = midas_m
            for _ in range(hlayers):
                x = GRU(units=nodes, activation='relu', return_sequences=True)(x)
                x = Dropout(rate=dropout_rate)(x)
            x = Flatten()(x)
            x = Dense(units=nodes, activation='relu')(x)
            Output = Dense(1, activation='sigmoid')(x)
            model = Model([Input_daily,Input_monthly], Output)

            # Model fit 
            model.compile(optimizer=custom_optimizer, loss='mean_squared_error') 
            random.seed(99)
            np.random.seed(99)
            tf.random.set_seed(99)
            history = model.fit([X_gdp_train_d,X_gdp_train_m], y_train, 
                                epochs=200, batch_size=16, validation_split=0.15,
                                callbacks=[early_stopping])
            val_loss_sub_sublist.append(history.history['val_loss'][-1])
        val_loss_sublist.append(val_loss_sub_sublist)
    last_val_loss.append(val_loss_sublist)

valloss = np.array(last_val_loss)
position_gru_midas_gdp = np.array(np.unravel_index(np.argmin(valloss),shape=valloss.shape))+1
# Attending to validation loss, best tuning is 3 layers, 10 nodes and 0 dropout
def create_gru_midas_gdp():
    Input_daily = Input(shape=(259,237))
    Input_monthly = Input(shape=(12,44))
    midas_d = CustomConv1D_d()(Input_daily)
    Concat = tf.keras.layers.Concatenate(axis=-1)([Input_monthly, midas_d])
    midas_m = CustomConv1D_m()(Concat)
    x = midas_m
    for _ in range(3):
        x = GRU(units=10, activation='relu', return_sequences=True)(x)
        x = Dropout(rate=0)(x)
    x = Flatten()(x)
    x = Dense(units=10, activation='relu')(x)
    Output = Dense(1, activation='sigmoid')(x)
    model = Model([Input_daily, Input_monthly], Output)
    return model


# Grid Search for Inflation
last_val_loss = []

for hlayers in [1,2,3,4]:
    mse_sublist = []
    val_loss_sublist = []
    for nodes in [10,50,100,250]:
        mse_sub_sublist = []
        val_loss_sub_sublist = []
        for dropout_rate in [0,0.1,0.2]:
            Input_daily = Input(shape=(259,237))
            Input_monthly = Input(shape=(12,44))
            midas_d = CustomConv1D_d()(Input_daily)
            Concat = tf.keras.layers.Concatenate(axis=-1)([Input_monthly, midas_d])
            midas_m = CustomConv1D_m()(Concat)
            x = midas_m
            for _ in range(hlayers):
                x = GRU(units=nodes, activation='relu', return_sequences=True)(x)
                x = Dropout(rate=dropout_rate)(x)
            x = Flatten()(x)
            x = Dense(units=nodes, activation='relu')(x)
            Output = Dense(1, activation='sigmoid')(x)
            model = Model([Input_daily,Input_monthly], Output)

            # Model fit 
            model.compile(optimizer=custom_optimizer, loss='mean_squared_error') 
            random.seed(99)
            np.random.seed(99)
            tf.random.set_seed(99)
            history = model.fit([X_inf_train_d,X_inf_train_m], p_train, 
                                epochs=200, batch_size=16, validation_split=0.15,
                                callbacks=[early_stopping])
            val_loss_sub_sublist.append(history.history['val_loss'][-1])
        val_loss_sublist.append(val_loss_sub_sublist)
    last_val_loss.append(val_loss_sublist)

valloss = np.array(last_val_loss)
position_gru_midas_inf = np.array(np.unravel_index(np.argmin(valloss),shape=valloss.shape))+1
# Attending to validation loss, best tuning is 2 layers, 10 nodes and 10% dropout
def create_gru_midas_inf():
    Input_daily = Input(shape=(259,237))
    Input_monthly = Input(shape=(12,44))
    midas_d = CustomConv1D_d()(Input_daily)
    Concat = tf.keras.layers.Concatenate(axis=-1)([Input_monthly, midas_d])
    midas_m = CustomConv1D_m()(Concat)
    x = midas_m
    for _ in range(2):
        x = GRU(units=10, activation='relu', return_sequences=True)(x)
        x = Dropout(rate=0.1)(x)
    x = Flatten()(x)
    x = Dense(units=10, activation='relu')(x)
    Output = Dense(1, activation='sigmoid')(x)
    model = Model([Input_daily, Input_monthly], Output)
    return model


#### 10.8 SA MIDAS

# Custom optimizer and learning rate
custom_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# Grid Search for GDP
last_val_loss = []

for hlayers in [1,2,3,4]:
    mse_sublist = []
    val_loss_sublist = []
    for nodes in [10,50,100,250]:
        mse_sub_sublist = []
        val_loss_sub_sublist = []
        for dropout_rate in [0,0.1,0.2]:

            Input_daily = Input(shape=(259,237))
            Input_monthly = Input(shape=(12,44))
            midas_d = CustomConv1D_d()(Input_daily)
            Concat = tf.keras.layers.Concatenate(axis=-1)([Input_monthly, midas_d])
            midas_m = CustomConv1D_m()(Concat)
            x = midas_m
            for _ in range(hlayers):
                x = MultiHeadAttention(num_heads=8, key_dim=nodes)(x, x)
                x = Dropout(rate=dropout_rate)(x)
            x = Flatten()(x)
            x = Dense(units=nodes, activation='relu')(x)
            Output = Dense(1, activation='sigmoid')(x)
            model = Model([Input_daily,Input_monthly], Output)

            # Model fit 
            model.compile(optimizer=custom_optimizer, loss='mean_squared_error') 
            random.seed(99)
            np.random.seed(99)
            tf.random.set_seed(99)
            history = model.fit([X_gdp_train_d,X_gdp_train_m], y_train, 
                                epochs=200, batch_size=16, validation_split=0.15,
                                callbacks=[early_stopping])
            val_loss_sub_sublist.append(history.history['val_loss'][-1])
        val_loss_sublist.append(val_loss_sub_sublist)
    last_val_loss.append(val_loss_sublist)

valloss = np.array(last_val_loss)
position_sa_midas_gdp = np.array(np.unravel_index(np.argmin(valloss),shape=valloss.shape))+1
# Attending to validation loss, best tuning is 3 layers, 10 nodes and 0.2 dropout
def create_sa_midas_gdp():
    Input_daily = Input(shape=(259,237))
    Input_monthly = Input(shape=(12,44))
    midas_d = CustomConv1D_d()(Input_daily)
    Concat = tf.keras.layers.Concatenate(axis=-1)([Input_monthly, midas_d])
    midas_m = CustomConv1D_m()(Concat)
    x = midas_m
    for _ in range(3):
        x = MultiHeadAttention(num_heads=8, key_dim=10)(x, x)
        x = Dropout(rate=0.2)(x)
    x = Flatten()(x)
    x = Dense(units=10, activation='relu')(x)
    Output = Dense(1, activation='sigmoid')(x)
    model = Model([Input_daily, Input_monthly], Output)
    return model


# Grid Search for Inflation
last_val_loss = []

for hlayers in [1,2,3,4]:
    mse_sublist = []
    val_loss_sublist = []
    for nodes in [10,50,100,250]:
        mse_sub_sublist = []
        val_loss_sub_sublist = []
        for dropout_rate in [0,0.1,0.2]:
            Input_daily = Input(shape=(259,237))
            Input_monthly = Input(shape=(12,44))
            midas_d = CustomConv1D_d()(Input_daily)
            Concat = tf.keras.layers.Concatenate(axis=-1)([Input_monthly, midas_d])
            midas_m = CustomConv1D_m()(Concat)
            x = midas_m
            for _ in range(hlayers):
                x = MultiHeadAttention(num_heads=8, key_dim=nodes)(x, x)
                x = Dropout(rate=dropout_rate)(x)
            x = Flatten()(x)
            x = Dense(units=nodes, activation='relu')(x)
            Output = Dense(1, activation='sigmoid')(x)
            model = Model([Input_daily,Input_monthly], Output)

            # Model fit 
            model.compile(optimizer=custom_optimizer, loss='mean_squared_error') 
            random.seed(99)
            np.random.seed(99)
            tf.random.set_seed(99)
            history = model.fit([X_inf_train_d,X_inf_train_m], p_train, 
                                epochs=200, batch_size=16, validation_split=0.15,
                                callbacks=[early_stopping])
            val_loss_sub_sublist.append(history.history['val_loss'][-1])
        val_loss_sublist.append(val_loss_sub_sublist)
    last_val_loss.append(val_loss_sublist)

valloss = np.array(last_val_loss)
position_sa_midas_inf = np.array(np.unravel_index(np.argmin(valloss),shape=valloss.shape))+1
# Attending to validation loss, best tuning is 1 layers, 10 nodes and 0% dropout
def create_sa_midas_inf():
    Input_daily = Input(shape=(259,237))
    Input_monthly = Input(shape=(12,44))
    midas_d = CustomConv1D_d()(Input_daily)
    Concat = tf.keras.layers.Concatenate(axis=-1)([Input_monthly, midas_d])
    midas_m = CustomConv1D_m()(Concat)
    x = midas_m
    for _ in range(1):
        x = MultiHeadAttention(num_heads=8, key_dim=10)(x, x)
        x = Dropout(rate=0)(x)
    x = Flatten()(x)
    x = Dense(units=10, activation='relu')(x)
    Output = Dense(1, activation='sigmoid')(x)
    model = Model([Input_daily, Input_monthly], Output)
    return model


### 11 RESULTS FOR MODELS WITH NO SPARSITY, DIM REDUCTION

# Defining a callback to avoid overfitting
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                  patience=3,restore_best_weights=True)

# Custom optimizer and learning rates
custom_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# Number of initializations
inits = 20

# GDP

model_functions = [create_ann_naive_gdp, create_ann_midas_gdp,
                   create_lstm_naive_gdp, create_lstm_midas_gdp,
                   create_gru_naive_gdp, create_gru_midas_gdp,
                   create_sa_naive_gdp, create_sa_midas_gdp]

all_models_errors = []

for model_func in model_functions:
    horizon_errors = []

    for quarter in [1,2,3,4]:
        h_error = []
        for i in range(inits):
            model = model_func()
            model.compile(optimizer=custom_optimizer, loss='mean_squared_error')
            model.fit([data_lists[quarter-1][0],data_lists[quarter-1][1]],data_lists[quarter-1][2],epochs=200, 
                    batch_size=16, validation_split=0.15,callbacks=[early_stopping])
            preds = np.squeeze(model.predict([data_lists[quarter-1][3],data_lists[quarter-1][4]]))*(max(gdp_growth_y) - min(gdp_growth_y)) + min(gdp_growth_y)
            y_test = np.squeeze(data_lists[quarter-1][5])*(max(gdp_growth_y) - min(gdp_growth_y)) + min(gdp_growth_y)
            rmse = np.sqrt(np.sum((y_test - preds)**2))
            h_error.append(rmse)
        
        emean = np.mean(np.array(h_error))
        emax = np.max(np.array(h_error))
        emin = np.min(np.array(h_error))
        horizon_errors.append([emean,emax,emin])

    all_models_errors.append(horizon_errors)
    print(str(model_func),"is done")

array = np.around(np.array(all_models_errors), 2)
np.save("mse_gdp_1.npy",array)


### 12 RESULTS FOR MODELS WITH NO SPARSITY, DIM REDUCTION (INFLATION)

# Defining a callback to avoid overfitting
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                  patience=3,restore_best_weights=True)

# Custom optimizer and learning rates
custom_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# Number of initializations
inits = 20

# GDP

model_functions = [create_ann_naive_inf, create_ann_midas_inf,
                   create_lstm_naive_inf, create_lstm_midas_inf,
                   create_gru_naive_inf, create_gru_midas_inf,
                   create_sa_naive_inf, create_sa_midas_inf]

all_models_errors = []

for model_func in model_functions:
    horizon_errors = []

    for quarter in [1,2,3,4]:
        h_error = []
        for i in range(inits):
            model = model_func()
            model.compile(optimizer=custom_optimizer, loss='mean_squared_error')
            model.fit([data_lists[quarter-1][0],data_lists[quarter-1][1]],data_lists[quarter-1][2],epochs=200, 
                    batch_size=16, validation_split=0.15,callbacks=[early_stopping])
            preds = np.squeeze(model.predict([data_lists[quarter-1][3],data_lists[quarter-1][4]]))*(max(inflation_data) - min(inflation_data)) + min(inflation_data)
            y_test = np.squeeze(data_lists[quarter-1][5])*(max(inflation_data) - min(inflation_data)) + min(inflation_data)
            rmse = np.sqrt(np.sum((y_test - preds)**2))
            h_error.append(rmse)
        
        emean = np.mean(np.array(h_error))
        emax = np.max(np.array(h_error))
        emin = np.min(np.array(h_error))
        horizon_errors.append([emean,emax,emin])

    all_models_errors.append(horizon_errors)
    print(str(model_func),"is done")

array = np.around(np.array(all_models_errors), 2)












