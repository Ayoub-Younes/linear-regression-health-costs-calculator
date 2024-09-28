# Import libraries. You may or may not use all of these.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers # type: ignore
from tensorflow.keras.layers import Dense, Flatten, Dropout, Input, StringLookup, CategoryEncoding, IntegerLookup # type: ignore
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow_docs as tfdocs # type: ignore # type: ignore
import tensorflow_docs.plots # type: ignore
import tensorflow_docs.modeling # type: ignore

# Import data
url = 'https://cdn.freecodecamp.org/project-data/health-costs/insurance.csv'
current_dir = os.path.dirname(os.path.abspath(__file__))
filename = os.path.join(current_dir, "insurance.csv")
response = requests.get(url)

'''with open(filename, "wb") as file:
  file.write(response.content)'''

dataset = pd.read_csv(filename)

# Prepare labels
y = dataset.pop('expenses')

# Select numeric and categorical columns
NUMERIC_COLUMNS = ['age', 'bmi']
CATEGORICAL_COLUMNS = ['sex', 'children', 'smoker', 'region']


dataset = pd.get_dummies(dataset, columns=CATEGORICAL_COLUMNS, drop_first=True)

# Standardize numeric features
scaler = StandardScaler()
dataset[NUMERIC_COLUMNS] = scaler.fit_transform(dataset[NUMERIC_COLUMNS])




# Prepare features
X = dataset



# Split data into training and test sets
train_dataset, test_dataset, train_labels, test_labels = train_test_split(X, y,  test_size=0.2, random_state=42)




# Define model using Sequential API
model = tf.keras.Sequential([
    Input(shape=(train_dataset.shape[1],)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(1, activation='linear')
])

# Compile the model
model.compile(optimizer='Adam', loss='mean_squared_error', metrics=['mae', 'mse'])



# Train the model
model.fit(X, y, epochs=1000, verbose=2)

loss, mae, mse = model.evaluate(test_dataset, test_labels, verbose=2)

print(mae)
