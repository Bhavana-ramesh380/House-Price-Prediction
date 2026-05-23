import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load csv/ data
#data = pd.read_csv("data.csv")
#print(data)  # just to see data

# Check missing values
#print(data.isnull().sum())


# Features (multiple inputs)
#X = data[['size', 'bedrooms', 'age']] #model now learn 3 relationship at once

# Extract
#X = data[['size']]
#y = data['price'] 

# Target
#y = data['price']

# Train / Model
#model = LinearRegression()
#model.fit(X, y)

# User Input
#size = int(input("Enter house size: "))
#prediction = model.predict([[size]])

#print("Predicted price:", prediction)

# Plot
#plt.scatter(X, y)
#plt.plot(X, model.predict(X))
#plt.show()

# User input
#size = int(input("Enter size: "))
#bedrooms = int(input("Enter bedrooms: "))
#age = int(input("Enter age of house: "))

#prediction = model.predict([[size, bedrooms, age]])

#print("Predicted price:", prediction)

# Take input
#size = int(input("Enter size: "))
#bedrooms = int(input("Enter bedrooms: "))
#age = int(input("Enter age of house: "))

# Create DataFrame (IMPORTANT)
#input_data = pd.DataFrame([[size, bedrooms, age]], 
                          #columns=['size', 'bedrooms', 'age'])

# Predict
#prediction = model.predict(input_data)
#print("Predicted price:", prediction)
#====================================
# Load data
#data = pd.read_csv("data.csv")

# Check missing values
#print(data.isnull().sum())

# Fill missing values
#data['bedrooms'] = data['bedrooms'].fillna(data['bedrooms'].mean())
#data['age'] = data['age'].fillna(data['age'].mean())

# Drop missing target
#data = data.dropna(subset=['price'])

# Features & target
#X = data[['size', 'bedrooms', 'age']]
#y = data['price']

# Train
#model = LinearRegression()
#model.fit(X, y)

# Input
#size = int(input("Enter size: "))
#bedrooms = int(input("Enter bedrooms: "))
#age = int(input("Enter age: "))

# Predict
#input_data = pd.DataFrame([[size, bedrooms, age]],
                      #    columns=['size','bedrooms','age'])

#prediction = model.predict(input_data)
#print("Predicted price:", prediction)

#==========================

# Load data
#data = pd.read_csv("data.csv")

# Handle missing values #cleaning
#data['bedrooms'] = data['bedrooms'].fillna(data['bedrooms'].mean())
#data['age'] = data['age'].fillna(data['age'].mean())
#data = data.dropna(subset=['price'])

# Features & target
#X = data[['size', 'bedrooms', 'age']]
#y = data['price']

# Split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#  Scaling
#scaler = StandardScaler()
#X_train = scaler.fit_transform(X_train)   # learn + transform
#X_test = scaler.transform(X_test)         # only transform

# Train/ model
#model = LinearRegression()
#model.fit(X_train, y_train)

# Accuracy
#print("Accuracy:", model.score(X_test, y_test))

# Input
#size = int(input("Enter size: "))
#bedrooms = int(input("Enter bedrooms: "))
#age = int(input("Enter age: "))

# Scale input (VERY IMPORTANT)
#input_data = scaler.transform([[size, bedrooms, age]])

# Predict
#prediction = model.predict(input_data)

#print("Predicted price:", prediction)
#======================

# Load data
#data = pd.read_csv("data.csv")

# Cleaning
#data['bedrooms'] = data['bedrooms'].fillna(data['bedrooms'].mean())
#data['age'] = data['age'].fillna(data['age'].mean())
#data = data.dropna(subset=['price'])

# Features & target
#X = data[['size', 'bedrooms', 'age']]
#y = data['price']

#@st.cache_resource
#def train_model():
    #scaler = StandardScaler()
    #X_scaled = scaler.fit_transform(X)

    #model = LinearRegression()
    #model.fit(X_scaled, y)

    #return model, scaler

# Call function
#model, scaler = train_model()


# Split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Scaling
#scaler = StandardScaler()
#X_train = scaler.fit_transform(X_train)
#X_test = scaler.transform(X_test)

# Train
#model = LinearRegression()
#model.fit(X_train, y_train)

# Predict test data
#y_pred = model.predict(X_test)

# Errors
#mae = mean_absolute_error(y_test, y_pred)
#mse = mean_squared_error(y_test, y_pred)
#rmse = np.sqrt(mse)

#print("MAE:", mae)
#print("RMSE:", rmse)
#print("R2 Score:", model.score(X_test, y_test))

# User input
#size = int(input("Enter size: "))
#bedrooms = int(input("Enter bedrooms: "))
#age = int(input("Enter age: "))

#input_data = scaler.transform([[size, bedrooms, age]])
#prediction = model.predict(input_data)

#print("Predicted price:", prediction)
#===================

# Load data
data = pd.read_csv("data.csv")

# Cleaning
data['bedrooms'] = data['bedrooms'].fillna(data['bedrooms'].mean())
data['age'] = data['age'].fillna(data['age'].mean())
data = data.dropna(subset=['price'])

# Features & target
X = data[['size', 'bedrooms', 'age']]
y = data['price']

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
model = LinearRegression()
model.fit(X_scaled, y)

# ---------- USER INPUT ----------
try:
    size = float(input("Enter size: "))
    bedrooms = int(input("Enter bedrooms: "))
    age = int(input("Enter age: "))

    # Predict
    input_data = scaler.transform([[size, bedrooms, age]])
    prediction = model.predict(input_data)

    print(f"Predicted Price: ₹ {prediction[0]:.2f} Lakhs")

except ValueError:
    print("❌ Invalid input. Please enter numeric values only.")