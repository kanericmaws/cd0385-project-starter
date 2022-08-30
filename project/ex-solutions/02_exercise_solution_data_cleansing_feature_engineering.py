#!/usr/bin/env python
# coding: utf-8

# # Exercise: Data Cleansing and Feature Engineering
# 
# In this exercise, we'll be loading in a dataset that has some problems. In order for us to get it ready for our models, we will apply some of the technics we learned.
# 
# Apply these changes to the `data.csv` dataset.
# 1. Load `data.csv` into a dataframe.
# 2. Output the table info to see if there are any null values.
# 3. Remove all null values from the dataframe.
# 4. Change the `date` column from an object to a `datetime64[ns]` type.
# 5. Change the `weather` column to a category type.
# 6. One hot encode the `date` column to year, month, and day.
# 7. Normalized the columns from the `all_features` list so each feature has a zero mean.
# 8. Create and save the cleaned dataframe, as well as the train/validation/test dataframes to CSV.

# In[1]:


import random
from datetime import datetime
import pandas as pd
import numpy as np

from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# In[2]:


# Reading the dataset created by 02_exercise_dataset_creation.ipynb
df = pd.read_csv("data.csv")


# In[3]:


# Always good to check to see if the data looks right
df.head()


# In[4]:


# Output general info about the table, notice we have some null values in all of our features
df.info()


# In[5]:


# Drop all null values
df = df.dropna()


# In[6]:


# Change the date column to a datetime
df.loc[:, "date"] = pd.to_datetime(df.loc[:, "date"])
# Change weather column to a category 
df.loc[:, "weather"] = df["weather"].astype("category")


# In[7]:


# Extract year, month, and day into separate columns
df["year"] = df.date.dt.year
df["month"] = df.date.dt.month
df["day"] = df.date.dt.day


# In[8]:


# One hot encode the weather category to have individual features. Prefix with `weather`
weather_one_hot_df = pd.get_dummies(df.weather, prefix="weather")


# In[9]:


# Add the one hot encoded values back to the df
df[weather_one_hot_df.columns.tolist()] = weather_one_hot_df


# In[10]:


# Verify now that are table info has no nulls and correct Dtypes
df.info()


# In[11]:


# These may change if you decided to call your columns different from above
all_features = [
    "feature0",
    "feature1",
    "feature2",
    "year",
    "month",
    "day",
    "weather_cloudy",
    "weather_rainy",
    "weather_sunny",
]


# In[12]:


# Table summary, notice the mean to many of our tables are not zero.
df[all_features].describe()


# In[13]:


# Standarize feature values to have a zero mean
scaler = StandardScaler()
scaler.fit(df[all_features])
df.loc[:, all_features] = scaler.transform(df[all_features])


# In[14]:


# Verify our features we are using now all have zero mean
df[all_features].describe()


# In[15]:


# train: 0.8 | test: 0.2
df_train, df_test = train_test_split(df, test_size=0.2, random_state=0)

# train: 0.6 | validation: 0.2
df_train, df_val = train_test_split(df_train, test_size=0.25, random_state=0)

# Final dataset sizes: train: 0.6, validation: 0.2, text: 0.2,


# In[16]:


# Output each shape to confirm the size of train/validation/test
print(f"Train: {df_train.shape}")
print(f"Validation: {df_val.shape}")
print(f"Test: {df_test.shape}")


# In[17]:


# Save all clean data, and the train, validation, test data as csv
df.to_csv("data_clean.csv", index=False)
df_train.to_csv("train.csv", index=False)
df_val.to_csv("validation.csv", index=False)
df_test.to_csv("test.csv", index=False)


# In[ ]:




