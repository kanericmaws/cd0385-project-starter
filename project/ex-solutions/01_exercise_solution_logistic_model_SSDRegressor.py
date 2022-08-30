#!/usr/bin/env python
# coding: utf-8

# # Exercise: Linear Models
# 
# In this exercise, we'll be exploring two types of linear models, one regression, one classification. While regression is what you typically think of for a linear model, they can also be used effectively in classification problems.
# 
# You're tasked with compeleting the following steps:
# 1. Load in the wine dataset from scikit learn.
# 2. For the wine dataset, create a train and test split, 80% train / 20% test.
# 3. Create a LogisticRegression model with these hyper parameters: random_state=0, max_iter=10000
# 4. Evaluate the model with the test dataset
# 5. Load the diabetes dataset from scikit learn
# 6. For the Diabetes dataset, create a train and test split, 80% train / 20% test.
# 7. Create a SGDRegressor model model with these hyper parameters: random_state=0, max_iter=10000
# 8. Evaluate the model with the test dataset

# In[1]:


import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, SGDRegressor


# ## Linear Classifier

# In[2]:


# Load in the wine dataset
wine = datasets.load_wine()


# In[3]:


# Create the wine `data` dataset as a dataframe and name the columns with `feature_names`
df = pd.DataFrame(wine["data"], columns=wine["feature_names"])

# Include the target as well
df['target'] = wine["target"]


# In[4]:


# Check your dataframe by `.head()`
df.head()


# In[5]:


# Split your data with these ratios: train: 0.8 | test: 0.2
df_train, df_test = train_test_split(df, test_size=0.2, random_state=0)


# In[6]:


# How does the model perform on the training dataset and default model parameters?
# Using the hyperparameters in the requirements, is there improvement?
# Remember we use the test dataset to score the model
clf = LogisticRegression(random_state=0, max_iter=10000).fit(
    df_train.loc[:, df_train.columns != "target"], df_train["target"]
)
clf.score(df_test.loc[:, df_test.columns != "target"], df_test["target"])


# ## Linear Regression

# In[11]:


# Load in the diabetes dataset
diabetes = datasets.load_diabetes()


# In[12]:


# Create the diabetes `data` dataset as a dataframe and name the columns with `feature_names`
dfd = pd.DataFrame(diabetes["data"], columns=diabetes["feature_names"])

# Include the target as well
dfd['target'] = diabetes["target"]


# In[13]:


# Check your dataframe by `.head()`
dfd.head()


# In[14]:


# Split your data with these ratios: train: 0.8 | test: 0.2
dfd_train, dfd_test = train_test_split(dfd, test_size=0.2, random_state=0)


# In[15]:


# How does the model perform on the training dataset and default model parameters?
# Using the hyperparameters in the requirements, is there improvement?
# Remember we use the test dataset to score the model
reg = SGDRegressor(random_state=0, max_iter=10000).fit(
    dfd_train.loc[:, dfd_train.columns != "target"], dfd_train["target"]
)
reg.score(dfd_test.loc[:, dfd_test.columns != "target"], dfd_test["target"])

