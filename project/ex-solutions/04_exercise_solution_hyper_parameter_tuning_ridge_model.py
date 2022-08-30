#!/usr/bin/env python
# coding: utf-8

# # Exercise: Diabetes Model
# 
# In this exercise, we're going to take the knowledge we gained from the lesson and apply it to the [Diabetes dataset](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset). This well known dataset already has it's data cleaned and normalized, so no need to do any of those steps. The steps required to complete this exercise are as follows:
# 
# 1. Load the diabetes dataset into a dataframe.
# 2. Check the table summary to show that indeed the mean is zero for all features.
# 3. Split the dataset into train, validation, and test sets
# 4. Use a linear regression `Ridge` model to fit and score:
#     1. Fit and score on the whole dataset
#     2. Fit on train, score on validation, using default model
#     3. Fit on train, score on validation, using hyperparameters model
#     4. Fit on train, score on test, using hyperparameterized model
# 5. Plot all scores in a bar graph

# In[1]:


import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge


# In[2]:


# Load in the diabetes dataset
diabetes = datasets.load_diabetes()


# In[3]:


# Create the diabetes `data` dataset as a dataframe and name the columns with `feature_names`
df = pd.DataFrame(diabetes["data"], columns=diabetes["feature_names"])

# Include the target as well
df["target"] = diabetes["target"]


# In[4]:


df.head()


# In[5]:


# Describe df using table summary.
# No need to normalize, near zero mean.
df.describe()


# In[6]:


# train: 0.8 | test: 0.2
df_train, df_test = train_test_split(df, test_size=0.2, random_state=0)

# train: 0.6 | validation: 0.2
df_train, df_val = train_test_split(df_train, test_size=0.25, random_state=0)

# Final dataset sizes: train: 0.6, validation: 0.2, text: 0.2


# In[7]:


# How does the model perform on the entire dataset and default model parameters
reg = Ridge().fit(df[diabetes["feature_names"]], df["target"])
all_df_score = reg.score(df[diabetes["feature_names"]], df["target"])
all_df_score


# In[8]:


# How does the model perform on the training dataset and default model parameters
# Remember we use the validation dataset score the model
reg = Ridge().fit(df_train[diabetes["feature_names"]], df_train["target"])
val_df_score = reg.score(df_val[diabetes["feature_names"]], df_val["target"])
val_df_score


# In[9]:


# How does the model perform on the training dataset and different model parameters
# Change alpha, solver, and max_iter
reg_h = Ridge(alpha=0.01, solver="saga", max_iter=10000).fit(
    df_train[diabetes["feature_names"]], df_train["target"]
)
val_df_h_score = reg_h.score(df_val[diabetes["feature_names"]], df_val["target"])
val_df_h_score


# In[10]:


# Use optimized data on the held out test dataset.
test_df_h_score = reg_h.score(df_test[diabetes["feature_names"]], df_test["target"])
test_df_h_score


# In[39]:


# Bar plot of all scores from each model fit: all_df_score, val_df_score, val_df_h_score, test_df_h_score
pd.Series(
    {
        "all_df_score": all_df_score,
        "val_df_score": val_df_score,
        "val_df_h_score": val_df_h_score,
        "test_df_h_score": test_df_h_score,
    }
).plot(kind="bar", legend=False, title="R2 Score of Ridge Model")


# In[ ]:




