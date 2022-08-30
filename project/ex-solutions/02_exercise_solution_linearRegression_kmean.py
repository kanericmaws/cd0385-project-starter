#!/usr/bin/env python
# coding: utf-8

# # Exercise: Regression and Classification Machine Learning
# 
# In this exercise, we'll dive deeper into the ML concepts by creating a regression and classification model.
# 
# Your tasks for this exercise are:
# 1. Load the iris dataset into a dataframe
# 2. Create a LinearRegression model and fit it to the dataset
# 3. Score the regression model on the dataset and predict it's values
# 4. Create a RidgeClassifier model and fit it to the dataset, use `alpha=3.0` when initializing the model
# 5. Score the classification model on the dataset and predict it's values

# In[1]:


import numpy as np
import pandas as pd
import sklearn
from sklearn import datasets


# In[2]:


# Load in the iris dataset
iris = datasets.load_iris()


# In[3]:


# Create the iris `data` dataset as a dataframe and name the columns with `feature_names`
df = pd.DataFrame(iris["data"], columns=iris["feature_names"])

# Include the target as well
df['target'] = iris["target"]


# In[4]:


# Check your dataframe by `.head()`
df.head()


# ## Regression ML

# In[5]:


from sklearn.linear_model import LinearRegression


# In[6]:


# Fit a standard regression model, we've done this in other exercises
reg = LinearRegression().fit(df[iris["feature_names"]], df["target"])


# In[7]:


# Score the model on the same dataset
reg.score(df[iris["feature_names"]], df["target"])


# In[8]:


# Predicting values shows they are not that useful to a classification model
reg.predict(df[iris["feature_names"]])


# In[9]:


# If we really wanted to, we could do something like round each regression value to an int
# and have it "act" like a classification model
# This is not required, but something to keep in mind for future reference
reg_cls = np.abs(np.rint(reg.predict(df[iris["feature_names"]])))
reg_cls


# In[10]:


# Evaluate accuracy
sum(reg_cls == df["target"]) / df.shape[0]


# # Classification ML

# In[11]:


from sklearn.linear_model import RidgeClassifier


# In[12]:


# Fit a ridge classifier, which matches with the problem space of being a classification problem
clf = RidgeClassifier(alpha=3.0).fit(df[iris["feature_names"]], df["target"])


# In[13]:


# Score the model
clf.score(df[iris["feature_names"]], df["target"])


# In[14]:


# Predict the class values for the dataset, these will look much better!
clf.predict(df[iris["feature_names"]])

