#!/usr/bin/env python
# coding: utf-8

# # Exercise: Iris Dataset
# Now that you have a good understanding of exploratory data analysis and its importance, it's time to put your knowledge to a more practical example. We'll be focusing on an iris public dataset from the scikit-learn library.
# 
# Our main objectives for this dataset are:
# 1. Load the iris dataset into a pandas dataframe
# 2. Create a table summary of the features and target values
# 3. Create a histogram of all the features and target
# 4. Create a correlation matrix of the features and target
# 5. Create scatter plots of all the features and target

# In[1]:


import pandas as pd
import sklearn
from sklearn import datasets
import matplotlib.pyplot as plt


# In[2]:


# Load in the iris dataset
iris = datasets.load_iris()


# In[3]:


# For clarity, the iris dataset is a dictionary with the data and target separated
iris.keys()


# In[4]:


# Create the iris `data` dataset as a dataframe and name the columns with `feature_names`
df = pd.DataFrame(iris["data"], columns=iris["feature_names"])

# Include the target as well
df['target'] = iris["target"]


# In[5]:


# Check your dataframe by `.head()`
df.head()


# In[6]:


# Simple table summary
df.describe()


# In[7]:


# Histogram to show all the data distributions including the target
df.hist()


# In[8]:


# Investigate to see if any data are correlated positively or negatively
df.corr()


# # Scatter Plot Of Features
# Create a scatter plot of the four features against eachother to visualize the results from the correlation matrix
# 1. `sepal length (cm)` vs. `sepal width (cm)`
# 2. `sepal length (cm)` vs. `petal length (cm)`
# 3. `sepal length (cm)` vs. `petal width (cm)`
# 4. `sepal width (cm)` vs. `petal length (cm)`
# 5. `sepal width (cm)` vs. `petal width (cm)`
# 6. `petal length (cm)` vs. `petal width (cm)`

# In[9]:


fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 8))
df.plot(ax=axes[0, 0], x="sepal length (cm)", y="sepal width (cm)", kind="scatter")
df.plot(ax=axes[0, 1], x="sepal length (cm)", y="petal length (cm)", kind="scatter")
df.plot(ax=axes[0, 2], x="sepal length (cm)", y="petal width (cm)", kind="scatter")
df.plot(ax=axes[1, 0], x="sepal width (cm)", y="petal length (cm)", kind="scatter")
df.plot(ax=axes[1, 1], x="sepal width (cm)", y="petal width (cm)", kind="scatter")
df.plot(ax=axes[1, 2], x="petal length (cm)", y="petal width (cm)", kind="scatter")


# # Scatter Plot Of Features And Target
# Create a scatter plot of the four features against the target
# 1. `sepal length (cm)`
# 2. `sepal width (cm)`
# 3. `petal length (cm)`
# 4. `petal width (cm)`

# In[10]:


fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))
df.plot(ax=axes[0, 0], x="sepal length (cm)", y="target", kind="scatter")
df.plot(ax=axes[0, 1], x="sepal width (cm)", y="target", kind="scatter")
df.plot(ax=axes[1, 0], x="petal length (cm)", y="target", kind="scatter")
df.plot(ax=axes[1, 1], x="petal width (cm)", y="target", kind="scatter")

