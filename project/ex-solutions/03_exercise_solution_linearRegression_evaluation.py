#!/usr/bin/env python
# coding: utf-8

# # Exercise: Model Training and Evaluation
# 
# Now that we have the data fundamentals for creating, cleaning, and modifying our datasets, we can train and evaluate a model, in this case it's a linear regression model.
# 
# Your tasks for this exercise are:
# 1. Create a dataframe with the regression dataset, include the features and target within the same dataframe.
# 2. Create a 60% Train / 20% Validation / 20% Test dataset group using the `train_test_split` method.
# 3. Fit the LinearRegression model on the training set.
# 4. Evaluate the model on the validation set.
# 5. Evaluate the model on the test set.

# In[1]:


import pandas as pd
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# In[2]:


regression_dataset = make_regression(
    n_samples=10000,
    n_features=10,
    n_informative=5,
    bias=0,
    noise=40,
    n_targets=1,
    random_state=0,
)


# In[3]:


# Create the dataframe using the dataset
df = pd.DataFrame(regression_dataset[0])
df["target"] = regression_dataset[1]


# In[4]:


# `.head()` to view what the dataset looks like
df.head()


# In[5]:


# train: 0.8 | test: 0.2
df_train, df_test = train_test_split(df, test_size=0.2, random_state=0)

# train: 0.6 | validation: 0.2
df_train, df_val = train_test_split(df_train, test_size=0.25, random_state=0)

# Final dataset sizes: train: 0.6, validation: 0.2, text: 0.2,


# In[6]:


# Output each shape to confirm the size of train/validation/test
print(f"Train: {df_train.shape}")
print(f"Validation: {df_val.shape}")
print(f"Test: {df_test.shape}")


# In[7]:


# Train the linear model by fitting it on the dataframe features and dataframe target
reg = LinearRegression().fit(df_train[list(range(10))], df_train["target"])


# In[8]:


# Evaluate the linear model by scoring it, by default it's the metric r2.
reg.score(df_val[list(range(10))], df_val["target"])


# In[9]:


# Once done optimizing the model using the validation dataset,
# Evaluate the linear model by scoring it on the test dataset.
reg.score(df_test[list(range(10))], df_test["target"])


# In[ ]:




