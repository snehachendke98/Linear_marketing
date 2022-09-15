#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# In[3]:


data = pd.read_csv(r"C:\Users\User\Desktop\Python Code\Linear Regression by Irfan\Advertising.csv") #for an earlier version of Excel, you may need to use the file extension of 'xls'


# In[4]:


X = new.drop('sales', axis= 1)

y = new['sales']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test =train_test_split(X, y, train_size=0.7 , random_state=100)


# In[5]:


reg = LinearRegression()
reg.fit(X_train,y_train)


# In[7]:


pickle.dump(reg, open('model_linear.pkl','wb'))
model = pickle.load(open('model_linear.pkl','rb'))
print(model.predict([[80, 1770000, 6000, 85]]))


