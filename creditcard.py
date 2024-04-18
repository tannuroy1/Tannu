#!/usr/bin/env python
# coding: utf-8

# In[1]:



import pandas as pd
import matplotlib.pyplot as plt
import sklearn as sk
import numpy as np
import seaborn as sns


# In[2]:


df=pd.read_csv("E:\creditcard aa.csv")


# In[3]:


df


# In[4]:


df.info()


# In[5]:


df.describe()


# In[6]:


df.isnull().sum().sum()


# In[7]:


df['Class'].value_counts()


# In[21]:


df1=df[df.Class==0]
df2=df[df.Class==1]


# In[22]:


print(df1.shape)


# In[23]:


print(df2.shape)


# In[24]:


df1


# In[11]:


x=df.drop('Class',axis=1)


# In[12]:


y=df.Class


# In[13]:


from sklearn.model_selection import train_test_split


# In[14]:


xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2)


# In[15]:


xtrain


# In[16]:


from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()


# In[17]:


x_scaler=scaler.fit_transform(x)


# In[18]:


from sklearn.linear_model import LogisticRegression


# In[19]:


model=LogisticRegression()


# In[20]:


model.fit(xtrain,ytrain)


# In[27]:


model.score(xtest,ytest)


# In[28]:


model.predict(xtest)


# In[ ]:




