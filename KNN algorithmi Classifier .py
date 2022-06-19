#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV
from sklearn.preprocessing import OrdinalEncoder,StandardScaler,LabelEncoder
from sklearn.neighbors import KNeighborsClassifier


# In[3]:


df = pd.read_csv("data/car.csv",header=None)
df


# In[4]:


df.isna().sum()


# In[5]:


df.info()


# In[6]:


df.describe()


# In[9]:


df[0].unique(),df[1].unique(),df[2].unique(),df[3].unique(),df[4].unique(),df[5].unique(),df[6].unique()


# # Encoder

# In[10]:


ordinal = OrdinalEncoder()


# In[11]:


ord_0 = ordinal.fit_transform(df[[0]])
ord_1 = ordinal.fit_transform(df[[1]])
ord_2 = ordinal.fit_transform(df[[2]])
ord_3 = ordinal.fit_transform(df[[3]])
ord_4 = ordinal.fit_transform(df[[4]])
ord_5 = ordinal.fit_transform(df[[5]])


# In[12]:


ord_0 = ord_0.reshape(-1)
ord_1 = ord_1.reshape(-1)
ord_2 = ord_2.reshape(-1)
ord_3 = ord_3.reshape(-1)
ord_4 = ord_4.reshape(-1)
ord_5 = ord_5.reshape(-1)


# In[14]:


labels = df[6].replace(['unacc', 'good', 'vgood', 'acc'],[0,1,2,3])


# In[21]:


labels.values


# In[18]:


df_n = pd.DataFrame({
    "ord_0" : ord_0,
    "ord_1" : ord_1,
    "ord_2" : ord_2,
    "ord_3" : ord_3,
    "ord_4" : ord_4,
    "ord_5" : ord_5,
})
df_n


# In[22]:


X = df_n.values
Y = labels.values


# In[23]:


x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size = 0.15,random_state = 42)


# # KNN - eng yaqin qoshnilar bilan aniqlash algoritmi

# In[27]:


KNN = KNeighborsClassifier(n_neighbors=8)


# In[80]:


dicti = dict(n_neighbors=np.arange(1,100))
grid = GridSearchCV(KNN,dicti,scoring="accuracy",cv=10)
result = grid.fit(x_train,y_train)
result.best_params_


# In[32]:


result.best_score_


# In[33]:


KNN.fit(x_train,y_train)


# In[34]:


KNN.score(x_test,y_test)


# In[35]:


preds = KNN.predict(x_test)
preds


# In[46]:


pd.DataFrame({
    "Actual " :y_test,
    "Prediction" : preds
})


# In[48]:


plt.figure(figsize=(10,10))
plt.plot(preds,"*",color='g')
plt.plot(y_test,"+",color='black')
plt.show()


# In[77]:


knn = KNeighborsClassifier(n_neighbors=8)


# # Cross validation,f1_score,

# In[78]:


CVS = cross_val_score(knn,x_train,y_train,cv=10)
CVS


# In[79]:


CVS.mean()

