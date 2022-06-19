#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree,DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sb
from sklearn.preprocessing import OrdinalEncoder,LabelEncoder


# In[3]:


df = pd.read_csv("../../../data/car.csv",header =None)
df = pd.DataFrame({
    "average_cost" : df[0],
    "cost" : df[1],
    "doors" : df[2],
    "seats" : df[3],
    'lug_boot' : df[4],
    "safety" : df[5],
    "decision" : df[6]
})
df


# In[4]:


df.info()


# In[5]:


df.describe()


# In[6]:


label = LabelEncoder()
ordinal = OrdinalEncoder()
df[["average_cost","cost","doors","seats","lug_boot","safety"]]=ordinal.fit_transform(df[["average_cost","cost","doors","seats","lug_boot","safety"]])
df['decision'] = label.fit_transform(df['decision'])
df


# In[7]:


x_train = df.drop("decision",axis=1).values
y_train = df['decision'].values


# In[8]:


dtr = DecisionTreeClassifier(min_samples_leaf=2,min_impurity_decrease=10.0,max_depth=6)
dtr.fit(x_train,y_train)
columns = df.drop("decision",axis=1).columns
columns


# In[9]:


plt.figure(figsize=(30,10))
plot_tree(dtr,feature_names=columns,filled=True)
plt.show()


# In[10]:


df_test = df.sample(1728,replace=True,random_state=42)
df_test


# In[11]:


x_test = df_test.drop("decision",axis=1).values
y_test = df_test["decision"].values
x_test


# In[12]:


preds = dtr.predict(x_test)
preds


# In[13]:


dtr.score(x_test,y_test)


# In[14]:


sb.heatmap(confusion_matrix(y_test,preds),annot=True,linewidth=2,square=True)


# In[23]:


plt.plot(preds,"+",color="r")
plt.plot(y_test,",",color = 'g')
plt.show()


# In[ ]:




