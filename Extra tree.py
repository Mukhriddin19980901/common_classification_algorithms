#!/usr/bin/env python
# coding: utf-8

# In[7]:


from sklearn.ensemble import ExtraTreesRegressor
from sklearn.tree import plot_tree
import pandas as pd
import numpy as np
import seaborn as sb
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split , GridSearchCV


# In[2]:


df = pd.read_csv("../../datasets/concrete.csv")
df


# In[4]:


X = df.drop("strength",axis=1).values
Y = df["strength"].values


# In[18]:


x_train,x_test ,y_train,y_test= train_test_split(X,Y,test_size=0.2,random_state=42)


# In[19]:


ex =ExtraTreesRegressor(n_estimators=8,min_impurity_decrease=10.0, min_samples_leaf=5,max_depth=3)


# In[16]:


dicti = dict(n_estimators=np.arange(1,100))
cv_x = GridSearchCV(ex,dicti,cv=10)
cv_x.fit(x_train,y_train)
cv_x.best_score_


# In[17]:


cv_x.best_estimator_


# In[20]:


ex.fit(x_train,y_train)


# In[21]:


ex.score(x_test,y_test)


# In[22]:


plt.figure(figsize=(20,20))
plot_tree(ex[0],feature_names=df.drop("strength",axis=1).columns,filled=True)
plt.show()


# In[ ]:




