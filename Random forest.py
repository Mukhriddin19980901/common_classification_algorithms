#!/usr/bin/env python
# coding: utf-8

# In[118]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import plot_tree
import pandas as pd
import numpy as np
import seaborn as sb
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split , GridSearchCV


# In[101]:


df = pd.read_csv("../../datasets/concrete.csv")
df


# In[102]:


df.info()


# In[103]:


X = df.drop("strength",axis=1).values
Y = df["strength"].values


# In[104]:


x_test = (df.drop("strength",axis=1)).sample(len(df),replace=True,random_state=42)
y_test = df.strength[x_test.index].values
x_test = x_test.values


# In[120]:


rand_f = RandomForestRegressor(n_estimators=49,min_impurity_decrease = 10,min_samples_leaf = 5 ,max_depth=5)


# In[117]:


# Estimatorlar sonini maximal aniqlikda taxmin qilish uchun
from sklearn.model_selection import GridSearchCV
dicti = dict(n_estimators=np.arange(1,100))
cv = GridSearchCV(rand_f,dicti,cv=10)
cv.fit(X,Y)
cv.best_score_


# In[119]:


cv.best_estimator_


# In[121]:


rand_f.fit(X,Y)


# In[122]:


predict=rand_f.predict(x_test)


# In[123]:


rand_f.score(x_test,y_test)


# In[124]:


plt.figure(figsize=(30,20))
plot_tree(rand_f[0],feature_names=df.drop("strength",axis=1).columns,filled=True)
plt.show()    


# In[126]:


pd.DataFrame(y_test).value_counts()


# In[127]:


x_y = [[150.1,0.0,67.9,182.0,7.1,979.4,824.7,28]]
rand_f.predict(x_y)


# In[128]:


X1 = df.drop("strength",axis=1).values
Y1 = df["strength"].values


# In[129]:


x_train1,x_test1,y_train1,y_test1 = train_test_split(X1,Y1,test_size=0.2,random_state=42)
rand_f.fit(x_train1,y_train1)


# In[130]:


pred = rand_f.predict(x_test1)
pred


# In[131]:


rand_f.score(x_test1,y_test1)


# In[132]:


plt.plot(x_test1,y_test1,'.',color='red')
plt.plot(x_test1,pred,"+",color='black')
plt.show()


# In[133]:


plt.figure(figsize=(30,20))
plot_tree(rand_f[2],feature_names=df.drop("strength",axis=1).columns,filled=True)
plt.show()    


# In[ ]:




