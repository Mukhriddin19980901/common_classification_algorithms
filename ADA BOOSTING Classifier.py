#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import plot_tree,DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sb
from sklearn.preprocessing import OrdinalEncoder,LabelEncoder


# In[6]:


DF = pd.read_csv('../../../data/Iris.csv')
DF


# In[645]:


dataf= DF
dataf.Species.value_counts()


# In[563]:


label = LabelEncoder()
DF["Species"]=label.fit_transform(DF["Species"])


# In[564]:



x_train = DF.drop("Species",axis=1).values
y_train = DF["Species"].values


# In[565]:


DF


# In[566]:


x_train


# In[567]:


y_train


# In[568]:


ada=AdaBoostClassifier()


# In[569]:


ada.fit(x_train,y_train)


# In[570]:


plt.figure(figsize=(20,20))
plot_tree(ada[0],feature_names = DF.drop('Species',axis=1).columns,filled=True)
plt.show()


# In[571]:


DF["Species"] = DF["Species"].replace(0,-1)
DF


# # 1-step

# In[572]:


DF['weight_1'] = 1/150


# In[573]:


df_1 = DF.sample(len(DF),replace=True,weights=DF['weight_1'],random_state=42)
df_1


# In[574]:


x_train = df_1.drop(["Species","weight_1"],axis=1).values
y_train = df_1["Species"].values


# In[575]:


y_train


# In[576]:


dnt = DecisionTreeClassifier(max_depth=1)


# In[577]:


dnt.fit(x_train,y_train)
plot_tree(dnt)
plt.show()


# # TEST

# In[578]:


x_test = DF.drop(['Species',"weight_1"],axis=1).values
y_test = DF["Species"].values


# In[579]:


pred_1=dnt.predict(x_test)
pred_1


# In[580]:


acc_1 = dnt.score(x_test,y_test)
acc_1


# In[581]:


sb.heatmap(confusion_matrix(y_test,pred_1),annot=True,linewidth=1,square=True)


# In[582]:


DF["predict_1"] = pred_1


# In[583]:


DF


# In[584]:


DF.loc[DF.Species==DF.predict_1,"Loss_1"]=0
DF.loc[DF.Species!=DF.predict_1,"Loss_1"]=1
DF


# # Next 2 

# In[585]:


error = sum(DF.Loss_1 * DF.weight_1)
error


# In[586]:


coef_1 = 0.5 * np.log((1-error)/error)  # nima uchun topyapman
coef_1


# # coeff 
#  >                      sum(x*y)- sum(x) * sum(y) 
#     r = ---------------------------------------------------------------------
#          sqrt([n * (sum(x**2)) - (sum(x)**2)] * [n * (sum(y**2))-(sum(y)**2)])
#        

# In[587]:


w_2 = DF["weight_1"] * np.exp(-1 * coef_1 * DF['Species'] * DF["predict_1"])
DF['weight_2'] = w_2 / sum(w_2)


# In[588]:


DF


# In[589]:


df_2 = DF.sample(len(DF),replace=True,weights=DF["weight_2"],random_state=42)
df_2


# In[590]:


x_train = df_2.iloc[:,:5].values
y_train = df_2['Species'].values


# In[591]:


x_train


# In[592]:


dt = DecisionTreeClassifier(max_depth=1)
dt.fit(x_train,y_train)
plot_tree(dt)
plt.show()


# In[593]:


x_test[0]


# In[594]:


pred_2 = dt.predict(x_test)
pred_2


# In[595]:


acc_2 = dt.score(x_test,y_test)
acc_2


# In[596]:


sb.heatmap(confusion_matrix(y_test,pred_2),annot=True,linewidths=1,square=True)
plt.show()


# In[597]:


DF["predict_2"] = pred_2


# In[598]:


DF


# In[599]:


DF.loc[DF.Species!=DF.predict_2,"Loss_2"] = 1
DF.loc[DF.Species==DF.predict_2,"Loss_2"] = 0


# In[600]:


DF["Loss_2"].value_counts()


# # Next 3

# In[601]:


error_2 = sum(DF['Loss_2'] * DF['weight_2'])
error_2


# In[602]:


coef_2 = 0.5 * np.log((1-error_2)/error_2)
coef_2


# In[603]:


w_3 = DF["weight_2"] * np.exp(-1 * coef_2 * DF['Species'] * DF["predict_2"])
DF['weight_3'] = w_3 / sum(w_3)


# In[604]:


DF


# In[605]:


df_3 = DF.sample(len(DF),replace=True,weights=DF["weight_3"],random_state=42)
df_3


# In[606]:


x_train = df_3.iloc[:,:5].values
y_train = df_3['Species'].values


# In[607]:


dt = DecisionTreeClassifier(max_depth=1)
dt.fit(x_train,y_train)
plot_tree(dt)
plt.show()


# In[608]:


pred_3 = dt.predict(x_test)
pred_3


# In[609]:


acc_3 = dt.score(x_test,y_test)
acc_3


# In[610]:


sb.heatmap(confusion_matrix(y_test,pred_3),annot=True,linewidths=1,square=True)
plt.show()


# In[611]:


DF["predict_3"] = pred_3


# In[612]:


DF.loc[DF.Species!=DF.predict_3,"Loss_3"] = 1
DF.loc[DF.Species==DF.predict_3,"Loss_3"] = 0
DF


# # Next

# In[613]:


error_3 = sum(DF['Loss_3'] * DF['weight_3'])
error_3


# In[614]:


coef_3 = 0.5 * np.log((1-error_3)/error_3)
coef_3


# In[615]:


w_4 = DF["weight_3"] * np.exp(-1 * coef_3 * DF['Species'] * DF["predict_3"])
DF['weight_4'] = w_4 / sum(w_4)


# In[616]:


df_4 = DF.sample(len(DF),replace=True,weights=DF["weight_4"],random_state=42)
df_4


# In[617]:


x_train = df_4.iloc[:,:5].values
y_train = df_4['Species'].values
dt = DecisionTreeClassifier(max_depth=1)
dt.fit(x_train,y_train)
plot_tree(dt)
plt.show()


# In[618]:


pred_4 = dt.predict(x_test)
pred_4


# In[619]:


acc_4 = dt.score(x_test,y_test)
acc_4


# In[620]:


sb.heatmap(confusion_matrix(y_test,pred_4),annot=True,linewidths=2,square=True)
plt.show()


# In[621]:


DF["predict_4"] = pred_4
DF.loc[DF.Species!=DF.predict_4,"Loss_4"] = 1
DF.loc[DF.Species==DF.predict_4,"Loss_4"] = 0
DF


# # Next 5

# In[622]:


error_4 = sum(DF['Loss_4'] * DF['weight_4'])
error_4


# In[623]:


coef_4 = 0.5 * np.log((1-error_4)/error_4)
coef_4


# In[624]:


w_5 = DF["weight_4"] * np.exp(-1 * coef_4 * DF['Species'] * DF["predict_4"])
DF['weight_5'] = w_5 / sum(w_5)


# In[625]:


df_5 = DF.sample(len(DF),replace=True,weights=DF["weight_5"],random_state=42)
df_5


# In[626]:


x_train = df_5.iloc[:,:5].values
y_train = df_5['Species'].values
dt = DecisionTreeClassifier(max_depth=1)
dt.fit(x_train,y_train)
plot_tree(dt)
plt.show()


# In[627]:


pred_5 = dt.predict(x_test)
pred_5


# In[628]:


sb.heatmap(confusion_matrix(y_test,pred_5),annot=True,linewidths=2,square=True)
plt.show()


# In[629]:


DF["predict_5"] = pred_5
DF.loc[DF.Species!=DF.predict_5,"Loss_5"] = 1
DF.loc[DF.Species==DF.predict_5,"Loss_5"] = 0
DF


# In[630]:


error_5 = sum(DF['Loss_5'] * DF['weight_5'])
error_5


# In[631]:


coef_5 = 0.5 * np.log((1-error_5)/error_5)
coef_5


# In[640]:


print(coef_1)
print(coef_2)
print(coef_3)
print(coef_4)
print(coef_5)


# In[641]:


result = coef_1 * DF.predict_1 + coef_2 * DF.predict_2 + coef_3 * DF.predict_3 + coef_4 * DF.predict_4 + coef_5 * DF.predict_5 
result


# In[649]:


a=pd.DataFrame(result)
a.value_counts()


# In[635]:


DF['Prediction'] = results


# In[636]:


sb.heatmap(confusion_matrix(y_test,results),annot=True,linewidths=1,square=True)


# In[639]:


results.value_counts()

