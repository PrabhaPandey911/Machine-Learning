#!/usr/bin/env python
# coding: utf-8

# # Question 1) Part 5:
# In this part implement regression with k-fold cross validation. Analyse how behavior changes with different values of k. Also implement a variant of this which is the
# leave-one-out cross validation.

# In[57]:


import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold


# Read the data set, drop serial number and chance of admit from it and add column of all 1's. Create initial theta of size 8 (7 coulumns + 1 for interceot), initialized by all zeros.

# In[58]:


filename="./AdmissionDataset/data.csv"
data = pd.read_csv(filename)
# data = data.sample(frac=1)

def normalise(data,columns):
    for i in columns:
        mean=data[i].mean()
        std=data[i].std()
        data[i]=(data[i]-mean)/std
    return data

columns=['GRE Score','TOEFL Score','University Rating','SOP','LOR ' ,'CGPA','Research']
data=normalise(data,columns)
data.insert(0,'all_ones',1)
data=data.drop(['Serial No.'],axis=1)

Y=data['Chance of Admit ']
data=data.drop(['Chance of Admit '],axis=1)
data=np.array(data)
print(np.shape(data))


# In[59]:


alpha=0.01
theta=np.zeros(8)#7 for columns + 1 for beta0
lambda_val=0.001


# Gradient decent over ridge regression

# In[60]:


def gradientDescent_ridge(x,yactual,theta,alpha,lambda_val):
    num_of_rows,cols=np.shape(x)
    col_length=np.shape(theta)
    x=np.array(x)
    for i in range(0,1000):
        pred=np.dot(x,theta.T)
        loss_value = pred - yactual
        gradient_0=np.sum(np.dot(x[:,0],loss_value))
        theta[0]=theta[0]-(alpha*(gradient_0/num_of_rows))
        for j in range(1,col_length[0]):
            gradient=np.sum(np.dot(x[:,j],loss_value))
            lamda_part=(2*lambda_val*theta[j]) #differentiation of square of theta[j]
            theta[j]=theta[j] - (alpha * ((gradient+lamda_part)/(num_of_rows)))
    return theta


# For each value of k in the range 2 to 40, call gradient decent, find the theta, and hence find the error.

# In[64]:


k_list=[]
error_list=[]
for k in range(2,41):
    kf = KFold(n_splits=k)
    error_mean=[]
    for train_index,test_index in kf.split(data):
        X_train,X_test = data[train_index],data[test_index]
        Y_train,Y_test = Y[train_index],Y[test_index]
        theta_temp=gradientDescent_ridge(X_train,Y_train,theta,alpha,lambda_val)
        pred=np.dot(theta_temp,X_test.T)
        loss= np.sum(np.square(pred-Y_test))/(np.shape(X_test)[0])
        error_mean.append(loss)
    error_mean=np.array(error_mean)
    mean_loss=np.mean(error_mean)
    print(k,mean_loss)
    k_list.append(k)
    error_list.append(mean_loss)


# In[67]:


plt.figure(figsize=(7,7))
plt.title('Ridge: K-values v/s error')
plt.xlabel('k-Folds', fontsize=18)
plt.ylabel('error', fontsize=16)
plt.plot(k_list,error_list)


# # Observation

# As the value of K increases, the error decreases.
# But after certain k value, the error tends to increase, as now increasing k reduces the number of data samples in each section, hence overfitting is observed.

# # leave-one-out cross validation.

# In[69]:


kf = KFold(n_splits=np.shape(X_test)[0])
error_mean=[]
for train_index,test_index in kf.split(data):
    X_train,X_test = data[train_index],data[test_index]
    Y_train,Y_test = Y[train_index],Y[test_index]
    theta_temp=gradientDescent_ridge(X_train,Y_train,theta,alpha,lambda_val)
    pred=np.dot(theta_temp,X_test.T)
    loss= np.sum(np.square(pred-Y_test))/(2*np.shape(X_test)[0])
    error_mean.append(loss)
error_mean=np.array(error_mean)
mean_loss=np.mean(error_mean)
print("LOOCV Error: ",mean_loss)

