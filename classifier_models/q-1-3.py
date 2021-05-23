#!/usr/bin/env python
# coding: utf-8

# In[17]:


import numpy as np
import pandas as pd

filename="./AdmissionDataset/data.csv"
data = pd.read_csv(filename)
data = data.sample(frac=1)
train, validate = np.split(data, [int(.8*len(data))])


# In[18]:


temp=train #for later use
train=train.drop(['Serial No.'],axis=1)
df=train
df=df.drop(['Chance of Admit '],axis=1)


# In[19]:


matrix=df.values
matrix=np.insert(matrix,0,1.,axis=1)
var= (np.linalg.inv((matrix.T).dot(matrix)))
var2= var.dot(matrix.T)
beta=var2.dot(train['Chance of Admit '].values)


# # Probabilities for validate

# In[20]:


tempv=validate #for later use
validate=validate.drop(['Serial No.'],axis=1)
dfv=validate
original=validate['Chance of Admit ']
original=original.tolist()
dfv=dfv.drop(['Chance of Admit '],axis=1)
matrixv=dfv.values
matrixv=np.insert(matrixv,0,1.,axis=1)
result=np.dot(matrixv,beta)
print result


# # Mean Square Error

# In[21]:


mse=0
for i in range(0,len(result)):
    mse+=((result[i]-original[i])**2)
mse=float(mse)/(len(result))
print mse


# # Mean Absolute Error

# In[22]:


mae=0
for i in range(0,len(result)):
    mae+=abs(result[i]-original[i])
mae=float(mae)/len(result)
print mae


# # Mean Absolute Percentage Error

# In[23]:


mape=0
for i in range(0,len(result)):
    mape+=abs((original[i]-result[i])/original[i])
mape=float(mape)/len(result)
mape*=100
print mape


# # Test File input and prediction: 

# In[24]:


testfile=raw_input("test File: ")
test = pd.read_csv(testfile)
test=test.drop(['Serial No.'],axis=1)
dfv1=test
# original=['Chance of Admit ']
# original=original.tolist()
# dfv=dfv.drop(['Chance of Admit '],axis=1)
matrixv1=dfv1.values
matrixv1=np.insert(matrixv1,0,1.,axis=1)
result1=np.dot(matrixv1,beta)
print result1

