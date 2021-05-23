#!/usr/bin/env python
# coding: utf-8

# # q-3
# 
# # One v/s All

# In[46]:


import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt

data=pd.read_csv("./input_data/wine-quality/data.csv",sep=';')
columns=["fixed acidity",  "volatile acidity",  "citric acid",  "residual sugar" , "chlorides","free sulfur dioxide", "total sulfur dioxide", "density",    "pH" , "sulphates", "alcohol" ]

def standerdise(data,columns):
    for i in columns:
        data[i]=(data[i]-data[i].mean())/data[i].std()   
    return data

data=standerdise(data,columns)

data.insert(0,'all_ones',1)

quality_set=set(data['quality'].values)
train, validate = np.split(data,[int(.8*len(data))])

yactual=train['quality'].values

temp_train=train
train=train.drop(['quality'],axis=1)

y_validate=validate['quality'].values

temp_validate=validate
validate=validate.drop(['quality'],axis=1)

def change_yactual_and_yvalidate(actual,valid,quality):
#     print actual
#     print 
#     print valid
    res_actual=[]
    res_validate=[]
    for i in range(0,len(actual)):
        if int(actual[i])==quality:
            res_actual.append(1)
        else:
            res_actual.append(0)
            
    for i in range(0,len(valid)):
        if int(valid[i])==quality:
            res_validate.append(1)
        else:
            res_validate.append(0)
            
#     print "quality = ",quality
#     print "yactual: ",
#     print res_actual
#     print "y_validate: ",
#     print res_validate
    return res_actual,res_validate

check_act,check_valid=change_yactual_and_yvalidate(yactual.tolist(),y_validate.tolist(),3)


# In[47]:


def gradientDescent(x,yactual,theta,alpha):
    num_of_rows,cols=np.shape(x)
    transpose = x.transpose()
    costlist=[]
    iterations=[]
    for i in range(0,1000):
        predicted_y = np.dot(x,theta)
        loss_value = predicted_y - yactual
        cost = np.sum(np.square(loss_value))/(2*num_of_rows)
        gradient=np.dot(transpose,loss_value)/num_of_rows
        theta=theta - alpha * gradient
        costlist.append(cost)
        iterations.append(i)
    return theta,costlist,iterations


# In[48]:


theta_wrt_quality={}
for i in quality_set:
    theta=np.zeros(12)#11 for columns + 1 for beta0
    temp_actual,temp_valid=change_yactual_and_yvalidate(yactual.tolist(),y_validate.tolist(),i)
#     theta1,cost1,iterations=gradientDescent(train,temp_actual,theta,0.001)
#     theta2,cost2,i1=gradientDescent(train,temp_actual,theta,0.005)
#     theta3,cost3,i2=gradientDescent(train,temp_actual,theta,0.01)
    theta4,cost4,i3=gradientDescent(train,temp_actual,theta,0.1)
#     print theta4
    theta_wrt_quality[i]=theta4
    
def predict(theta_wrt_quality):
    predict={}
    for quality in theta_wrt_quality.keys():
        y_last=np.dot(validate,theta_wrt_quality[quality])
        probability=1/(1+np.exp(-y_last))
        predict[quality]=probability
    return predict

predict=predict(theta_wrt_quality)
#print predict  #when quality=3 probability for each row of validate is stored as value of key=3


# In[49]:


#make a dictionary which would have quality corressponding max of probabilities told for each of the qualities in predict
predicted_quality_df=pd.DataFrame()
for k in predict.keys():
    predicted_quality_df[k] = predict[k]

# print predicted_quality_df

final_predicted_quality=[]
for index,row in predicted_quality_df.iterrows():
    max_predict_prob=-sys.maxint-1
    max_predict_quality=0
    for i in predicted_quality_df:
        if row[i]>max_predict_prob:
            max_predict_prob=row[i]
            max_predict_quality=i
    final_predicted_quality.append(max_predict_quality)

# print final_predicted_quality


# In[50]:


actu = pd.Series(y_validate, name='Actual')
pred = pd.Series(final_predicted_quality, name='Predicted')


# In[55]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

confumatrix=confusion_matrix(actu.values,pred.values)
print "confusion matrix"
print
print confumatrix
print 
print 'Accuracy Score :',accuracy_score(actu.values, pred.values)
print 'Precision :',precision_score(actu.values, pred.values,average=None).tolist()
print 'Recall :',recall_score(actu.values, pred.values,average=None).tolist()
print 'F1 score :',recall_score(actu.values, pred.values,average=None).tolist()


# # One-v/s-one 

# In[120]:


combinations=[]
rows,col=np.shape(validate)
vote=np.zeros(shape=(rows,11))


for i in quality_set:
    for j in range(i+1,11):
        combinations.append((i,j))

for i in combinations:
    temp_array=[i[0],i[1]]
    temporary_training_data=temp_train.loc[temp_train['quality'].isin(temp_array)]
    if not temporary_training_data.empty:
        for i1, row in temporary_training_data.iterrows():
            ifor_val = 0
            if row['quality']==temp_array[0]:
                ifor_val = 1
            temporary_training_data.set_value(i1,'quality',ifor_val)
         
        yactualovo=temporary_training_data['quality'].values

        temp_trainovo=temporary_training_data
        temporary_training_data=temporary_training_data.drop(['quality'],axis=1)
        
        thetaovo=np.zeros(12)
        theta4ovo,cost4ovo,i3ovo=gradientDescent(temporary_training_data,yactualovo,thetaovo,0.1)
        
        y_lastovo=np.dot(validate,theta4ovo)
        probabilityovo=1/(1+np.exp(-y_lastovo))
        predictovo=probabilityovo
        
        
        predictedovo=[]
        for p in range(0,len(probabilityovo)):
            if probabilityovo[p] >= 0.6:
                predictedovo.append(temp_array[0])
            else:
                predictedovo.append(temp_array[1])

        for values in range(len(predictedovo)):
            vote[values][predictedovo[values]]+=1
            

pridected_quality_ovo=[]
for i in vote.tolist():
    pred_quality=i.index(max(i))
    pridected_quality_ovo.append(pred_quality)
    


confumatrixovo=confusion_matrix(y_validate.tolist(),pridected_quality_ovo)
print "confusion matrix"
print
print confumatrixovo
print 
print 'Accuracy Score :',accuracy_score(y_validate.tolist(),pridected_quality_ovo)
print 'Precision :',precision_score(y_validate.tolist(),pridected_quality_ovo,average=None).tolist()
print 'Recall :',recall_score(y_validate.tolist(),pridected_quality_ovo,average=None).tolist()
print 'F1 score :',recall_score(y_validate.tolist(),pridected_quality_ovo,average=None).tolist()
        

