#!/usr/bin/env python
# coding: utf-8

# # q-2
# 
# #q-2-1

# In[22]:


import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt


# In[23]:


data=pd.read_csv("../input_data/AdmissionDataset/data.csv")
# data=data.sample(frac=1)
columns=['GRE Score','TOEFL Score','University Rating','SOP','LOR ' ,'CGPA','Research']


def normalise(data,columns):
    for i in columns:
        max_value=data[i].mean()
        min_value=data[i].std()
        data[i]=(data[i]-max_value)/min_value
#         print data[i]
    return data
        
data=normalise(data,columns)

data.insert(0,'all_ones',1)


train, validate = np.split(data,[int(.8*len(data))])
train=train.drop(['Serial No.'],axis=1)

yactual=train['Chance of Admit ']

temp_train=train
train=train.drop(['Chance of Admit '],axis=1)

y_validate=validate['Chance of Admit ']
validate=validate.drop(['Serial No.'],axis=1)
temp_validate=validate
validate=validate.drop(['Chance of Admit '],axis=1)

# print train

theta=np.zeros(8)#7 for columns + 1 for beta0


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


theta1,cost1,iterations=gradientDescent(train,yactual,theta,0.001)
theta2,cost2,i1=gradientDescent(train,yactual,theta,0.005)
theta3,cost3,i2=gradientDescent(train,yactual,theta,0.01)
theta4,cost4,i3=gradientDescent(train,yactual,theta,0.1)


def plotgraph(cost1,cost2,cost3,cost4):
    plt.figure(figsize=(7,7))
    plt.plot(iterations,cost1,label="learning rate=0.001")
    plt.plot(iterations,cost2,label="learning rate=0.005")
    plt.plot(iterations,cost3,label="learning rate=0.01")
    plt.plot(iterations,cost4,label="learning rate=0.1")
    plt.legend(loc='best')
    plt.xlabel("Iterations")
    plt.ylabel("Costs")
    
plotgraph(cost1,cost2,cost3,cost4)
   
    


# In[24]:


#as theta4 is giving steep change
#hence selecting theta4
y_last=np.dot(validate,theta4)

probability=1/(1+np.exp(-y_last))
mean_value=yactual.mean() #mean wrt train dataset as threshold is based on that only
predicted=[]
for p in range(0,len(probability)):
    if probability[p] > 0.3:
        predicted.append(1)
    else:
        predicted.append(0)


actual=[]
y_validate=y_validate.tolist()
for p in range(0,len(y_validate)):
    if y_validate[p] > 0.3:
        actual.append(1)

    else:
        actual.append(0)

def calculate_accuracy(probability,y_validate):
    tp=0
    tn=0
    fp=0
    fn=0
    for i in range(len(y_validate)):
        if probability[i]==1 and y_validate[i]==1:
            tp+=1
        if probability[i]==1 and y_validate[i]==0:
            fp+=1
        if probability[i]==0 and y_validate[i]==0:
            tn+=1
        if probability[i]==0 and y_validate[i]==1:
            fn+=1
    print "TP: ",tp
    print "TN: ",tn
    print "FP: ",fp
    print "FN: ",fn
    accuracy=float(tp+tn)/(tp+tn+fp+fn)
    precision=float(tp)/float(tp+fp)
    recal=float(tp)/float(tp+fn)
    f1score=2/((1/recal)+(1/precision))
    return accuracy,precision,recal,f1score

accuracy,precision,recal,f1score= calculate_accuracy(predicted,actual)
print "accuracy: ",accuracy
print "precision: ",precision
print "recall: ",recal
print "f1 score: ",f1score


# # q-2-2

# In[25]:


#KNN:
import math


# In[26]:


def predict(row,k,t,dis_func):
    #t: train data set
    #dis_func: distance function used
    #k: k value provided 
    #row: point under consideration
    
    #get the sorted list according to the function passed
    #structure of list, tuple of distance and corresponding class lable
    distance=dis_func(t,row)
#     print distance
    #knn is a list of size 2, zeroth index represents '0' of class lable, first index represents '1' of class lable
    knn=[0,0]
    
    #take the first k distances and calculate the total number of times '0' and '1' appearing in class lable
    for i in range(0,k):
        knn[distance[i][1]]+=1
    
    #return the class label which occured most number of times
    if knn[1]>=knn[0]:
        return 1 
    else:
        return 0


# In[27]:


def euclidean(t,row):
    distance=[]
    columns=['GRE Score','TOEFL Score','University Rating','SOP','LOR ' ,'CGPA','Research']
    
    #t is the training data, to find the distance with all points, a row equals a point
    #row is the given point with which distance is to be calculated
    for index1,row1 in t.iterrows():
        temp=0
        #for one row, iterate over all the columns and sum up the square of difference (formula for euclidean)
        for col in columns:
            temp+=((row1[col]-row[col])**2) 
        
        #take square root of the above aggregate, this is the euclidean distance between current considered points
        #save the distance and corresponding class of the training row under consideration, in a list "distance"
        distance.append((math.sqrt(temp),int(row1['new_col'])))
    
    
    #sorting all the distances in increasing order
    distance.sort()
    
    #return the distance list
    return distance


# In[28]:


#calculate accurancy for training data t and a given function func

def calculate_accuracy(t,func,validate):
    l=len(t)
    l=int(math.sqrt(l))
    l+=1
#     print t
    #for plotting graph
    k_level=[]
    accu_list=[]
    max_acc=-sys.maxint-1
    max_prec=0
    max_recall=0
    max_f1sc=0
    final_k=0
    #range of k is from 1 to square root of the number of rows of training data set provided
    for i in range(1,l,2):    #as k value should always be odd, therefore step size is equal to 2 (third arg)
        #for confusion matrix
        actual_value=[]
        predicted_value=[]
        tp=0
        tn=0
        fp=0
        fn=0
        #for each row in validate
        for index,row in validate.iterrows(): 
            x=predict(row,i,t,func)
            if row['new_col']==1 and x==1:
                tp+=1
            if row['new_col']==0 and x==0:
                tn+=1
            if row['new_col']==1 and x==0:
                fn+=1
            if row['new_col']==0 and x==1:
                fp+=1
            #for confusion matrix
            actual_value.append(row['new_col'])
            predicted_value.append(x)
            
        #for confusion matrix
        actu = pd.Series(actual_value, name='Actual')
        pred = pd.Series(predicted_value, name='Predicted')
        df_confusion = pd.crosstab(actu, pred)
        
        if tp+tn+fp+fn!=0:
            acc_cm=float(tp+tn)/(tp+tn+fp+fn)#accuracy_cm(df_confusion)
        else:
            acc_cm=0
        if tp+fn!=0:
            recal_cm=float(tp)/(tp+fn)#recall_cm(df_confusion)
        else:
            recal_cm=0
        if tp+fp!=0:
            preci_cm=float(tp)/(tp+fp)#precision_cm(df_confusion)
        else:
            preci_cm=0
        if recal_cm!=0 and preci_cm!=0:
            f1_sc=(2/((1/recal_cm)+(1/preci_cm)))#f1Score_cm(preci_cm,recal_cm)
        else:
            f1_sc=0
        #to plot the graph between different k values and its corresponding accuracy
        k_level.append(i)
        accu_list.append(acc_cm)
        if max_acc<acc_cm:
            max_acc=acc_cm
            final_k=i
            max_prec=preci_cm
            max_recall=preci_cm
            max_f1sc=preci_cm
    return (k_level,accu_list,final_k,max_acc,max_prec,max_recall,max_f1sc)


# In[29]:


new_col_t=[]
for i in temp_train['Chance of Admit ']:
    if i > 0.5: 
        new_col_t.append(1)
    else:
        new_col_t.append(0)

temp_train['new_col']=new_col_t
        
new_col_v=[]
for i in temp_validate['Chance of Admit ']:
    if i > 0.5:
        new_col_v.append(1)
    else:
        new_col_v.append(0)
        
temp_validate['new_col']=new_col_v

k_level,accu_list,final_k,max_acc,max_prec,max_recall,max_f1sc= calculate_accuracy(temp_train,euclidean,temp_validate)
print "k-value: ",final_k
print "maximum accuracy: ",max_acc
print "Precision: ",max_prec
print "Recall: ",max_recall
print "F1 Score: ",max_f1sc


# # q-2-3
# 
# #graph

# In[30]:


threshold_list=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
prec_list=[]
recal_list=[]

for i in threshold_list:
    y_last=np.dot(validate,theta4)
    probability=1/(1+np.exp(-y_last))
    predicted=[]
    for p in range(0,len(probability)):
        if probability[p] > i:
            predicted.append(1)
        else:
            predicted.append(0)
    
    actual=[]
    for p in range(0,len(y_validate)):
        if y_validate[p] > i:
            actual.append(1)
        else:
            actual.append(0)
            
    tp=0
    tn=0
    fp=0
    fn=0
    for i1 in range(len(actual)):
        if predicted[i1]==1 and actual[i1]==1:
            tp+=1
        if predicted[i1]==1 and actual[i1]==0:
            fp+=1
        if predicted[i1]==0 and actual[i1]==0:
            tn+=1
        if predicted[i1]==0 and actual[i1]==1:
            fn+=1   
    if (tp+tn+fp+fn)==0:
        accuracy=0
    else:
        accuracy=float(tp+tn)/(tp+tn+fp+fn)
    if tp+fp==0:
        precision=0
    else:
        precision=float(tp)/float(tp+fp)
        
    if tp+fn==0:
        recal=0
    else:
        recal=float(tp)/float(tp+fn)
    if recal==0 or precision==0:
        f1score=0
    else:
        f1score=2/((1/recal)+(1/precision))
    prec_list.append(precision)
    recal_list.append(recal)
    print "----------------------------------------"
    print tp,tn,fp,fn
    print "threshold: ",i
    print "accuracy: ",accuracy
    print "Precision: ",precision
    print "Recall: ",recal
    print "F1 Score: ",f1score
    print "----------------------------------------"


# In[31]:


plt.figure(figsize=(7,7))
plt.plot(threshold_list,prec_list,label="Prediction")
# plt.plot(threshold_list,recal_list,label="Recall")
plt.legend(loc='best')
plt.xlabel("Threshold")
plt.ylabel("Precision")


# In[32]:


plt.figure(figsize=(7,7))
# plt.plot(threshold_list,prec_list,label="Prediction")
plt.plot(threshold_list,recal_list,label="Recall")
plt.legend(loc='best')
plt.xlabel("Threshold")
plt.ylabel("Recall")


# # Criteria one should use while deciding the threshold value.
# 
# 

# In the case of a logistic regression classifier, we can adjust something called the threshold, 
# which is an internal number between 0 and 1 that determines whether a prediction is positive or not.
# 
# A threshold value which maximises both sensitivity (recall) and specificity (precision) is the desired value.
# 
# We can plot both of them against the threshold value and choose the intersection point as the threshold value.
# 
# In the given data set, we get maximum accuracy at 0.3, hence it is choosen as threshold.
