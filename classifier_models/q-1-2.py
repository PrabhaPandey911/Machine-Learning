#!/usr/bin/env python
# coding: utf-8

# In[81]:


from __future__ import division
import numpy as np
import pandas as pd
import sys
import math
filename="./LoanDataset/data.csv"


# In[82]:


colnames=['id','age','experience','income','zipcode','family_size','spending','education_level','mortgage_value','loan_status','security_account','CD_account','internet_banking','credit_card'] 
data = pd.read_csv(filename, names=colnames,skipinitialspace=True)
cat=['credit_card','internet_banking','CD_account','security_account','education_level','loan_status']
num=['age','experience','income','family_size','spending','mortgage_value','loan_status']


# In[83]:


data.pop('zipcode') #dropping zipcode and id columns
data.pop('id')
data = data.drop(data.index[:1])
print list(data)


# In[84]:


data = data.sample(frac=1)
train, validate = np.split(data, [int(.8*len(data))])
numerical=pd.DataFrame(train,columns=num)
categorical=pd.DataFrame(train,columns=cat)


# In[85]:


def categorical_probability(df):
    probability={}
    total_yes_count=df.loan_status.value_counts()[1]
    total_no_count=df.loan_status.value_counts()[0]
    for i in df.columns:
        if i!='loan_status':
            l={}
            yescount={}
            nocount={}
            for index,rows in df.iterrows():
                if rows[i] in l.keys():
                    l[rows[i]]+=1
                    if rows['loan_status']==0:
                        if rows[i] in nocount.keys():
                            nocount[rows[i]]+=1
                        else:
                            nocount[rows[i]]=1
                    else:
                        if rows[i] in yescount.keys():
                            yescount[rows[i]]+=1
                        else:
                            yescount[rows[i]]=1
                else:
                    l[rows[i]]=1
                    if rows['loan_status']==0:
                        if rows[i] in nocount.keys():
                            nocount[rows[i]]+=1
                        else:
                            nocount[rows[i]]=1
                    else:
                        if rows[i] in yescount.keys():
                            yescount[rows[i]]+=1
                        else:
                            yescount[rows[i]]=1
            for i2 in yescount.keys():
                probability[(i,(i2,1))]=(yescount[i2]/total_yes_count)
            for i3 in nocount.keys():
                probability[(i,(i3,0))]=(nocount[i3]/total_no_count )
    return probability

# categorical_dict=categorical_probability(categorical)
#structure of probability dictionary=>
#key=(attribute_name,(attribute_value,loan_status_value))
#value=probability coressponding to given key 


# In[86]:


def numerical_probability(df):
    mean={}
    standard_deviation={}
    for i in df.columns:
        if i!='loan_status':
            mean[(i,0)]=train[train['loan_status']==0][i].mean()
            mean[(i,1)]=train[train['loan_status']==1][i].mean()
            standard_deviation[(i,0)]=train[train['loan_status']==0][i].std()
            standard_deviation[(i,1)]=train[train['loan_status']==1][i].std()
    return mean,standard_deviation
# numerical_dict=numerical_probability(numerical)


# In[87]:


def gaussian_dist(w,sigma,mean):
    x=(w-mean)**2
    x=x/(2*sigma*sigma)
    x=0-x
    y=math.exp(x)
    deno=math.sqrt(2*(math.pi))
    deno=deno*sigma
    return y/deno


# In[88]:


temp_mean,temp_sd=numerical_probability(numerical)
temp_dict=categorical_probability(categorical)

def predict(row,columns,temp_mean,temp_sd,temp_dict):
    no=1
    yes=1
    total_yes_count=train.loan_status.value_counts()[1]
    total_no_count=train.loan_status.value_counts()[0]
    for i in columns:
        if i!='loan_status':
            if i in numerical:
                mean_val_yes=temp_mean[(i,1)]
                sd_val_yes=temp_sd[(i,1)]
                mean_val_no=temp_mean[(i,0)]
                sd_val_no=temp_sd[(i,0)]
                w=row[i]
                f1=gaussian_dist(w,sd_val_yes,mean_val_yes)
                f0=gaussian_dist(w,sd_val_no,mean_val_no)
                no*=f0
                yes*=f1
            else:
                probability1=temp_dict[(i,(row[i],1))]
                probability0=temp_dict[(i,(row[i],0))]
                no*=probability0
                yes*=probability1
    yes*=(total_yes_count/len(train))
    no*=(total_no_count/len(train))
    if yes>no:
        return 1
    else:
        return 0           


# # Observation

# In[89]:


def calculate_accuracy(df,temp_mean,temp_sd,temp_dict):
    tp=0
    tn=0
    fp=0
    fn=0
    for index,row in df.iterrows():
        pred=predict(row,df.columns,temp_mean,temp_sd,temp_dict)
        if row['loan_status']==1 and pred==1:
            tp+=1
        if row['loan_status']==0 and pred==0:
            tn+=1
        if row['loan_status']==1 and pred==0:
            fn+=1
        if row['loan_status']==0 and pred==1:
            fp+=1
    accuracy=float(tp+tn)/(tp+tn+fp+fn)
    print "True Positive: ",tp
    print "True Negative: ",tn
    print "False Positive: ",fp
    print "False Negative: ",fn
    return accuracy
calculate_accuracy(validate,temp_mean,temp_sd,temp_dict)


# In[90]:


testfile=raw_input("Test File: ")
colnames1=['age','experience','income','zipcode','family_size','spending','education_level','mortgage_value','security_account','CD_account','internet_banking','credit_card']
coltemp=['age','experience','income','family_size','spending','education_level','mortgage_value','security_account','CD_account','internet_banking','credit_card']
test = pd.read_csv(testfile, names=colnames1)#,skipinitialspace=True)
test.pop('zipcode') 


for index,row in test.iterrows():
    print predict(row,coltemp,temp_mean,temp_sd,temp_dict)

