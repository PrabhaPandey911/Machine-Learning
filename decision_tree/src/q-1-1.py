#!/usr/bin/env python
# coding: utf-8

# In[91]:


from __future__ import division
import numpy as np
import pandas as pd
import sys
# filename=raw_input("enter file name:")
# testname=raw_input("test file: ")


# In[92]:


data=pd.read_csv('../input_data/train.csv')
# print data
test=pd.read_csv('../input_data/sample_test.csv')
#splitting randomly in 80:20 ratio
data = data.sample(frac=1)
train, validate = np.split(data, [int(.8*len(data))])


# In[93]:


#calculate entropy of 'left' for complete dataset under consideration
def calculate_entropy(train):
    values=train.left.unique()
    entropy_node=0
    for v in values:
        num=train.left.value_counts()[v]
        deno=len(train.left)
        fraction=float(num)/deno
        if fraction!=0:
            entropy_node+=-(fraction*np.log2(fraction))
    return entropy_node


# In[94]:


#For the purpose of training only on categorical data
df=pd.DataFrame(train,columns=['Work_accident','left','promotion_last_5years','sales','salary'])


# In[95]:


#if outlook is made root then find the entropy_attribute for overcast,sunny and rainy
def entropy_attribute(df,attribute,original):
    variables=df[attribute].unique()
    values=df[original].unique()
    entropy_attribute=0
    I=0
    for v in variables:
        entropy_attribute=0
        n=df[attribute].value_counts()[v]
        d=len(df[attribute])
        for x in values:
            num=len(df[attribute][df[attribute]==v][df[original]==x])
            deno=len(df[attribute][df[attribute]==v])
            fraction=float(num)/deno
            if fraction!=0:
                entropy_attribute+=-(fraction*np.log2(fraction))
        I+=(float(n)/d)*entropy_attribute
    return I


# In[96]:


columns=['Work_accident','left','promotion_last_5years','sales','salary']


# In[97]:


#select the column causing max gain
def max_gain(df):
    p=""
    m=0
    for i in df.columns:
        gain=0
        if i!='left':
            gain=calculate_entropy(df)-entropy_attribute(df,i,'left')
            if m < gain:
                m=gain
                p=i
    return m,p


# In[98]:


class decisionTree():
    def __init__(self,name,df):
        self.lable=name
        self.child={}
        self.positive=len(df[name][df['left']==1])
        self.negative=len(df[name][df['left']==0])
        self.isLeaf=False
    def set_child(self,ch):
        self.child=ch


# In[99]:


def buildTree(df):
    if len(df.columns)<=1:
        leaf=decisionTree('left',df)
        leaf.isLeaf=True
        return leaf
    
    #select the label having highest gain and make it root
    gain,lable=max_gain(df)
    es=calculate_entropy(df)
#     print gain,es
#     if gain==0 and es!=0:
#         print "yes i am here"

    #if gain==0 then exit
    if gain==0:
        leaf=decisionTree('left',df)
        leaf.isLeaf=True
        return leaf
    
    
    root=decisionTree(lable,df)
    
    #child for outlook would be outcast, sunny and rainy
    childs=df[lable].unique()
    children={}
    df2=df
    for i in childs:
        rows=df2[df2[lable]==i]
        rows=rows.drop(columns=[lable])
        ch_root=buildTree(rows)
        children[i]=ch_root
    root.set_child(children)
    return root


# In[100]:


root=buildTree(df)


# In[101]:


def traverse(root):
    if len(root.child)==0:
        print "return root: ",root.lable
        return
    
    print "Root: ",root.lable, root.isLeaf#, root.child
    
    for k,v in root.child.items():
        print "root: ",root.lable, "key: ",k
        if v!=None:
            traverse(v)


# In[102]:


def predict(model,X):
    root=model
    row=X
    #if leaf is reached then declare result
    if root.isLeaf==True:
        if root.positive > root.negative:
            return "YES"
        else:
            return "NO"
        
    
    row1=row
    
    #go to the children of selected node
    ch_node=root.child[row1[root.lable]]
    
    if ch_node!=None:
        if ch_node.lable=='left':
            if ch_node.positive>ch_node.negative:
                return "YES"
            else:
                return "NO"
    #if child_node == None, then declare result
    if ch_node==None:
        if root.positive > root.negative:
            return "YES"
        else:
            return "NO"
    return predict(ch_node,row)


# In[103]:


def calculate_accuracy(df):
    tn=0
    tp=0
    fn=0
    fp=0
    for index,rows in df.iterrows():
        predicted_val=predict(root,rows)
        if rows['left']==1 and predicted_val=="YES":
            tp+=1
        if rows['left']==0 and predicted_val=="NO":
            tn+=1
        if rows['left']==1 and predicted_val=="NO":
            fn+=1
        if rows['left']==0 and predicted_val=="YES":
            fp+=1
    accuracy=((tp+tn)/(tp+tn+fp+fn))*100
    if tp+fn!=0:
        recall=(tp/(tp+fn))*100
    else:
        recall=0
    if tp+fp!=0:
        precision=(tp/(tp+fp))*100
    else:
        precision=0
    if recall!=0 and precision!=0:
        f1score=(2/((1/recall)+(1/precision)))
    else:
        f1score=0
    print "True Positive: "+str(tp)
    print "True Negative: "+str(tn)
    print "False Positve: "+str(fp)
    print "False Negative: "+str(fn)
    print "accuracy: "+str(accuracy)+"%"
    print "precision: "+str(precision)+"%"
    print "recall: "+str(recall)+"%"
    print "f1 score: ",f1score
    return 
    


# In[104]:


calculate_accuracy(validate)


# In[105]:


for index, row in test.iterrows():
    print predict(root,row)


# In[ ]:




