#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import division
import numpy as np
import pandas as pd
import sys
# filename=raw_input("train file:")
# testname=raw_input("test file: ")


# In[2]:


data=pd.read_csv('../input_data/train.csv')
# print data
test=pd.read_csv('../input_data/sample_test.csv')

data = data.sample(frac=1)
train, validate = np.split(data, [int(.8*len(data))])


# In[3]:


numerical=pd.DataFrame(train,columns=['satisfaction_level','last_evaluation','number_project','average_montly_hours','time_spend_company','left'])
categorical=pd.DataFrame(train,columns=['Work_accident','left','promotion_last_5years','sales','salary'])


# In[4]:


def calculate_gini(train):
    if train.size==0:#when data has all 1's 
        return 0
    gini_node=0
    num=len(train['left'][train['left']==0])
    deno=len(train.left)
    fraction=float(num)/deno
    if fraction!=0:
        gini_node=(2*fraction)*(1-fraction)
    return gini_node


# In[5]:


def calculate_misclassificationrate(train):
    values=train.left.unique()
    result=0
    for v in values:
        num=train.left.value_counts()[v]
        deno=len(train.left)
        fraction=float(num)/deno
        result=min(fraction,(1-fraction))
    return result


# In[6]:


def gini_attribute(df,attribute,original):
    variables=df[attribute].unique()
    gini_attribute=0
    for v in variables:
        num=df[attribute][df[attribute]==v][df[original]==0]
        n=len(num)
        deno=df[attribute][df[attribute]==v]
        d=len(deno)
        q=float(n)/d
        gini_attribute+=((2*q*(1-q))*(d/len(df)))
        
    return gini_attribute        


# In[7]:


def misclassification_attribute(df,attribute,original):
    variables=df[attribute].unique()
    gini_attribute=0
    for v in variables:
        num=df[attribute][df[attribute]==v][df[original]==0]
        n=len(num)
        deno=df[attribute][df[attribute]==v]
        d=len(deno)
        q=float(n)/d
        gini_attribute+=(min(q,(1-q))*(d/len(df)))
        
    return gini_attribute 


# In[8]:


#for numerical data 
def find_node_gini(df,column):
    m=-sys.maxint-1
    value=0
    g=0
    df2=df
    c=''
#     for column in df:
    if column!='left':
        values=df2[column].unique()
        for i in values:
            rows1=df2[df2[column]<=i]
            rows2=df2[df2[column]>i]
            ent1=calculate_gini(rows1)
            ent2=calculate_gini(rows2)
            size1=len(rows1)
            size2=len(rows2)
            sizeall=len(df2)
            ent=(ent1*(size1/sizeall))+(ent2*(size2/sizeall))
            ent_total=calculate_gini(df2)
            gain=ent_total-ent
            if m<gain:
    #             print "here"
                m=gain
                value=i
                c=column
    return m,value,c


# In[9]:


#for numerical data 
def find_node_misclassification(df,column):
    m=-sys.maxint-1
    value=0
    g=0
    df2=df
    c=''
#     for column in df:
    if column!='left':
        values=df2[column].unique()
        for i in values:
            rows1=df2[df2[column]<=i]
            rows2=df2[df2[column]>i]
            ent1=calculate_misclassificationrate(rows1)
            ent2=calculate_misclassificationrate(rows2)
            size1=len(rows1)
            size2=len(rows2)
            sizeall=len(df2)
            ent=(ent1*(size1/sizeall))+(ent2*(size2/sizeall))
            ent_total=calculate_misclassificationrate(df2)
            gain=ent_total-ent
            if m<gain:
    #             print "here"
                m=gain
                value=i
                c=column
    return m,value,c


# In[10]:


def max_gain_gini(df):
    p=""
    m=-sys.maxint-1
    split_pt=None
    for i in df.columns:
        gain=0
        if i in numerical:
            if i!='left':
                gain,value,col=find_node_gini(df,i)
                if m<gain:
                    m=gain
                    p=i
                    split_pt=value
        if i in categorical:
            if i!='left':
                gain=calculate_gini(df)-gini_attribute(df,i,'left')
                if m < gain:
                    m=gain
                    p=i
    return m,p,split_pt


# In[11]:


def max_gain_misclassification(df):
    p=""
    m=-sys.maxint-1
    split_pt=None
    for i in df.columns:
        gain=0
        if i in numerical:
            if i!='left':
                gain,value,col=find_node_misclassification(df,i)
                if m<gain:
                    m=gain
                    p=i
                    split_pt=value
        if i in categorical:
            if i!='left':
                gain=calculate_misclassificationrate(df)-misclassification_attribute(df,i,'left')
                if m < gain:
                    m=gain
                    p=i
    return m,p,split_pt


# In[12]:


print max_gain_gini(train)


# In[13]:


class decisionTree():
    def __init__(self,name,df):
        self.label=name
        self.child={}
        self.positive=len(df[df['left']==1])
        self.negative=len(df[df['left']==0])
        self.isLeaf=False
    def set_child(self,ch):
        self.child=ch


# In[14]:


def buildTreeGini(df):
#     print "calling"
    if len(df.columns)<=1:
        leaf=decisionTree('left',df)
        leaf.isLeaf=True
        return leaf
    gain,label,value=max_gain_gini(df)
    if gain<=0:
        leaf=decisionTree('left',df)
        leaf.isLeaf=True
        return leaf
    if label in numerical:
        root=decisionTree(label,df)
        df2=df
        children={}
        i=value
        rows1=df2[df2[label]<=i]
        rows2=df2[df2[label]>i]
        if rows1.size==df.size or rows2.size==df.size:
            leaf=decisionTree('left',df)
            leaf.isLeaf=True
            return leaf
        ch_root1=buildTree(rows1)
        key1=i
        key2=1
        children[key1]=ch_root1
        ch_root2=buildTree(rows2)
        children[key2]=ch_root2
        root.set_child(children)
    else:
        root=decisionTree(label,df)
        childs=df[label].unique()
        children={}
        df2=df
        for i in childs:
            rows=df2[df2[label]==i]
            rows=rows.drop(columns=[label])
            ch_root=buildTree(rows)
            children[i]=ch_root
        root.set_child(children)
        
    return root


# In[15]:


def buildTree(df):
#     print "calling"
    if len(df.columns)<=1:
        leaf=decisionTree('left',df)
        leaf.isLeaf=True
        return leaf
    gain,label,value=max_gain_misclassification(df)
    if gain<=0:
        leaf=decisionTree('left',df)
        leaf.isLeaf=True
        return leaf
    if label in numerical:
        root=decisionTree(label,df)
        df2=df
        children={}
        i=value
        rows1=df2[df2[label]<=i]
        rows2=df2[df2[label]>i]
        if rows1.size==df.size or rows2.size==df.size:
            leaf=decisionTree('left',df)
            leaf.isLeaf=True
            return leaf
        ch_root1=buildTree(rows1)
        key1=i
        key2=1
        children[key1]=ch_root1
        ch_root2=buildTree(rows2)
        children[key2]=ch_root2
        root.set_child(children)
    else:
        root=decisionTree(label,df)
#     print root.lable, root.positive, root.negative
        childs=df[label].unique()
        children={}
        df2=df
        for i in childs:
            rows=df2[df2[label]==i]
            rows=rows.drop(columns=[label])
            ch_root=buildTree(rows)
            children[i]=ch_root
        root.set_child(children)
        
    return root


# In[16]:


timeSpent_vs_leave={}


# In[17]:


root=buildTreeGini(train)
root1=buildTree(train)


# In[18]:


def predict(model,X):
    root=model
    row=X  
    if root.isLeaf==True:
        if root.positive>root.negative:
            return "YES"
        else:
            return "NO"
    row1=row
    ch_node=None
#     print root.label,row1[root.label]
    if root.label in numerical:
        keys=root.child.keys()
        x=0
        for a in keys:
            if a!=1:
                x=a
        takeside=0
        if row1[root.label]>x:
            takeside=1
        else:
            takeside=x
        ch_node=root.child[takeside]
    else:
        if row1[root.label] in root.child.keys():
            ch_node=root.child[row1[root.label]]
            if ch_node!=None:
                if ch_node.label=='left':
                    if ch_node.positive>ch_node.negative:
                        return "YES"
                    else:
                        return "NO"
            if ch_node==None:
                if root.positive>root.negative:
                    return "YES"
                else:
                    return "NO"
    if ch_node!=None:
        return predict(ch_node,row)


# In[19]:


def calculate_accuracy(df,root):
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
    f1score=(2/((1/recall)+(1/precision)))
    print "True Positive: "+str(tp)
    print "True Negative: "+str(tn)
    print "False Positive: "+str(fp)
    print "False Negative: "+str(fn)
    print "accuracy: "+str(accuracy)+"%"
    print "precision: "+str(precision)+"%"
    print "recall: "+str(recall)+"%"
    print "f1 score: ",f1score
    return 


# In[20]:


calculate_accuracy(validate,root) #gini


# In[21]:


calculate_accuracy(validate,root1) #misclassification


# In[22]:


for index, row in test.iterrows():
    print predict(root,row)

