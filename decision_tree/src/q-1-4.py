#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import division
import numpy as np
import pandas as pd
import sys
import matplotlib
import matplotlib.pyplot as plt
from pylab import *
# filename = raw_input("enter file name: ")


# In[2]:


data=pd.read_csv('../input_data/train.csv')
# print data
# test=pd.read_csv('../input_data/sample_test.csv')
train=data.sample(frac=0.8,random_state=200)
validate=data.drop(train.index)


# In[3]:


xlabel='satisfaction_level'
ylabel='last_evaluation'

sl0 = train[xlabel][train['left'] == 0]
sl1 = train[xlabel][train['left'] == 1]
ll0 = train[ylabel][train['left'] == 0]
ll1 = train[ylabel][train['left'] == 1]
axes = plt.subplots(figsize=(10,10))[1]
#label is used for printing left=0 and left=1 in bottom left corner
axes.scatter(sl0, ll0,  label=r"$left = 0$")
axes.scatter(sl1, ll1,  label=r"$left = 1$")
legend = axes.legend(loc='best')

axes.set_title('Comparison')
plt.xlabel(xlabel)
plt.ylabel(ylabel)


# In[ ]:




