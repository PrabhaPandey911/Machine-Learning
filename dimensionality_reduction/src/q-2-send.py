#!/usr/bin/env python
# coding: utf-8

# In[1]:


input_sequence="CTTCATGTGAAAGCAGACGTAAGTCA"
state_path="EEEEEEEEEEEEEEEEEE5IIIIIII$"


# In[2]:


transition={"EE":0.9,"E5":0.1,"5I":1.0,"II":0.9,"I$":0.1}
emission={"EA":0.25,"EC":0.25,"EG":0.25,"ET":0.25,"5A":0.05,"5C":0,"5G":0.95,"5T":0,"IA":0.4,"IC":0.1,"IG":0.1,"IT":0.4}


# In[3]:


print len(input_sequence)
print len(state_path)


# In[4]:


import math
x=0
for i in range(len(state_path)-1):
    e=state_path[i]+input_sequence[i]
    t=state_path[i]+state_path[i+1]
    x+=math.log(emission[e])
    x+=math.log(transition[t])
print x

