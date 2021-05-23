#!/usr/bin/env python
# coding: utf-8

# In[1]:


# find mean of every column
# matrix-mean
# q=xt.x
# q==>covariance
# eigen values and vectors


# # q-1

# # q-1-1

# In[2]:


import numpy as np
import pandas as pd
import sys

data=pd.read_csv("../input_data/intrusion_detection /data.csv")

target=data['xAttack']
data=data.drop(['xAttack'],axis=1)

columns=['duration','service','src_bytes','dst_bytes','hot','num_failed_logins','num_compromised','num_root','num_file_creations','num_access_files','count','srv_count','serror_rate','srv_serror_rate','rerror_rate','srv_rerror_rate','same_srv_rate','diff_srv_rate','srv_diff_host_rate','dst_host_count','dst_host_srv_count','dst_host_same_srv_rate','dst_host_diff_srv_rate','dst_host_same_src_port_rate','dst_host_srv_diff_host_rate','dst_host_serror_rate','dst_host_srv_serror_rate','dst_host_rerror_rate','dst_host_srv_rerror_rate']

x=data

#covariance matrix 
def covariance(data):
    for i in columns:
        data[i]=(data[i]-data[i].mean())/data[i].std()   
    x=data
    xt=x.transpose()
    cov=np.dot(xt,x)
    return cov

            
cov_matrix=covariance(data)
rows,col=np.shape(x)
# print rows-1

cov_matrix=cov_matrix/(rows-1)

    
# print cov_matrix
eigen_values,eigen_vector=np.linalg.eig(cov_matrix)

#variance=sum of eigen values
#contribution=eigen /variance * 100
#sort eigen values


# In[3]:


variance=[]
count=0
sum_value=np.sum(eigen_values)

#contribution
eigen_values=eigen_values/(sum_value)
eigen_values=eigen_values*100



for i in eigen_values:
    variance.append((i,(columns[count],eigen_vector[count])))
    count+=1
    

variance.sort(reverse=True)

temp=[]

total=0
for i in range(0,len(variance)):
    total+=variance[i][0]
    temp.append(total)


# print temp

def number_of_features(temp):
    count=1
    for i in temp:
        if i>=90: #given in question
            break
        count+=1
    return count


features=number_of_features(temp)

x_axis=[]
for i in range(1,features+1):
    x_axis.append(i)
print temp
x_axis = range(1,30)
print x_axis


# In[4]:


import matplotlib.pyplot as plt
fig,axes=plt.subplots(figsize=(7,7))
axes.plot(x_axis,temp)
axes.grid(True)
axes.set_title('Cumulative variance v/s Iterations')
axes.legend(loc="best")
plt.xlabel('Iterations')
plt.ylabel('Cumulative Variance')


# In[5]:


feature_vector=[]
i=0
temp_col=[]
while features!=0:
    feature_vector.append(variance[i][1][1])
    temp_col.append(variance[i][1][0])
    i+=1
    features-=1
    

feature_vector_df=pd.DataFrame(feature_vector)
xtranspose=x.transpose()


final_data=np.dot(feature_vector_df, xtranspose)


print "*************part 1 answer*************"
print final_data

#for part 2
final_data=final_data.transpose()


# # q-1-2
# #k-means

# In[6]:


k=5 #number of clusters
n=final_data.shape[0] #number of training data
c=final_data.shape[1] #number of features in the data


#generate random centers
mean = np.mean(final_data, axis=0)
std = np.std(final_data, axis=0)


centers = np.random.randn(k,c)*std + mean


# In[7]:


from copy import deepcopy
import math

def create_clusters(centers,data):
    clusts=[]
    for k in range(0,len(data)):
        min_dist=sys.maxint
        cluster=0
        for i in range(0,len(centers)):
            dist=0
            for j in range(0,14):
                dist+=((centers[i][j]-data[k][j])**2)
            dist=math.sqrt(dist)
            if min_dist > dist:
                min_dist=dist
                cluster=i
        clusts.append(cluster)
    return clusts

clusters=create_clusters(centers,final_data)
# print (clusters)


# In[8]:


data=pd.DataFrame(final_data,columns=temp_col)
data['xAttack']=target
data['clusters']=clusters

centers_old=np.zeros(centers.shape)
centers_new=deepcopy(centers)


def find_new_centers(centers_new,data):
    data=data.drop(columns='xAttack')
    for i in range(0,5):
        arr=np.mean(data[data['clusters']== i],axis=0)
        arr=arr.drop(arr.index[-1])
        centers_new[i]=arr
    return centers_new
            
error = np.linalg.norm(centers_new - centers_old)
# times=0
print error
while error!=0.0:
#     times+=1
    centers_old=deepcopy(centers_new)
    clusters=create_clusters(centers_new,data.values)
    data['clusters']=clusters
    centers_new=find_new_centers(centers_new,data)
    error = np.linalg.norm(centers_new - centers_old)

# print times


# In[9]:


purity_kmeans=[]
for i in range(5):
    normal=0
    dos=0
    probe=0
    r2l=0
    u2l=0
    total=0
    for index,row in data.iterrows():
        if row['clusters']==i:
            
            if row['xAttack']=='normal':
                normal+=1
            
            if row['xAttack']=='dos':
                dos+=1
            
            if row['xAttack']=='probe':
                probe+=1
            
            if row['xAttack']=='r2l':
                r2l+=1
            
            if row['xAttack']=='u2r':
                u2l+=1
            temp=[]
            
            total=normal+dos+probe+r2l+u2l
    temp.append((float(normal)/total)*100)
    temp.append((float(dos)/total)*100)
    temp.append((float(probe)/total)*100)
    temp.append((float(r2l)/total)*100)
    temp.append((float(u2l)/total)*100)
    purity_kmeans.append(temp)
    print "for cluster ",i
    print "normal purity: ",(float(normal)/total)*100
    print "dos purity: ",(float(dos)/total)*100
    print "probe purity: ",(float(probe)/total)*100
    print "r2l purity: ",(float(r2l)/total)*100
    print "u2r purity: ",(float(u2l)/total)*100
    print 


# In[10]:


def plot_pie_graph(labels,values,cluster):
    colors = ['lightblue', 'green', 'yellow', 'purple', 'red']
    y=np.array(values)
    legends = ['{0} - {1:1.2f} %'.format(i,j) for i,j in zip(labels, y)]
    fig, ax = plt.subplots(figsize=(5, 5))
    wedges, labels = ax.pie(y, colors=colors, shadow=True, startangle=90)
    ax.set_title("cluster: "+str(cluster))
    wedges, legends, dummy =  zip(*sorted(zip(wedges, legends, y),
                                          key=lambda x: x[2],
                                          reverse=True))

    ax.legend(wedges, legends, loc='best', bbox_to_anchor=(-0.1, 1.),
           fontsize=12)


labels=['normal','dos','probe','r2l','u2r']
clust=[0,1,2,3,4]


# In[11]:


print "\t\t\t\t**********k_means**********\t"
for i in range(5):
    plot_pie_graph(labels,purity_kmeans[i],clust[i])


# # q-1-3
# # GMM

# In[12]:


from sklearn.mixture import GaussianMixture as GMM

def calculateGMM(data):
    gmm = GMM(n_components=5).fit(data)
    labels = gmm.predict(data)
    return labels


data=data.drop(['xAttack'],axis=1)

clusters_for_gmm=calculateGMM(data)

data['clusters']=clusters_for_gmm
data['xAttack']=target


purity_gmm=[]
for i in range(5):
    temp=[]
    normal=0
    dos=0
    probe=0
    r2l=0
    u2l=0
    total=0
    for index,row in data.iterrows():
        if row['clusters']==i:
            
            if row['xAttack']=='normal':
                normal+=1
            
            if row['xAttack']=='dos':
                dos+=1
            
            if row['xAttack']=='probe':
                probe+=1
            
            if row['xAttack']=='r2l':
                r2l+=1
            
            if row['xAttack']=='u2r':
                u2l+=1
            temp=[]
            
            total=normal+dos+probe+r2l+u2l
    temp.append((float(normal)/total)*100)
    temp.append((float(dos)/total)*100)
    temp.append((float(probe)/total)*100)
    temp.append((float(r2l)/total)*100)
    temp.append((float(u2l)/total)*100)
    purity_gmm.append(temp)
    print "for cluster ",i
    print "normal purity: ",(float(normal)/total)*100
    print "dos purity: ",(float(dos)/total)*100
    print "probe purity: ",(float(probe)/total)*100
    print "r2l purity: ",(float(r2l)/total)*100
    print "u2r purity: ",(float(u2l)/total)*100
    print 


# In[13]:


print "\t\t\t\t**********GMM**********\t"
for i in range(5):
    plot_pie_graph(labels,purity_gmm[i],clust[i])


# # q-1-4
# 
# # Hierarchical clustering 

# In[14]:


from sklearn.cluster import AgglomerativeClustering
# data=data.drop(['xAttack'],axis=1)
cluster = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='single') 
temp=data.values
test = temp[:2000,0:14]
clusters_hierarchical=cluster.fit_predict(test)  
# print clusters_hierarchical
# plt.figure(figsize=(10, 7))  
# plt.scatter(test[:,0], test[:,1], c=cluster.labels_)


# In[15]:


t=data.values
t=t[:2000,0:14]
data=pd.DataFrame(t)
data['clusters']=clusters_hierarchical
data['xAttack']=target

purity=[]
for i in range(5):
    temp=[]
    normal=0
    dos=0
    probe=0
    r2l=0
    u2l=0
    total=0
    for index,row in data.iterrows():
        if row['clusters']==i:
            
            if row['xAttack']=='normal':
                normal+=1
            
            if row['xAttack']=='dos':
                dos+=1
            
            if row['xAttack']=='probe':
                probe+=1
            
            if row['xAttack']=='r2l':
                r2l+=1
            
            if row['xAttack']=='u2r':
                u2l+=1
            temp=[]
            
            total=normal+dos+probe+r2l+u2l
    temp.append((float(normal)/total)*100)
    temp.append((float(dos)/total)*100)
    temp.append((float(probe)/total)*100)
    temp.append((float(r2l)/total)*100)
    temp.append((float(u2l)/total)*100)
    purity.append(temp)
    print "for cluster ",i
    print "normal purity: ",(float(normal)/total)*100
    print "dos purity: ",(float(dos)/total)*100
    print "probe purity: ",(float(probe)/total)*100
    print "r2l purity: ",(float(r2l)/total)*100
    print "u2r purity: ",(float(u2l)/total)*100
    print 


# In[16]:


print "\t\t\t  **********Hierarchical Custering**********\t"
for i in range(5):
    plot_pie_graph(labels,purity[i],clust[i])


# # q-1-5
# 

# PCA works by minimizing variance, which is not applicable for binary variables.
# Although converting categorical feature to nominal values might help us in using PCA on categorical features.
# But using categorical features in that way does not give much meaningful result.
# 
# Also finding suitable way to represent distances between categorical attributes does not make sense.
# 
# So, we can use PCA for such type of data set but it would not be a very good approach.
# 
# Instead we can go for multiple correspondence analysis, if categorical features are present in dataset.
