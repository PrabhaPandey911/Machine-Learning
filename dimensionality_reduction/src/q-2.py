#!/usr/bin/env python
# coding: utf-8

# # Question 2: Problem of Generating New Data

# In[121]:


from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# handwritten digits dataset available in sklearn.

# In[122]:


digits = datasets.load_digits()

n_samples = len(digits.images)
data = digits.data
print np.shape(data)
column=set(digits.target)
print column
print data


# # Apply dimensionality reduction using PCA , to reduce the number of features to 3 values in the range 15 to 41.

# In[123]:


print "original data: ", np.shape(data)
from sklearn import decomposition
def pca_decom(data,features):
    pca = decomposition.PCA(n_components=features)
    pca.fit(data)
    data = pca.transform(data)
    return data,pca


d15,pca15=pca_decom(data,15)
print "pca reduction to 15 features",np.shape(d15)
d30,pca30=pca_decom(data,30)
print "pca reduction to 30 features",np.shape(d30)
d41,pca41=pca_decom(data,41)
print "pca reduction to 41 features",np.shape(d41)


# # Part-1: (10 points): Kernel Density Estimation: 
# 
# Use grid search cross validation on the reduced feature data to optimize bandwidth
# Compute Kernel Density Estimate

# In[124]:


from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV

def kde(data):
    # use grid search cross-validation to optimize the bandwidth
    parameters = {'bandwidth': np.logspace(-1, 1, 20)}
    grid = GridSearchCV(KernelDensity(), parameters, cv=5)
    grid.fit(data)

    print("best bandwidth: {0}".format(grid.best_estimator_.bandwidth))

    # use the best estimator to compute the kernel density estimate
    kde = grid.best_estimator_
    return kde


# In[125]:


kde_15=kde(d15)
print kde_15


# In[126]:


kde_30=kde(d30)
print kde_30


# In[127]:


kde_41=kde(d41)
print kde_41


# # Part-2: (10 points) Gaussian Mixture Model based Density Estimation:
# 
# Use Bayesian Information Criteria to find the number of GMM components we
# should use Apply GMM using the above number of components

# In[128]:


from sklearn.mixture import GaussianMixture as GMM
def gmm_bic(data):
    n_components = np.arange(50, 600,10)
    models = [GMM(n, covariance_type='full', random_state=0) for n in n_components]
    bics = [model.fit(data).bic(data) for model in models]
    plt.plot(n_components, bics)


# In[129]:


gmm_bic(d15)


# In[130]:


gmm = GMM(190, covariance_type='full', random_state=0)
gmm_15=gmm.fit(d15)
# print(gmm.converged_)


# In[131]:


gmm_bic(d30)


# In[132]:


gmm = GMM(110, covariance_type='full', random_state=0)
gmm_30=gmm.fit(d30)


# In[133]:


gmm_bic(d41)


# In[134]:


gmm = GMM(90, covariance_type='full', random_state=0)
gmm_41=gmm.fit(d41)


# # Part-3: (10 points)
# Draw 48 new points in the projected spaces using both the above generative models. Use Inverse transform of PCA to construct new digits. Plot these points from both the models.

# In[ ]:


def plot_digits(data):
    fig, ax = plt.subplots(8, 6, figsize=(8, 8), subplot_kw=dict(xticks=[], yticks=[]))
    for i, axi in enumerate(ax.flat):
        im = axi.imshow(data[i].reshape(8, 8), cmap = 'Greys')
        im.set_clim(0, 16)


# GMM for data with 15 features
# 

# In[139]:



newData_gmm_15,Y = gmm_15.sample(48)
print "shape: ",np.shape(newData_gmm_15)
digits_new_gmm_15= pca15.inverse_transform(newData_gmm_15)
plot_digits(digits_new_gmm_15)


# KDE for data with 15 features

# In[141]:


newData_kde_15 = kde_15.sample(48)
digits_new_kde_15 = pca15.inverse_transform(newData_kde_15)
plot_digits(digits_new_kde_15)


# GMM for data with 30 features

# In[142]:


newData_gmm_30, y1= gmm_30.sample(48)
digits_new_gmm_30 = pca30.inverse_transform(newData_gmm_30 )
plot_digits(digits_new_gmm_30)



# KDE for data with 30 features

# In[145]:


newData_kde_30 = kde_30.sample(48)
digits_new_kde_30 = pca30.inverse_transform(newData_kde_30)
plot_digits(digits_new_kde_30)


# GMM for data with 41 features

# In[146]:



newData_gmm_41,y2 = gmm_41.sample(48)
digits_new_gmm_41 = pca41.inverse_transform(newData_gmm_41)
plot_digits(digits_new_gmm_41)


# KDE for data with 41 features

# In[147]:


newData_kde_41 = kde_41.sample(48)
digits_new_kde_41 = pca41.inverse_transform(newData_kde_41)
plot_digits(digits_new_kde_41)

