#!/usr/bin/env python
# coding: utf-8

# In[388]:


import numpy as np
import pandas as pd
import matplotlib.image as mpimg 
import matplotlib.pyplot as plt 
import cv2


# In[389]:


img = mpimg.imread('./input_data/pandas.jpeg',0) 
img = cv2.resize(img, (32, 32))
print img.shape
plt.imshow(img)


# In[390]:


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])


# In[391]:


img=rgb2gray(img)
print img.shape
fig, ax = plt.subplots(figsize = (4,6))
plt.imshow(img)
plt.show()


# In[392]:


def kernel(size):
    filtr=np.random.randn(size,size)
    return filtr


# In[393]:


def sigmoid(value): 
    a = np.exp(-value)
    return 1.0/ (1.0 + a)

def tanh(x):
    return np.tanh(x)
def Relu(x):
#     return x * (x > 0)
    return np.maximum(0,x)


# In[394]:


def create_convolved_feature(img,filtr, function):
    img_rows,img_cols=img.shape
    fil_rows,fil_cols=filtr.shape
    bias = np.random.randn(img_rows-fil_rows+1,img_cols-fil_cols+1)
    result=np.zeros((img_rows-fil_rows+1,img_cols-fil_cols+1))
    for i in range(0,img_rows-fil_rows+1):
        for j in range(0,img_cols-fil_cols+1):
            submat=img[i:i+fil_rows,j:j+fil_cols]
            result[i][j]=np.sum(np.multiply(submat,filtr)) + bias[i][j]
            
#     result = np.sum(result,bias)
    result = applyNonLinearity(result, function)
    result = np.array(result)
    return result


# In[395]:


def applyNonLinearity(convolution_list, function):
    new_conv_list = []
    for one_conv in convolution_list:
        one_conv = function(one_conv)
        new_conv_list.append(one_conv)
    new_conv_list=np.array(new_conv_list)
    return new_conv_list


# In[396]:


def create_cube(img,filtr,depth_of_cube, function):
    cube=[]
    for i in range(depth_of_cube):
        filtr=kernel(5)
        result=create_convolved_feature(img,filtr, function)
        cube.append(result)
    cube=np.array(cube)
    return cube


# In[397]:


def displayCube(cube):
    depth = cube.shape[0]
    rows = 2
    cols = depth // rows
    f, axarr = plt.subplots(rows, cols, figsize=(8,8))
    r, c = 0, 0
    for d in cube:
        axarr[r,c].imshow(d)
        c += 1
        if c == cols:
            r += 1
            c = 0


# In[398]:


def max_pooling_single(matrix):
    matrix_rows,matrix_cols = matrix.shape
    half = matrix_rows/2
    max_pool_matrix = np.zeros((half,half))
    r, c = 0,0
    for i in range(0,matrix_rows,2): #0,1,2
        for j in range(0,matrix_rows,2): #0,1,2
            submat = matrix[i: i + 2, j : j + 2] #0:2, 0:2
            maxi = np.max(submat)
            max_pool_matrix[r][c] = maxi
            c += 1
            if c == half:
                c = 0
                r += 1
    return max_pool_matrix


# In[399]:


def max_pooling(cube):
    length=len(cube)
    result=[]
    for i in range(length):
        temp=max_pooling_single(cube[i])
        result.append(temp)
    result=np.array(result)
    return result


# In[400]:


def create_filtr_cube(size,times):
    filtr_cube1=[]
    for i in range(times):
        filtr_temp=kernel(size)
        filtr_cube1.append(filtr_temp)
    filtr_cube1=np.array(filtr_cube1)
    return filtr_cube1


# In[401]:


def create_cube_from_cubic_filter(img_cube,filtr_cube,depth_of_new_cube,function):
    img_depth,img_rows,img_cols=img_cube.shape
    fil_depth,fil_rows,fil_cols=filtr_cube.shape
    result_cube=[]
    for d in range(depth_of_new_cube):
        filtr_cube=create_filtr_cube(fil_rows,fil_depth)
        convs=[]
        for i in range(fil_depth):
            convs.append(create_convolved_feature(img_cube[i],filtr_cube[i], function))
        convs=np.array(convs)
        d_x,r_x,c_x=np.shape(convs)
        temp_convolv=np.zeros((r_x,c_x))
        bias = np.random.randn(r_x,c_x)
        for a in range(r_x):
            for b in range(c_x):
                val=0
                for d in range(d_x):
                    val+=convs[d][a][b]
                temp_convolv[a][b]=val+bias[a][b] 
#         temp_convolv=np.sum(temp_convolv,bias)        
        temp_convolv=applyNonLinearity(temp_convolv, function)
        result_cube.append(temp_convolv)
    result_cube=np.array(result_cube)
    return result_cube


# In[402]:


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


# # Relu

# In[403]:


filtr=kernel(5)


# In[404]:


cube1=create_cube(img,filtr,6, Relu)

# cube1=applyNonLinearity(cube1, Relu)


# In[405]:


print "filter dimentions: ",filtr.shape
print "cube at level 1: ",cube1.shape
displayCube(cube1)


# In[406]:


cube1=max_pooling(cube1)
print "cube dimension after max-pooling: ",cube1.shape


# In[407]:


print "cube after max-pooling"
displayCube(cube1)


# In[408]:


filtr_cube1=create_filtr_cube(5,6)
cube2=create_cube_from_cubic_filter(cube1,filtr_cube1,16,Relu)


# In[409]:


print "Filter cube dimension: ",filtr_cube1.shape
print "Cube at level 2: ", cube2.shape
displayCube(cube2)


# In[410]:


cube2=max_pooling(cube2)
print "cube dimension after max-pooling: ",cube2.shape


# In[411]:


print "cube after max-pooling"
displayCube(cube2)


# In[412]:


#flattening
full_connection_layer1=cube2.flatten()
full_connection_layer1=np.array(full_connection_layer1)


# In[413]:


rand_wt1=np.random.randn(120,400)
fc1_bias_r=np.random.randn(120)

full_connection_layer2=np.dot(rand_wt1,full_connection_layer1)+fc1_bias_r
print full_connection_layer2.shape
full_connection_layer2=applyNonLinearity(full_connection_layer2, Relu)
print "Dimensions of FC-layer1: ",full_connection_layer2.shape
# print full_connection_layer2_t


# In[414]:


rand_wt2=np.random.randn(84,120)
fc2_bias_r=np.random.randn(84)

full_connection_layer3=np.dot(rand_wt2,full_connection_layer2)+fc2_bias_r
full_connection_layer3=applyNonLinearity(full_connection_layer3,Relu)
print "Dimensions of FC-layer2: ",full_connection_layer3.shape
# print full_connection_layer3_t


# In[415]:


#gaussian    
# print full_connection_layer3.shape[0]
rand_wt3=np.random.randn(10,84)
full_connection_layer4=np.zeros((10,1))
for i in range(10):
    full_connection_layer4[i,0] = np.sum((full_connection_layer3-rand_wt3[i])**2)
# full_connection_layer4_t=applyNonLinearity(full_connection_layer4_t, tanh)
print "Dimensions of final output layer: ",full_connection_layer4.shape
# print full_connection_layer4


# In[416]:


probability=softmax(full_connection_layer4)
print "Final output"
print probability


# # tanh

# In[417]:


filtr_t=kernel(5)


# In[418]:


cube1_t=create_cube(img,filtr_t,6, tanh)
# cube1_t=applyNonLinearity(cube1_t, tanh)


# In[419]:


print "filter dimentions: ",filtr_t.shape
print "cube at level 1: ",cube1_t.shape
displayCube(cube1_t)


# In[420]:


cube1_t=max_pooling(cube1_t)
print "cube dimension after max-pooling: ",cube1_t.shape


# In[421]:


print "cube after max-pooling"
displayCube(cube1_t)


# In[422]:


filtr_cube1_t=create_filtr_cube(5,6)
cube2_t=create_cube_from_cubic_filter(cube1_t,filtr_cube1_t,16,tanh)


# In[423]:


print "Filter cube dimension: ",filtr_cube1_t.shape
print "Cube at level 2: ", cube2_t.shape
displayCube(cube2_t)


# In[424]:


cube2_t=max_pooling(cube2_t)
print "cube dimension after max-pooling: ",cube2_t.shape


# In[425]:


print "cube after max-pooling"
displayCube(cube2_t)


# In[426]:


#flattening
full_connection_layer1_t=cube2_t.flatten()
full_connection_layer1_t=np.array(full_connection_layer1_t)


# In[427]:


print full_connection_layer1_t.shape[0]
rand_wt1_t=np.random.randn(120,400)
fc1_bias=np.random.randn(120)
full_connection_layer2_t=np.dot(rand_wt1_t,full_connection_layer1_t)+fc1_bias
full_connection_layer2_t=applyNonLinearity(full_connection_layer2_t, tanh)
print "Dimensions of FC-layer1: ",full_connection_layer2_t.shape
# print full_connection_layer2_t


# In[428]:


rand_wt2_t=np.random.randn(84,120)
fc2_bias=np.random.randn(84)
full_connection_layer3_t=np.dot(rand_wt2_t,full_connection_layer2_t)+fc2_bias
full_connection_layer3_t=applyNonLinearity(full_connection_layer3_t,tanh)
print "Dimensions of FC-layer2: ",full_connection_layer3_t.shape
# print full_connection_layer3_t


# In[429]:


#gaussian    
rand_wt3_t=np.random.randn(10,84)
full_connection_layer4_t=np.zeros((10,1))
for i in range(10):
    full_connection_layer4_t[i,0] = np.sum((full_connection_layer3_t-rand_wt3_t[i])**2)
# full_connection_layer4_t=applyNonLinearity(full_connection_layer4_t, tanh)
print "Dimensions of final output layer: ",full_connection_layer4_t.shape
# print full_connection_layer4_t


# In[430]:


probability=softmax(full_connection_layer4_t)
print "Final output"
print probability


# # Sigmoid

# In[431]:


filtr_s=kernel(5)


# In[432]:


cube1_s=create_cube(img,filtr_s,6, sigmoid)
# cube1_s=applyNonLinearity(cube1_s, sigmoid)
# print cube1_s


# In[433]:


print "filter dimentions: ",filtr_s.shape
print "cube at level 1: ",cube1_s.shape
displayCube(cube1_s)


# In[434]:


cube1_s=max_pooling(cube1_s)
print "cube dimension after max-pooling: ",cube1_s.shape


# In[435]:


print "cube after max-pooling"
displayCube(cube1_s)


# In[436]:


filtr_cube1_s=create_filtr_cube(5,6)
cube2_s=create_cube_from_cubic_filter(cube1_s,filtr_cube1_s,16,sigmoid)


# In[437]:


print "Filter cube dimension: ",filtr_cube1_s.shape
print "Cube at level 2: ", cube2_s.shape
displayCube(cube2_s)


# In[438]:


cube2_s=max_pooling(cube2_s)
print "cube dimension after max-pooling: ",cube2_s.shape


# In[439]:


print "cube after max-pooling"
displayCube(cube2_s)


# In[440]:


#flattening
full_connection_layer1_s=cube2_s.flatten()
full_connection_layer1_s=np.array(full_connection_layer1_s)


# In[441]:


print full_connection_layer1_s.shape[0]
rand_wt1_s=np.random.randn(120,400)
fc1_bias_s=np.random.randn(120)
full_connection_layer2_s=np.dot(rand_wt1_s,full_connection_layer1_s)+fc1_bias_s
full_connection_layer2_s=applyNonLinearity(full_connection_layer2_s, sigmoid)
print "Dimensions of FC-layer1: ",full_connection_layer2_s.shape
# print full_connection_layer2_t


# In[442]:


rand_wt2_s=np.random.randn(84,120)
fc2_bias_s=np.random.randn(84)
full_connection_layer3_s=np.dot(rand_wt2_s,full_connection_layer2_s)+fc2_bias_s
full_connection_layer3_s=applyNonLinearity(full_connection_layer3_s,sigmoid)
print "Dimensions of FC-layer2: ",full_connection_layer3_s.shape
# print full_connection_layer3_t


# In[443]:


#gaussian    
rand_wt3_s=np.random.randn(10,84)
full_connection_layer4_s=np.zeros((10,1))
for i in range(10):
    full_connection_layer4_s[i,0] = np.sum((full_connection_layer3_s-rand_wt3_s[i])**2)
# full_connection_layer4_t=applyNonLinearity(full_connection_layer4_t, tanh)
print "Dimensions of final output layer: ",full_connection_layer4_s.shape
# print full_connection_layer4_s


# In[444]:


probability=softmax(full_connection_layer4_s)
print "Final output"
print probability


# In[ ]:




