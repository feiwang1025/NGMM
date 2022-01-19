#!/usr/bin/env python
# coding: utf-8

# In[1]:


from numpy import *
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import pdist,squareform
from sklearn.neighbors import LocalOutlierFactor,kneighbors_graph
import pandas as pd
import seaborn as sns
from sklearn import datasets
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import fowlkes_mallows_score


# In[14]:


from sklearn import kernel_approximation
from sklearn.neighbors import KernelDensity


# In[15]:


def densityCaculating(data,radius):
    m=data.shape[0]
    distMat=pairwise_distances(data,n_jobs=-1)
    comprehensive_mat=zeros((m,4))-1
    for i in range(m):
        valid_objects=sum(distMat[i][1:]<radius)
        comprehensive_mat[i,1]=valid_objects
    for i in range(m):
        den_array=comprehensive_mat[:,1]
        denser_indices=where(den_array>den_array[i])[0]
        if sum(denser_indices):
            denser_closest_pointNo=denser_indices[argmin(distMat[i][denser_indices])]
            distance=distMat[i][denser_closest_pointNo]
            comprehensive_mat[i,2]=denser_closest_pointNo
            comprehensive_mat[i,3]=distance
        else:
            comprehensive_mat[i,3]=distMat.max()
    return comprehensive_mat    


# In[16]:


def kernalDensityCaculating(data,bandwidth):
    m=data.shape[0]
    distMat=pairwise_distances(data,n_jobs=-1)
    comprehensive_mat=zeros((m,4))-1
    
    kDEst=KernelDensity(bandwidth)
    kDEst.fit(data)
    kdensity=kDEst.score_samples(data)
    comprehensive_mat[:,1]=kdensity
    for i in range(m):
        den_array=comprehensive_mat[:,1]
        denser_indices=where(den_array>den_array[i])[0]
        if sum(denser_indices):
            denser_closest_pointNo=denser_indices[argmin(distMat[i][denser_indices])]
            distance=distMat[i][denser_closest_pointNo]
            comprehensive_mat[i,2]=denser_closest_pointNo
            comprehensive_mat[i,3]=distance
        else:
            comprehensive_mat[i,3]=distMat.max()
    return comprehensive_mat    


# In[17]:


def dpcluster(data,bandwidth=1,clusterNumber=None,γ=1,plot=False):
    comprehensive_mat=kernalDensityCaculating(data,bandwidth)
    norm_density=MinMaxScaler().fit_transform(comprehensive_mat)[:,1]
    final_decision=norm_density*comprehensive_mat[:,3]
    if clusterNumber==None:
        peakCandidates=argsort(-final_decision)
        peaks=where(final_decision>γ)[0]
    else:
        peaks=argsort(final_decision)[-clusterNumber:]
        
    norm_density[norm_density==0]=sorted(norm_density[norm_density!=0])[0]
    #print('i',sorted(norm_density[norm_density!=0])[0])
    min_x=min(norm_density)
    #min_x=0.05
    max_x=max(norm_density)
    x=linspace(min_x,max_x,100)
    y=γ/x
    if plot==True:
        plt.figure(figsize=(17,5))
        plt.subplot(1,3,1)
        plt.scatter(norm_density,comprehensive_mat[:,3],s=5)
        #plt.scatter(norm_density[norm_density>=0.05],comprehensive_mat[:,3][norm_density>=0.05],s=5)
        plt.plot(x,y,c='g')
    
    
    
    for i,p in enumerate(peaks):
        comprehensive_mat[p,0]=i
    
    if plot==True:
        plt.scatter(norm_density[peaks],comprehensive_mat[:,3][peaks],c=comprehensive_mat[peaks,0])
        print('peaks',peaks)
        plt.subplot(1,3,2)
        plt.scatter(data[:,0],data[:,1],s=10)
        plt.scatter(data[:,0][peaks],data[:,1][peaks],c=comprehensive_mat[peaks,0],marker='*')

    
    comprehensive_mat=clusteringAccordingPeaks(comprehensive_mat)
    if plot==True:
        plt.subplot(1,3,3)
        plt.scatter(data[:,0],data[:,1],s=10,c=comprehensive_mat[:,0])
    #print('ari',adjusted_rand_score(,comprehensive_mat[:,0]))
    return comprehensive_mat

def clusteringAccordingPeaks(comprehensive_mat):
    m=comprehensive_mat.shape[0]
    for i in range(m):
        if comprehensive_mat[i,0]==-1:
            pointedObject=int(comprehensive_mat[i,2])
            comprehensive_mat[i,0]=-2
            #print(pointedObject)
            while comprehensive_mat[pointedObject,0]==-1:
                
                comprehensive_mat[pointedObject,0]=-2
                pointedObject=int(comprehensive_mat[pointedObject,2])
            comprehensive_mat[comprehensive_mat[:,0]==-2]= comprehensive_mat[pointedObject,0]   
    return comprehensive_mat
            


            


# In[ ]:








