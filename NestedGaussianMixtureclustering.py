#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
from numpy import *
import operator
import matplotlib.pyplot as plt
import time
from sklearn.datasets import fetch_openml
import pandas as pd
from sklearn.neighbors import kneighbors_graph,NearestNeighbors
from sklearn import datasets
import random as rd 

import seaborn as sns
from sklearn.cluster import KMeans,MiniBatchKMeans
from sklearn.mixture import GaussianMixture as GMM
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.metrics import fowlkes_mallows_score,adjusted_mutual_info_score,adjusted_rand_score


# In[ ]:


def optimalApproxiamtion(data,m_arange=arange(2, 30,1),random_state=0,covariance_type='diag',warm_start=True,plot=False):
        n_components = m_arange
        models = [GMM(n, covariance_type=covariance_type, random_state=random_state,max_iter=50,warm_start=warm_start).fit(data)
                  for n in n_components]
        bics=array([m.bic(data) for m in models])
        best_t= n_components[bics.argmin()]
        print('best t',best_t)
        if plot==True:
            plt.plot(n_components, bics, label='J(M_t)')

            plt.legend(loc='best')
            plt.xlabel('t',fontsize=18)
            #print('best t',n_components[bics.argmin()])
            print('best score',bics[bics.argmin()])
            plt.text(n_components[bics.argmin()],bics.min()-2,'x',fontsize=15)
        optimal_t=n_components[bics.argmin()]
        best_Mt=models[bics.argmin()]
  
        return optimal_t,best_Mt


# In[ ]:


def compressedSimilarityMatrix(X,Mt,plot=False,gmm=None):
    t=Mt.n_components
    labels=Mt.predict(X)
    probabilities=Mt.predict_proba(X)
    sm=zeros((t,t))
    
    for i in range(t):
        sm[i]=probabilities[labels==i].sum(axis=0)
        sm[i,i]=0
    S=sm+sm.T
    
    #S_=MinMaxScaler().fit_transform(S.reshape(-1,1)).reshape(t,t)
    
    if plot==True:
        plt.figure(figsize=(14,14))
        plt.subplot(2,2,1)
        SubCluster_display(X,labels,figsize=None,markSize=[100,600,300],gmm=gmm)
        plt.xticks(())
        plt.yticks(())
        plt.axis('off')
        plt.subplot(2,2,2)
        sns.heatmap(S,robust =False,square=True,linewidths=0.004,vmin=0.0)
    
    return S,labels

def SubCluster_display(data,C,figsize=(10,8),markSize=[100,1000,1200],gmm=None):
    if figsize!=None:
        plt.figure(figsize=figsize)
    clusterNos=pd.unique(C)
    if gmm!=None:
        plot_gmm(gmm, data)
    for i in clusterNos:
        cluster=data[C==i]
        #print('i=',i),print(cluster.shape[0])
        
        
        
        plt.scatter(cluster[:,0],cluster[:,1],s=markSize[0],marker='$'+str(i)+'$')
            
        if i>=10:
            plt.scatter(cluster.mean(axis=0)[0],cluster.mean(axis=0)[1],s=markSize[1],marker='$'+str(i)+'$',c='black')
        else:
            plt.scatter(cluster.mean(axis=0)[0],cluster.mean(axis=0)[1],s=markSize[2],marker='$'+str(i)+'$',c='black')

def plot_gmm(gmm, X, label=True, ax=None):
    ax = ax or plt.gca()
    labels = gmm.fit(X).predict(X)
    if label:
        print(1)
        ax.scatter(X[:, 0], X[:, 1],  s=8,c=labels, zorder=2)#
    else:
        ax.scatter(X[:, 0], X[:, 1], s=4, zorder=2)
    #ax.axis('equal')
    w_factor = 0.4 / gmm.weights_.max()
    for pos, covar, w in zip(gmm.means_, gmm.covariances_  , gmm.weights_):
        draw_ellipse(pos, covar, alpha=w * w_factor)            
        
from matplotlib.patches import Ellipse


def draw_ellipse(position, covariance, ax=None, **kwargs):
    ax = ax or plt.gca()
   
    if covariance.shape == (2, 2):
        U, s, Vt = linalg.svd(covariance)
        angle = degrees(arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * sqrt(s)
    else:
        angle = 0
        #print(covariance.shape)
        if covariance.shape==():
            width=2 * sqrt(covariance)
            height=width
        else:
            width, height = 2 *sqrt(covariance)

 
    for nsig in range(1, 4):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height,
                             angle, **kwargs))


# In[ ]:


def SingleLink_Agg(S):
    
    S=triu(S,1)
    t=S.shape[0]
    h=arange(t)
    #H=tile(h,(t,1))
    H=zeros((t,t))-1
    H[0]=arange(t)
    T=zeros((t-1,7))-1
    N=0
    T_No=t
    #print('H_0',H)
    while N<=t-2:
        #print('round',N)
        #print('H',H)
        s=S.max()
        if s<=0:
            print('break s<=0')
            break
        #print(S.argmax())
        i,j=int(floor(S.argmax()/t)),   S.argmax()%t
        #print(i,j)
        T_No_i=int(H[N,i])
        T_No_j=int(H[N,j])
        
        if T_No_i==T_No_j:
            #print('Ti and Tj belong to the same cluster')
            S[i,j]=-2
        else:
            
            #print('T_No_i,T_No_j')
            #print(T_No_i,T_No_j)
            index_i=where(H[N]==T_No_i)[0]
            index_j=where(H[N]==T_No_j)[0]
            #print('index_i',index_i)
            H[N+1]=H[N]
            H[N+1,index_i]=T_No
            #print('H[N+1,:]',H[N+1,:])
            H[N+1,index_j]=T_No

            if T_No_i<t and T_No_j<t:
                #print('1')

                T[N]=T_No,0,T_No_i,T_No_j,-1,s,-1
            elif T_No_i<t:
                #print('2')

                s=T[T_No_j-t,5]
                T[N]=T_No,1,T_No_i,T_No_j,-1,s,-1
                T[T_No_j-t,4]=T_No
            elif T_No_j<t:
                #print('3')
                #print(T_No_i-t)
                #print(T[T_No_i-t,3])
                s=T[T_No_i-t,5]
                T[N]=T_No,1,T_No_j,T_No_i,-1,s,-1,
                T[T_No_i-t,4]=T_No
            elif T_No_i!=T_No_j:
                #print('4')

                T[N]=T_No,2,T_No_i,T_No_j,-1,s,-1
                #print(T_No_i,t,T_No_i-t)

                T[T_No_i-t,4]=T_No
                T[T_No_i-t,6]=(T[T_No_i-t,5]-s)

                T[T_No_j-t,4]=T_No
                T[T_No_j-t,6]=(T[T_No_j-t,5]-s)
            N+=1
            T_No+=1
            S[i,j]=-2
    #print('H_1',H)
        
    return H,T        
            


# In[ ]:


def NGMM(X,target,N,targetN=False,sim_threshold=0.001,covariance_type='diag',n_init=10,random_state=0,plot=False):
    start=time.time()
    if targetN==False:
        targetN=unique(target).shape[0]
        
    Mt=GMM(N, covariance_type=covariance_type, random_state=random_state,n_init=n_init,max_iter=50,warm_start=True).fit(X) 
    S,labels=compressedSimilarityMatrix(X,Mt,plot=False,gmm=None)
    ci,oi=PruningAccordingtoSim(S,labels,sim_threshold=sim_threshold)
    S_=S[ci][:,ci];labels_=labels[oi]
    X_=X[oi];target_=target[oi]
    #print('remain_components:',len(pd.unique(ci)))
    #print('remain',X_.shape[0])
    
    H,T=SingleLink_Agg(S_)
    ass=clusterExtraction(X_,labels_,H,targetN)
    scores_3=[adjusted_mutual_info_score(target_,ass),adjusted_rand_score(target_,ass),fowlkes_mallows_score(target_,ass),]
    
    RunTime=time.time()-start
    criteria=['AMI','ARI','FMI','RunTime']
    scores_3=[[adjusted_mutual_info_score(target_,ass),adjusted_rand_score(target_,ass),fowlkes_mallows_score(target_,ass),RunTime]]
    re=pd.DataFrame(scores_3,columns=criteria)
    if plot==True:
        print('AMI',adjusted_mutual_info_score(target_,ass))
        print('Ari',adjusted_rand_score(target_,ass))
        print('Fmi',fowlkes_mallows_score(target_,ass))
        plt.scatter(X_[:, 0], X_[:, 1],c=ass)
    return ass,re
    
def PruningAccordingtoSim(sm,labels,sim_threshold=0.1):
    remain_cluster_index=where(sm.max(axis=1)>sim_threshold)
    remain_objects_index=array([i==labels for i in remain_cluster_index[0]]).sum(axis=0)
    return remain_cluster_index[0],remain_objects_index.astype(bool)

def clusterExtraction(D,labels,H,targetN):
    Ass=labels
    AssIndex=unique(labels)
    Ass_r=zeros(Ass.shape[0])-1
    #print('Ass',Ass)
    LevelofTN=H[H.shape[0]-targetN]
    #print('LevelofTN',LevelofTN)
    CluterIndex=unique(LevelofTN)
    #print(CluterIndex)
    for i,c in enumerate(CluterIndex):
        #print('i',i)
        SCNo_c=where( LevelofTN==c)[0]
        #print('SCNo_c',SCNo_c)
        for sc in SCNo_c:
            #print('sc',sc)
            Ass_r[Ass==AssIndex[sc]]=i
            #print('Ass_r',Ass_r)
    return Ass_r
def clustering(D,labels,H,targetN):
    Ass=labels
    Ass_r=zeros(Ass.shape[0])-1
    #print('Ass',Ass)
    LevelofTN=H[H.shape[0]-targetN]
    #print('LevelofTN',LevelofTN)
    CluterIndex=unique(LevelofTN)
    #print(CluterIndex)
    for i,c in enumerate(CluterIndex):
        #print('i',i)
        SCNo_c=where( LevelofTN==c)[0]
        #print('SCNo_c',SCNo_c)
        for sc in SCNo_c:
            #print('sc',sc)
            Ass_r[Ass==sc]=i
            #print('Ass_r',Ass_r)
    return Ass_r
            

def RunTimeTest_NGMM_A(X,target,m_arange=arange(2,100,10),targetN=None,random_state=0,covariance_type='diag',warm_start=True):
    start_search=time.time()
    t,Mt=optimalApproxiamtion(X,m_arange=m_arange,random_state=random_state,covariance_type=covariance_type,warm_start=warm_start)
    end_search=time.time()
    time_1=end_search-start_search
    S,labels=compressedSimilarityMatrix(X,Mt)
    H,T=SingleLink_Agg(S)
    if targetN==None:
        targetN=unique(target).shape[0]
        
    Ass=clustering(X,labels,H,targetN=targetN)
    end_alg=time.time()
    time_2=end_alg-end_search
    time_all=end_alg-start_search
    return time_1,time_2,time_all

# In[ ]:




