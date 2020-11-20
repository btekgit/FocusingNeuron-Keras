#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  1 17:04:12 2020

@author: btek
"""

import os
os.environ['CUDA_VISIBLE_DEVICES']="0"
os.environ['TF_FORCE_GPU_ALLOW_GROWTH']="true"
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math

ISize = 200
Wsize = 2
neurons = 16
wdist = 'uniform'

#W = tf.random.uniform((Wsize,1),-1.0,1.0,name='W')
idx = tf.linspace(0.0,1.0,ISize, name='idx')
mu = tf.linspace(0.1,0.9,neurons, name='mu')

sis = np.linspace(0.001,5,50)
plt.figure(231)
plt.figure(232)
plt.figure(233)
varlistUW=[]
varlistW=[]
for s in sis:
    
    
    si = tf.random.uniform(mu.shape,s*0.99,s*1.11,dtype='float32', name='si')
    #sigma = tf.clip_by_value(si,1e-3,5.0)
    #si =  tf.Variable(np.array([math.pow(2,-x+1) for x in range(neurons)],np.float32))
    U=tf.exp((-(idx[:,tf.newaxis]-mu)**2)/(2*si**2)) #/ np.sqrt(1.0-np.exp(-s))
    U = U / tf.sqrt(tf.reduce_sum(U**2,axis=0))
    print("NORMS 1.0", tf.reduce_sum(U**2,axis=0))
    
    #result /= K.sqrt(K.sum(K.square(result),axis=-1,keepdims=True))
    U = U * tf.sqrt(tf.constant(ISize,dtype=tf.float32))
    print("NORMS N.0", tf.reduce_sum(U**2,axis=0))
    #input()
    #result *= K.sqrt(K.constant(self.input_dim))
    #masks /= tf.sqrt(tf.reduce_sum(masks**2, axis=(0, 1, 2),keepdims=True))
    #masks *= tf.sqrt(tf.constant(kernel_size*kernel_size,dtype=tf.float32))
    #mn_U = tf.reduce_mean(U,axis=0)/U.shape[0]
    #print("Means", mn_U.numpy())
    if wdist=='normal':
        W = tf.random.normal((ISize,neurons),0, np.sqrt(2.0/ISize),name='W')
    elif wdist=='uniform':
        W = tf.random.uniform((ISize,neurons),-np.sqrt(3.0), np.sqrt(3.0),name='W')
    elif wdist=='delta':
        W = np.zeros((ISize,neurons))
        for i in range(neurons):
            W[np.int32(mu[i]*ISize),i] = np.sqrt(ISize)/np.max(U)# #np.sqrt(2.0/ISize) 
        W = tf.Variable(W, dtype=tf.float32)
    
    
    print(U.shape)
    
    #UWfilt = tf.nn.conv1d(U[:,:,np.newaxis],W[:,:,np.newaxis],
    #                      stride=[1],padding='SAME')
    UWfilt = U*W
    UWfilt = tf.squeeze(UWfilt)
    print(UWfilt.shape)
    
    UW_numpy = UWfilt.numpy()
    plt.figure(231)
    #plt.plot(U.numpy()[0:12:3,:].T, alpha=0.3)
    plt.plot(U.numpy(), alpha=0.3)
    #plt.plot(W)
    plt.figure(232)
    #plt.plot(U.numpy()[0:12:3,:].T, alpha=0.3)
    #plt.plot(UW.numpy()[:,0:-1:1],alpha=0.5)
    plt.plot(UWfilt.numpy(),alpha=0.5)
    
    
    plt.figure(233)
    #plt.plot(U.numpy()[0:12:3,:].T, alpha=0.3)
    plt.plot(np.var(UWfilt.numpy(),axis=1), alpha=0.5)
    #plt.legend(sis)
    #plt.ylim([0,2.0])
    plt.title('Variance')
    
    if wdist!='delta':
        varlistUW.append(np.mean(np.var(UWfilt.numpy(),axis=1)))
        varlistW.append(np.mean(np.var(W.numpy(),axis=1)))
    else:
        varlistUW.append(np.mean(np.var(UWfilt.numpy(),axis=1)))
        varlistW.append(np.mean(np.var(W.numpy(),axis=1)))
    print("Mean for ", s, ": ", varlistUW[-1]," -W: ",varlistW[-1])
    
plt.figure()
plt.plot(np.var(UWfilt.numpy(),axis=0))

plt.figure(235)
plt.plot(sis,varlistUW)
plt.plot(sis,varlistW)
plt.legend([r'var[$\Phi$ W]',r'var[W]'])
plt.xlabel(r'$\sigma$')
plt.ylabel('Average variance')


#plt.plot(sis,1-np.exp(-sis))