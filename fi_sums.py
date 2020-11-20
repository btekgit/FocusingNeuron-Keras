#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 17:08:08 2020

@author: btek
"""

#U sums
import matplotlib.pyplot as plt
import numpy as np
import math
def U_numeric(idxs, mus, sis, scaler, normed=2):
    '''
    This function provides a numeric computed focus coefficient vector for plots
    
    idxs: the set of indexes (positions) to calculate Gaussian focus coefficients
    
    mus: a numpy array of focus centers
    
    sis: a numpy array of focus aperture sigmas
    
    scaler: a scalar value
    
    normed: apply sum normalization
        
    '''
    
    up = (idxs - mus[:, np.newaxis]) ** 2
    down = (2 * (sis[:, np.newaxis] ** 2))
    ex = np.exp(-up / down)
    ex /= (sis[:, np.newaxis]*np.sqrt(2.0*np.pi))
    
    if normed==1:
        ex /= np.sqrt(np.sum(np.square(ex), axis=-1,keepdims=True))
    elif normed==2:
        ex /= np.sqrt(np.sum(np.square(ex), axis=-1,keepdims=True))
        ex *= np.sqrt(idxs.shape[0])
    elif normed==3:
        ex /= np.sqrt(np.sum(np.square(ex)))
        

    return (np.transpose(ex.astype(dtype='float32')))

def weight_initializer_fw_bg(kernel):
        '''
        Initializes weights for focusing neuron. The main idea is that
        gaussian focus effects the input variance. The weights must be
        initialized by considering focus coefficients norm"
        '''
        # the paper results were taken with this. 
        initer = 'Glorot'
        distribution = 'uniform'
        
        sqrt32 = lambda x: np.sqrt(x,dtype=np.float32) 
        gain = 1.0
        
        W = np.zeros_like(kernel)
        # for Each Gaussian initialize a new set of weights
        verbose=True
        if verbose:
            print("Kernel max, mean, min: ", np.max(kernel), np.mean(kernel), np.min(kernel))
            print("kernel shape:", kernel.shape, ", W shape: ",W.shape)
        
        fan_out = 0
        sum_over_domain = np.sum(kernel**2,axis=1) # r base
        sum_over_neuron = np.sum(kernel**2,axis=0)
        for c in range(W.shape[1]):
            for r in range(W.shape[0]):
                fan_out = sum_over_domain[r]
                fan_in = sum_over_neuron[c]
                #fan_out = np.mean(sum_over_domain)
                
                #fan_in *= self.input_channels no need for this in repeated U. 
                if initer == 'He':
                    std = gain * sqrt32(2.0) / sqrt32(fan_in)
                else:
                    #std = self.gain * sqrt32(1.0) / sqrt32(W.shape[1])
                    std = gain * sqrt32(2.0) / sqrt32(fan_in+fan_out)
                    
                
                std = np.float32(std)
                if c == 0 and verbose:
                    print("Std here: ",std, type(std),W.shape[0],
                          " fan_in", fan_in, "mx U", np.max(kernel[:,c]))
                    print(r,",",c," Fan in ", fan_in, " Fan_out:", fan_out, W[r,c])
                    
                if distribution == 'uniform':
                    std = std * sqrt32(3.0)
                    std = np.float32(std)
                    w_vec = np.random.uniform(low=-std, high=std, size=1)
                elif distribution == 'normal':
                    std = std/ np.float32(.87962566103423978)           
                    w_vec = np.random.normal(scale=std, size=1)
                    
                W[r,c] = w_vec.astype('float32')
                
        return W

def weight_initializer_delta_ortho(kernel,mu, si):
    # this does not work better
    initer = 'He'
    sqrt32 = lambda x: np.sqrt(x,dtype=np.float32) 
    gain = 1.0
    
    W = np.zeros_like(kernel)
    # for Each Gaussian initialize a new set of weights
    verbose=True
    #if verbose:
    print("Kernel max, mean, min: ", np.max(kernel), np.mean(kernel), np.min(kernel))
    print("kernel shape:", kernel.shape, ", W shape: ",W.shape)
    
    
    mu =np.int32(mu*W.shape[0])
    print(mu)
    std = sqrt32(1.0)
    for n in range(W.shape[1]):
        #W[mu[n], n] =  np.random.choice([-std,std], size=1) #np.random.uniform(low=-std, high=std, size=1)#/W.shape[0])/kernel[mu[n],n]
        #W[mu[n], n] =  std*0.25 # 
        #W[mu[n], n] =  std*1.0/W.shape[0] # 
        W[mu[n], n] = std
                        
        
    #print(W[:,0])
    #print(W[:,1])
    
    return W.astype('float32')

Na = 784# input
Nb = 784 # output
st = 0.0125
K = 1.0
st2 = st/K
Na2 = Na*K
init_sigma = np.ones((Nb))*(st2)
init_mu = np.linspace(0.000, 0.999, Nb)#np.random.normal(loc=0.5, scale=0.2,size=Nb)
idxs = np.linspace(0,1.0,Na2)
U =U_numeric(idxs, init_mu,init_sigma ,scaler=1.0, normed=0)
W = weight_initializer_fw_bg(U)
#W = weight_initializer_delta_ortho(U,init_mu,init_sigma)
plt.figure(figsize=(10,10))
plt.imshow(U)

plt.colorbar()
plt.figure(figsize=(10,10))
plt.plot(U)
plt.xlim([0,Na])

plt.figure(figsize=(10,10))
plt.imshow(U*W)
plt.colorbar()
plt.figure(figsize=(10,10))
plt.plot(U*W)
plt.xlim([0,Na])


rs= np.sqrt(np.sum(U**2,axis=0))
cs= np.sqrt(np.sum(U**2,axis=1))
rsw= np.sqrt(np.sum((U*W)**2,axis=0))
csw= np.sqrt(np.sum((U*W)**2,axis=1))
plt.figure(figsize=(12,8))
plt.subplot(221)
plt.plot(rs,'o')
plt.subplot(222)
plt.plot(rsw,'*')
plt.ylim([0,2])
plt.title('col sums')

locs, labels = plt.yticks()
plt.ticklabel_format(style='plain')
print(locs, labels)
plt.subplot(223)
plt.plot(cs,'o')
plt.subplot(224)
plt.plot(csw,'*')
plt.ylim([0,2])
plt.title('row sums')