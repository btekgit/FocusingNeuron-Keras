#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Monday April 27 15:36:24 2020

@author: btek

This file contains 1D focusing layer implementation: FocusedLayer1D 
in tensorflow 2 keras.
There are some differerences from the early implementation in Keras-TF.1.4. 

1) Keras utils Optimizers do not work with tf2. Because tf2 optimizers are 
very different. Solution for this can be to cancel model.fit function and use
gradient tapes and do the training differently for Mu and Sigma. 
I have chosen a simple walk-around solution. I have added a ClipCallBack 
function to keras_utils. This works with all optimizers. However we can not
give separate learning rates.

Clipping could be done with tf.Constraints also but not sure how it works.

2) This file includes the model and tests (mnist,cifar-10, lfw-faces)
3) It requires keras_utils.py for ClipCallback

Running the file would train mnist 2-hidden focused layer network. 
Unfortunately, IT DOES NOT REACH 99.25~ without separate learning rates.
STILL WORKS BETTER THAN DENSE!
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
from tensorflow.keras import activations, regularizers, constraints
from tensorflow.keras import initializers
from tensorflow.keras.layers import InputSpec


#Keras TF implementation of Focusing Neuron.
class FocusedLayer1D(Layer):
    def __init__(self, units,
                 activation=None,
                 use_bias=True,
                 kernel_initializer=None,
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 si_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 gain=1.0,
                 init_mu = 'spread',
                 init_w = None,
                 init_sigma=0.1,
                 init_bias = initializers.Constant(0.0),
                 train_mu=True,
                 train_sigma=True, 
                 train_weights=True,
                 reg_bias=None,
                 normed=2,
                 verbose=False,
                 perrow=False,
                 sharedWeights=0,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(FocusedLayer1D, self).__init__(**kwargs)
        self.units = units
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.si_regularizer = regularizers.get(si_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True
        self.gain = gain
        self.init_sigma=init_sigma
        self.init_mu = init_mu
        self.train_mu = train_mu
        self.train_sigma = train_sigma
        self.train_weights = train_weights
        self.normed = normed
        self.verbose = verbose
        self.sharedWeights = sharedWeights
        self.sigma=None
        self.perrow=perrow # by this setting every row gets one neuron. 
        
            
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'))
        print(kwargs)
        
        #super(Focused, self).__init__(**kwargs)


    def build(self, input_shape):
        assert len(input_shape) >= 2
        self.input_dim = input_shape[-1]
        
        #self.kernel = None
        self.input_spec = InputSpec(min_ndim=2, axes={-1: self.input_dim})
        
        
        mu, si = mu_si_initializer(self.init_mu, self.init_sigma, self.input_dim,
                                   self.units, verbose=self.verbose)
        
        idxs = np.linspace(0, 1.0,self.input_dim)
        idxs = idxs.astype(dtype='float32')
        
        # idx is not trainable
        self.idxs = K.constant(value=idxs, shape=(self.input_dim,), 
                                   name="idxs")
        
        
        constant = initializers.constant
         # create trainable params.
        self.mu = self.add_weight(shape=(self.units,), 
                                  initializer=constant(mu), 
                                  name="Mu", 
                                  trainable=self.train_mu)
        self.sigma = self.add_weight(shape=(self.units,), 
                                     initializer=constant(si), 
                                     name="Sigma", 
                                     regularizer=self.si_regularizer,
                                     trainable=self.train_sigma)
    
          
        # value caps for MU and SI values
        # however these can change after gradient update.
        # MINIMUM SIGMA CAN EFFECT THE PERFORMANCE.
        # BECAUSE NEURON CAN GET SHRINK TOO MUCH IN INITIAL EPOCHS, and GET STUCK!
        MIN_SI = 0.01  # zero or below si will crashed calc_u
        MAX_SI = 1.0 
        
        # create shared vars.
        self.MIN_SI = np.float32(MIN_SI)#, dtype='float32')
        self.MAX_SI = np.float32(MAX_SI)#, dtype='float32')
        
        w_init = initializers.glorot_uniform()
        #w_init = initializers.get(self.kernel_initializer) if self.kernel_initializer else self.weight_initializer_fw_bg
        #w_init = initializers.get(self.kernel_initializer) if self.kernel_initializer else self.weight_initializer_delta_ortho
        #w_init = initializers.get(self.kernel_initializer) if self.kernel_initializer else self.weight_initializer_sr_sc
        w_shape = (self.input_dim, self.units)
        if self.sharedWeights>0:
            w_shape=(self.input_dim, self.sharedWeights) # just one copy of weights
            
        self.W = self.add_weight(shape=w_shape,
                                      initializer=w_init,
                                      name='Weights',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint,
                                      trainable=self.train_weights)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        
        self.built = True
        
        #super(FocusedLayer1D, self).build(input_shape)  # Make sure to call this somewhere!
        
        
    def call(self, inputs):
        u = self.calc_U()
        if self.verbose:
            print("weights shape", self.W.shape)
        W = self.W
        if self.sharedWeights>0:
            W = K.tile(W,[1,u.shape[1]//W.shape[1]])
            print(W.shape)
        self.kernel = W*u
        
        if not self.perrow:
            output = K.dot(inputs, self.kernel)   # XW
        else:
            output = K.sum(inputs * self.kernel, axis=-1)   # Sum(X.*W)over rows.
        if self.use_bias:
            output = K.bias_add(output, self.bias, data_format='channels_last')
        if self.activation is not None:
            output = self.activation(output)
        return output


    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1]
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)

    def get_config(self):
        config = {
            'units': self.units,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer':
                regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint),
        }
        base_config = super(self.__class__, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    def weight_initializer(self,shape):
        #only implements channel last and HE uniform
        initer = 'He'
        distribution = 'uniform'
        
        kernel = K.eval(self.calc_U())
        W = np.zeros(shape=shape, dtype='float32')
        # for Each Gaussian initialize a new set of weights
        verbose=self.verbose
        verbose=self.verbose
        if verbose:
            print("Kernel max, mean, min: ", np.max(kernel), np.mean(kernel), np.min(kernel))
            print("kernel shape:", kernel.shape, ", W shape: ",W.shape)
        
        fan_out = self.units
        
        for c in range(W.shape[1]):
            fan_in = np.sum((kernel[:,c])**2)
            
            #fan_in *= self.input_channels no need for this in repeated U. 
            if initer == 'He':
                std = self.gain * sqrt32(2.0) / sqrt32(fan_in)
            else:
                std = self.gain * sqrt32(2.0) / sqrt32(fan_in+fan_out)
            
            std = np.float32(std)
            if c == 0 and verbose:
                print("Std here: ",std, type(std),W.shape[0],
                      " fan_in", fan_in, "mx U", np.max(kernel[:,:,:,c]))
            if distribution == 'uniform':
                std = std * sqrt32(3.0)
                std = np.float32(std)
                w_vec = np.random.uniform(low=-std, high=std, size=W.shape[:-1])
            elif distribution == 'normal':
                std = std/ np.float32(.87962566103423978)           
                w_vec = np.random.normal(scale=std, size=W.shape[0])
                
            W[:,c] = w_vec.astype('float32')
            
        return W

    def weight_initializer_fw_bg(self,shape, dtype='float32'):
        '''
        Initializes weights for focusing neuron. The main idea is that
        gaussian focus effects the input variance. The weights must be
        initialized by considering focus coefficients norm"
        '''
        # the paper results were taken with this. 
        initer = 'Glorot'
        distribution = 'uniform'
        
        kernel = K.eval(self.calc_U())
        dtype = dtype.as_numpy_dtype()
        
        W = np.zeros(shape=shape, dtype=dtype)
        # for Each Gaussian initialize a new set of weights
        verbose=self.verbose
        if verbose:
            print("Kernel max, mean, min: ", np.max(kernel), np.mean(kernel), np.min(kernel))
            print("kernel shape:", kernel.shape, ", W shape: ",W.shape)
        
        fan_out = self.units
        sum_over_domain = np.sum(kernel**2,axis=1) # r base
        sum_over_neuron = np.sum(kernel**2,axis=0)
        for c in range(W.shape[1]):
            for r in range(W.shape[0]):
                fan_out = sum_over_domain[r]
                fan_in = sum_over_neuron[c]
                
                #fan_in *= self.input_channels no need for this in repeated U. 
                if initer == 'He':
                    std = self.gain * sqrt32(2.0) / sqrt32(fan_in)
                else:
                    #std = self.gain * sqrt32(1.0) / sqrt32(W.shape[1])
                    std = self.gain * sqrt32(2.0) / sqrt32(fan_in+fan_out)
                
                std = np.float32(std)
                if c == 0 and verbose:
                    print("Std here: ",std, type(std),W.shape[0],
                          " fan_in", fan_in, "mx U", np.max(kernel[:,:,:,c]))
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
    
    def weight_initializer_sr_sc(self,shape, dtype='float32'):
        '''
        Initializes weights for focusing neuron. The main idea is that
        gaussian focus effects the input variance. The weights must be
        initialized by considering focus coefficients norm"
        '''
        # the paper results were taken with this. 
        initer = 'Glorot'
        distribution = 'uniform'
        
        kernel = K.eval(self.calc_U())
        dtype = dtype.as_numpy_dtype()
        
        W = np.zeros(shape=shape, dtype=dtype)
        # for Each Gaussian initialize a new set of weights
        verbose=self.verbose
        if verbose:
            print("Kernel max, mean, min: ", np.max(kernel), np.mean(kernel), np.min(kernel))
            print("kernel shape:", kernel.shape, ", W shape: ",W.shape)
        
        #fan_out = self.units
        sum_over_domain = np.sum(kernel**2,axis=1) # r base
        sum_over_neuron = np.sum(kernel**2,axis=0)
        print("fan Out fan in", sum_over_domain, sum_over_neuron)
        fan_out = np.median(sum_over_domain)
        fan_in = np.median(sum_over_neuron)
        print("fan Out fan in", fan_out, fan_in)
                
        for c in range(W.shape[1]):
            for r in range(W.shape[0]):
                
                #fan_in *= self.input_channels no need for this in repeated U. 
                if initer == 'He':
                    std = self.gain * 2.0 / sqrt32(fan_in)
                else:
                    #std = self.gain * sqrt32(1.0) / sqrt32(W.shape[1])
                    std = self.gain *  2.0 / sqrt32(fan_in+fan_out)
                
                std = np.float32(std)
                if c == 0 and verbose:
                    print("Std here: ",std, type(std),W.shape[0],
                          " fan_in", fan_in, "mx U", np.max(kernel[:,:,:,c]))
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
    
    
    def weight_initializer_dif_var(self,shape, dtype='float32'):
        # this does not work better
        initer = 'He'
        distribution = 'uniform'
        
        kernel = K.eval(self.calc_U())
        dtype = dtype.as_numpy_dtype()
        
        W = np.zeros(shape=shape, dtype=dtype)
        # for Each Gaussian initialize a new set of weights
        verbose=self.verbose
        #if verbose:
        print("Kernel max, mean, min: ", np.max(kernel), np.mean(kernel), np.min(kernel))
        print("kernel shape:", kernel.shape, ", W shape: ",W.shape)
        
        
        fan_out = self.units
        #MIN_FOC = 0.5
        #MAX_FOC = np.max(kernel)
        N = W.shape[0]
        for c in range(W.shape[1]):
            for r in range(W.shape[0]):
                fan_in = kernel[r,c]
                #fan_in = fan_in if fan_in>=MIN_FOC else (MAX_FOC-fan_in)
                
                #fan_in *= self.input_channels no need for this in repeated U. 
                if initer == 'He':
                    std = self.gain * sqrt32(2.0) / (sqrt32(N)*fan_in)
                else:
                    std = self.gain * sqrt32(2.0) / sqrt32(fan_in+fan_out)
                
                std = np.float32(std)
                if c == 0 and verbose:
                    print("Std here: ",std, type(std),W.shape[0],
                          " fan_in", fan_in, "mx U", np.max(kernel[:,:,:,c]))
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
    
    def weight_initializer_delta_ortho(self,shape, dtype='float32'):
        # this does not work better
        initer = 'He'
        distribution = 'uniform'
        
        kernel = K.eval(self.calc_U())
        dtype = dtype.as_numpy_dtype()
        
        W = np.zeros(shape=shape, dtype=dtype)
        # for Each Gaussian initialize a new set of weights
        verbose=self.verbose
        #if verbose:
        print("Kernel max, mean, min: ", np.max(kernel), np.mean(kernel), np.min(kernel))
        print("kernel shape:", kernel.shape, ", W shape: ",W.shape)
        
        mu, si = mu_si_initializer(self.init_mu, self.init_sigma, self.input_dim,
                                   self.units)
        mu =np.int32(mu*W.shape[0])
        print(mu)
        std = sqrt32(1.0)
        for n in range(W.shape[1]):
            #W[mu[n], n] =  np.random.choice([-std,std], size=1) #np.random.uniform(low=-std, high=std, size=1)#/W.shape[0])/kernel[mu[n],n]
            #W[mu[n], n] =  std*0.25 # 
            #W[mu[n], n] =  std*1.0/W.shape[0] # 
            t = 1.0/kernel.shape[1] # was np.prod(shape)
            print("----->>>>>>>T is here:", t)
            
        
            W[mu[n], n] = np.random.choice([-t,t])
                            
            
        #print(W[:,0])
        #print(W[:,1])
        
        return W.astype('float32')
    
    def calc_U(self,verbose=False):
        """
        function calculates focus coefficients. 
        normalization has three options. 
        1) no normalization. Max u is 1.0, min u is 0.0 because of exp(-x)
        2) norm_1: the coefficient vector norm is 1.0. sum(sqr(u))==1
        3) norm_2: the coefficient vector norm is  sum(sqr(u))==sqr(n_inputs)
            norm 2 is to match the norm 1.0 when sigma is very large
        """
        up= (self.idxs - K.expand_dims(self.mu,1))**2
        #print("up.shape", up.shape)
        #up = K.expand_dims(up,axis=1,)
        #print("up.shape",up.shape)
        # clipping scaler in range to prevent div by 0 or negative cov. 
        sigma = K.clip(self.sigma,self.MIN_SI,self.MAX_SI)
        #cov_scaler = self.cov_scaler
        dwn = K.expand_dims(2 * ( sigma ** 2), axis=1)
        #scaler = (np.pi*self.cov_scaler**2) * (self.idxs.shape[0])
        #print("down shape :",dwn.shape)
        result = K.exp(-up / dwn)
        #kernel= K.eval(result)
        
        if self.normed==1:
            result /= K.sqrt(K.sum(K.square(result), axis=-1,keepdims=True))
        
        elif self.normed==2:
            result /= K.sqrt(K.sum(K.square(result), axis=-1,keepdims=True))
            result *= K.sqrt(K.constant(self.input_dim))
        
        elif self.normed==3:
            #result += 0.1 # adding a bias
            print(result.shape)
            result/= tf.expand_dims(tf.sqrt(1.0-tf.exp(-sigma)),axis=1)


            if verbose:
                kernel= K.eval(result)
                print("RESULT after NORMED max, mean, min: ", np.max(kernel), np.mean(kernel), np.min(kernel))
      
        return K.transpose(result)

        
def mu_si_initializer(initMu, initSi, num_incoming, num_units, verbose=True):
    '''
    Initialize focus centers and sigmas with regards to initMu, initSi
    
    initMu: a string, a value, or a numpy.array for initialization
    spread2d : is trying to distribute initial points on a 2D grid. This is incomplete,
    may have a bug. See try_mu_initializer.py
    initSi: a string, a value, or a numpy.array for initialization
    num_incoming: number of incoming inputs per neuron
    num_units: number of neurons in this layer
    '''
    
    if isinstance(initMu, str):
        if initMu == 'middle':
            #print(initMu)
            mu = np.repeat(.5, num_units)  # On paper we have this initalization                
        elif initMu =='middle_random':
            mu = np.repeat(.5, num_units)  # On paper we have this initalization
            mu += (np.random.rand(len(mu))-0.5)*(1.0/(float(20.0)))  # On paper we have this initalization                
            
        elif initMu == 'spread':
            #paper results were taken with np.linspace(0.2, 0.8, num_units)  . 
            # THIS AFFECTS RESULTS!!!
            mu = np.linspace(0.2, 0.8, num_units)  
            #mu = np.linspace(0.0, 1.0, num_units)  
            #mu = np.linspace(0.1, 0.9, num_units)
        elif initMu == 'spread2d':
            # I create 2D grid equally distributed to cover most of the image
            # howewer this assumes gray scale I need a color version
            in_wid = np.sqrt(num_incoming)
            ns = np.sqrt(num_units)
            if in_wid != int(in_wid) or ns!=int(in_wid):
                print(initMu, "Works best in square images")
            
            in_wid= int(in_wid)
            in_hei = num_incoming//in_wid
            
            ns_wid= int(ns)
            ns_hei = num_units//ns_wid
            ns_min = min(ns_wid,ns_hei)
            vert = np.linspace(in_hei*0.2,in_hei*0.8, ns_min) 
            hor = np.linspace(in_wid*0.2,in_wid*0.8, ns_min)

            mu_ = vert*in_wid
            mu= (mu_+ hor[:,np.newaxis]).reshape(in_wid*in_hei) / (num_incoming)
            
    elif isinstance(initMu, float):  #initialize it with the given scalar
        mu = np.repeat(initMu, num_units)  # 

    elif isinstance(initMu,np.ndarray):  #initialize it with the given array , must be same length of num_units
        if initMu.max() > 1.0:
            print("Mu must be [0,1.0] Normalizing initial Mu value")
            initMu /=(num_incoming - 1.0)
            mu = initMu        
        else:
            mu = initMu
    
    #Initialize sigma
    if isinstance(initSi,str):
        if initSi == 'random':
            si = np.random.uniform(low=0.05, high=0.25, size=num_units)
        elif initSi == 'normal':
            si = np.random.uniform(low=0.05, high=0.25, size=num_units)
            #si = np.repeat((initSi / num_units), num_units)
            pass
            print("not implemented")

    elif isinstance(initSi,float):  #initialize it with the given scalar
        si = np.repeat(initSi, num_units)# 
        
    elif isinstance(initSi, np.ndarray):  #initialize it with the given array , must be same length of num_units
        si = initSi
        
    # Convert Types for GPU
    mu = mu.astype(dtype='float32')
    si = si.astype(dtype='float32')

    if verbose:
        print("mu init:", mu)
        print("si init:", si)
        
    return mu, si



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
    
    if normed==1:
        ex /= np.sqrt(np.sum(np.square(ex), axis=-1,keepdims=True))
    elif normed==2:
        ex /= np.sqrt(np.sum(np.square(ex), axis=-1,keepdims=True))
        ex *= np.sqrt(idxs.shape[0])

    return (np.transpose(ex.astype(dtype='float32')))

def calculate_fi_and_weights(layer_instance):
    ''' 
    This aux function calculates its focus functions, 
    focused weights for a given
    a layer instance
    '''
    w = layer_instance.get_weights()
    mu = w[0]
    si = w[1]
    we = w[2]
    idxs = np.linspace(0, 1.0,layer_instance.input_shape[1])
    fi = U_numeric(idxs, mu, si, scaler=1.0, normed=2)
    fiwe =  fi*we
    
    return fi, we, fiwe

def prune_out_of_focus_weights(layer_instance, threshold=1e-8):
    ''' 
    This aux function prunes weights using focus function value
    '''
    w = layer_instance.get_weights()
    fi, we, fiwe = calculate_fi_and_weights(layer_instance)
    in_focus = (fi>threshold).astype('float')
    we_p = we*in_focus
    w[2]=we_p
    layer_instance.set_weights(w)
    
    return fi, in_focus, we_p

def sqrt32(x):
    return np.sqrt(x,dtype='float32')
    

#%% For tests
  
  
    
def create_simple_model(input_shape, num_classes=10, settings={}):
    from tensorflow.keras.models import  Model
    from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, BatchNormalization, AlphaDropout
    from tensorflow.keras.layers import Activation # MaxPool2D
    act_func = 'relu'
    act_func = 'selu'
    act_func = 'relu'
    #drp_out = AlphaDropout
    drp_out = Dropout
    #from keras.regularizers import l2
    
    node_in = Input(shape=input_shape, name='inputlayer')

    node_fl = Flatten(data_format='channels_last')(node_in)
    
    node_ = Dropout(0.2)(node_fl)
    heu= initializers.he_uniform
    h = 1 
    for nh in settings['nhidden']:
        if settings['neuron']=='focused':
            init_mu = settings['focus_init_mu']
            node_ = FocusedLayer1D(units=nh,
                                   name='focus-'+str(h),
                                   activation='linear',
                                   init_sigma=settings['focus_init_sigma'], 
                                   init_mu=init_mu,
                                   init_w= None,
                                   train_sigma=settings['focus_train_si'], 
                                   train_weights=settings['focus_train_weights'],
                                   si_regularizer=settings['focus_sigma_reg'],
                                   train_mu = settings['focus_train_mu'],
                                   normed=settings['focus_norm_type'],
                                   gain=1.0,sharedWeights=0)(node_)
        else:
            node_ = Dense(nh,name='dense-'+str(h),activation='linear',
                          kernel_initializer=heu())(node_)
        
        node_ = BatchNormalization()(node_)
        node_ = Activation(act_func)(node_)
        node_ = drp_out(0.25)(node_)
        h = h + 1
    
    node_fin = Dense(num_classes, name='softmax', activation='softmax', 
                     kernel_initializer=initializers.he_uniform(),
                    kernel_regularizer=None)(node_)
    
    model = Model(inputs=node_in, outputs=[node_fin])
    
    return model


def create_simple_residual_model(input_shape,num_classes=10, settings={}):
    from keras.models import  Model
    from keras.layers import Input, Dense, Dropout, Flatten, BatchNormalization
    from keras.layers import Activation, AveragePooling2D #,Add
    #from keras.regularizers import l2
    node_in = Input(shape=input_shape, name='inputlayer')
    
    node_in_pooled = AveragePooling2D()(node_in)
    #node_in_pooled_fl =Flatten(data_format='channels_last')(node_in_pooled)
    node_fl = Flatten(data_format='channels_last')(node_in)
    #node_fl = node_in
    node_ = Dropout(0.2)(node_fl)
    heu= initializers.he_uniform
    h = 1
    
    for nh in settings['nhidden']:
        if settings['neuron']=='focused':
            if settings['focus_init_mu'] == 'spread':
                init_mu = np.linspace(0.1,0.90,nh)
            else:
                init_mu = settings['focus_init_mu']
            node_ = FocusedLayer1D(units=nh,
                                   name='focus-'+str(h),
                                   activation='linear',
                                   init_sigma=settings['focus_init_sigma'], 
                                   init_mu=init_mu,
                                   init_w= None,
                                   train_sigma=settings['focus_train_si'], 
                                   train_weights=settings['focus_train_weights'],
                                   si_regularizer=settings['focus_sigma_reg'],
                                   train_mu = settings['focus_train_mu'],
                                   normed=settings['focus_norm_type'])(node_)
        else:
            node_ = Dense(nh,name='dense-'+str(h),activation='linear',kernel_initializer=heu())(node_)
        
        
        node_ = BatchNormalization()(node_)
        #node_ = Add()([node_, node_in_pooled_fl])
        node_ = Activation('relu')(node_)
        node_ = Dropout(0.5)(node_)
        h = h + 1
    
    node_fin = Dense(num_classes, name='softmax', activation='softmax', 
                     kernel_initializer=initializers.he_uniform(),
                    kernel_regularizer=None)(node_)
    
    model = Model(inputs=node_in, outputs=[node_fin])
    
    return model

    

def create_cnn_model(input_shape,  num_classes=10, settings={}):
    from tensorflow.keras.models import  Model
    from tensorflow.keras.layers import Input, Dense, Dropout, Flatten,Conv2D, BatchNormalization
    from tensorflow.keras.layers import Activation, MaxPool2D
    
    node_in = Input(shape=input_shape, name='inputlayer')
    
    node_conv1 = Conv2D(filters=settings['nfilters'][0],kernel_size=settings['kn_size'][0], padding='same',
                        activation='relu')(node_in)
    node_conv2 = Conv2D(filters=settings['nfilters'][1],kernel_size=settings['kn_size'][0], padding='same',
                        activation='relu')(node_conv1)
    #node_conv3 = Conv2D(filters=nfilters,kernel_size=kn_size, padding='same',
    #                    activation='relu')(node_conv2)

    node_pool = MaxPool2D((2,2))(node_conv2)
    #node_pool = MaxPool2D((4,4))(node_conv2) works good. 
    node_fl = Flatten(data_format='channels_last')(node_pool)
    #node_fl = Flatten(data_format='channels_last')(node_conv2)

    #node_fl = node_in
    # smaller initsigma does not work well. 
    node_ = Dropout(0.5)(node_fl)
    heu= initializers.he_uniform
    h = 1
    
    for nh in settings['nhidden']:
        if settings['neuron']=='focused':
            init_mu = settings['focus_init_mu']
            node_ = FocusedLayer1D(units=nh,
                                   name='focus-'+str(h),
                                   activation='linear',
                                   init_sigma=settings['focus_init_sigma'], 
                                   init_mu=init_mu,
                                   init_w= None,
                                   train_sigma=settings['focus_train_si'], 
                                   train_weights=settings['focus_train_weights'],
                                   si_regularizer=settings['focus_sigma_reg'],
                                   #si_regularizer=None,
                                   train_mu = settings['focus_train_mu'],
                                   normed=settings['focus_norm_type'])(node_)
                                   #si_regularizer=None,
                                   
        else:
            node_ = Dense(nh,name='dense-'+str(h),activation='linear',
                          kernel_initializer=heu())(node_)
    
        node_ = BatchNormalization()(node_)
        node_ = Activation('relu')(node_)
        node_ = Dropout(0.5)(node_)
        h = h + 1
    
    node_fin = Dense(num_classes, name='softmax', activation='softmax', 
                     kernel_initializer=initializers.he_uniform(),
                     kernel_regularizer=None)(node_)

    #decay_check = lambda x: x==decay_epoch

    model = Model(inputs=node_in, outputs=[node_fin])
    
    return model
    
        

def test_comp(settings,random_sid=9):
    import tensorflow.keras as keras4
    #from keras.optimizers import SGD
    from tensorflow.keras.datasets import mnist,fashion_mnist, cifar10    
    #from skimage import filters
    from tensorflow.keras import backend as K
    #from keras_utils import WeightHistory as WeightHistory
    from keras_utils_tf2 import RecordVariable, RecordOutput, \
    PrintLayerVariableStats, SGDwithLR, eval_Kdict, standarize_image_025,\
    standarize_image_01, AdamwithClip, ClipCallback
    
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    K.clear_session()
    
    epochs = settings['Epochs']
    batch_size = settings['batch_size']

    sid = random_sid  
    np.random.seed(sid)
    tf.random.set_seed(sid)
    
    # MINIMUM SIGMA CAN EFFECT THE PERFORMANCE.
    # BECAUSE NEURON CAN GET SHRINK TOO MUCH IN INITIAL EPOCHS WITH LARGER GRADIENTS
    #, and GET STUCK!
    MIN_SIG = 0.01
    MAX_SIG = 1.0
    MIN_MU = 0.0
    MAX_MU = 1.0
    lr_dict = {'all':settings['lr_all']} #0.1 is default for MNIST
    mom_dict = {'all':0.9}    
    decay_dict = {'all':0.5} # works 99.29 with mnist
    clip_dict ={}
    #UPDATE_Clip = 0.25
    for i,n in enumerate(settings['nhidden']):
            lr_dict.update({'focus-'+str(i+1)+'/Sigma:0':0.01}) # best results 0.01
            lr_dict.update({'focus-'+str(i+1)+'/Mu:0':0.01}) #best results 0.01
            lr_dict.update({'focus-'+str(i+1)+'/Weights:0':0.1})
            
            mom_dict.update({'focus-'+str(i+1)+'/Sigma:0':0.9})
            mom_dict.update({'focus-'+str(i+1)+'/Mu:0':0.9})
            
            decay_dict.update({'focus-'+str(i+1)+'/Sigma:0':0.5})  #best results 0.5
            decay_dict.update({'focus-'+str(i+1)+'/Mu:0':0.9})
            
            clip_dict.update({'focus-'+str(i+1)+'/Sigma:0':(MIN_SIG,MAX_SIG)})
            clip_dict.update({'focus-'+str(i+1)+'/Mu:0':(MIN_MU,MAX_MU)})
            
    print("Loading dataset")
    if settings['dset']=='mnist':
        # input image dimensions 
        img_rows, img_cols = 28, 28  
        # the data, split between train and test sets
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        n_channels=1
           
        e_i = x_train.shape[0] // batch_size
        decay_epochs =np.array([e_i*100, e_i*150], dtype='int64')
        if settings['cnn_model']:
                   
           decay_epochs =[e_i*30,e_i*100]
    
    elif settings['dset']=='cifar10':
        img_rows, img_cols = 32,32
        n_channels=3
        
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        # works good as high as 77 for cnn-focus
        #decay_dict = {'all':0.9, 'focus-1/Sigma:0': 1.1,'focus-1/Mu:0':0.9,
        #          'focus-2/Sigma:0': 1.1,'focus-2/Mu:0': 0.9}
        #if cnn_model: batch_size=256 # this works better than 500 for cifar-10
        e_i = x_train.shape[0] // batch_size
        decay_epochs =np.array([e_i*30,e_i*80,e_i*120,e_i*180], dtype='int64')
        #decay_epochs =np.array([e_i*10], dtype='int64')
        
    elif settings['dset']=='fashion':
        img_rows, img_cols = 28,28
        n_channels=1

        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
               
        e_i = x_train.shape[0] // batch_size
        decay_epochs =np.array([e_i*100, e_i*150], dtype='int64')
        if  settings['cnn_model']:
            decay_dict = {'all':0.9, 'focus-1/Sigma:0': 0.9,'focus-1/Mu:0':0.9,
                  'focus-2/Sigma:0': 0.9,'focus-2/Mu:0': 0.9}

            decay_epochs =[e_i*30,e_i*100]
                     
    elif settings['dset']=='mnist-clut':
        
        img_rows, img_cols = 60, 60  
        # the data, split between train and test sets
        
        folder='/media/home/rdata/image/'
        data = np.load(folder+"mnist_cluttered_60x60_6distortions.npz")
    
        x_train, y_train = data['x_train'], np.argmax(data['y_train'],axis=-1)
        x_valid, y_valid = data['x_valid'], np.argmax(data['y_valid'],axis=-1)
        x_test, y_test = data['x_test'], np.argmax(data['y_test'],axis=-1)
        x_train=np.vstack((x_train,x_valid))
        y_train=np.concatenate((y_train, y_valid))
        n_channels=1
        
        lr_dict = {'all':0.01}
        
        e_i = x_train.shape[0] // batch_size
        decay_epochs =np.array([e_i*100, e_i*150], dtype='int64')
        if  settings['cnn_model']:
            decay_epochs =[e_i*30,e_i*100]
            
    elif settings['dset']=='lfw_faces':
        from sklearn.datasets import fetch_lfw_people
        lfw_people = fetch_lfw_people(min_faces_per_person=20, resize=0.4)
        
        # introspect the images arrays to find the shapes (for plotting)
        n_samples, img_rows, img_cols = lfw_people.images.shape
        n_channels=1
        
        X = lfw_people.data
        n_features = X.shape[1]
        
        # the label to predict is the id of the person
        y = lfw_people.target
        target_names = lfw_people.target_names
        n_classes = target_names.shape[0]
        
        print("Total dataset size:")
        print("n_samples: %d" % n_samples)
        print("n_features: %d" % n_features)
        print("n_classes: %d" % n_classes)
        
        from sklearn.model_selection import train_test_split
        
        #X -= X.mean()
        #X /= X.std()
        #split into a training and testing set
        x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42)
        
        import matplotlib.pyplot as plt
        
        plt.imshow(X[0].reshape((img_rows,img_cols)))
        plt.show()
        lr_dict = {'all':0.5}
        
        e_i = x_train.shape[0] // batch_size
        decay_epochs =np.array([e_i*50,e_i*100, e_i*150], dtype='int64')

    
    
    num_classes = np.unique(y_train).shape[0]
    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], n_channels, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], n_channels, img_rows, img_cols)
        input_shape = (n_channels, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, n_channels)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, n_channels)
        input_shape = (img_rows, img_cols, n_channels)
    if settings['dset']!='mnist-clut':
        
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        
        x_train, _, x_test = standarize_image_01(x_train, tst=x_test)
        #x_train, _, x_test = standarize_image_025(x_train, tst=x_test)
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, n_channels)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, n_channels)
    
    input_shape = (img_rows, img_cols, n_channels)    
    # convert class vectors to binary class matrices
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)

    #from keras_data_tf2 import load_dataset
    #dset = settings['dset']
    #ld_data = load_dataset(dset,normalize_data=True,options=[])
    #x_train,y_train,x_test,y_test,input_shape,num_classes=ld_data
    
    
    
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    
    sigma_reg = settings['focus_sigma_reg']
    sigma_reg = tf.keras.regularizers.l2(sigma_reg) if sigma_reg is not None else sigma_reg
    settings['focus_sigma_reg'] = sigma_reg
    if settings['cnn_model']:
        model=create_cnn_model(input_shape,num_classes, settings=settings)
    else:
        model=create_simple_model(input_shape, num_classes, settings=settings)
    
 
    model.summary()

    #opt= SGDwithLR(lr_dict, mom_dict,decay_dict,clip_dict, 
    #               decay_epochs,clipvalue=1.0)#, decay=None)
    #opt = AdamwithClip(clips=clip_dict)
    #opt= SGDwithLR(lr_dict, mom_dict,decay_dict,clip_dict, 
    #                decay_epochs,update_clip=UPDATE_Clip)#, decay=None)
    
    opt = tf.keras.optimizers.SGD(lr=settings['lr_all'], momentum=0.9, clipvalue=1.0)
    #opt = tf.keras.optimizers.Adam(lr=1e-1, clipvalue=1.0)
                   
    model.compile(loss=tf.keras.losses.categorical_crossentropy,
                  optimizer=opt,
                  metrics=['accuracy'])
    
    
    
        
    stat_func_name = ['max: ', 'mean: ', 'min: ', 'var: ', 'std: ']
    stat_func_list = [np.max, np.mean, np.min, np.var, np.std]
    #callbacks = [tb]
    callbacks = []
    
    def fix_lr(epoch, lr, decay_epoch=(100,150), decay=0.9):
        if epoch in decay_epoch: 
            return lr*decay
        return lr
            
    
    def gauss_lr(epoch, lr, min_lr=1e-3, max_lr=5e-1, center=60, lrsigma=0.2):
            
        return (min_lr+ max_lr*np.exp(-(epoch-center)**2/(center*lrsigma)**2))
    #red_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1, mode='auto', min_delta=0.0001, cooldown=10, min_lr=1e-5)
    red_lr = tf.keras.callbacks.LearningRateScheduler(fix_lr,verbose=1)
    callbacks+=[red_lr]
    #callbacks+=[fix_lr]
    
    if  settings['neuron']=='focused':
        ccp1 = ClipCallback('Sigma',[MIN_SIG,MAX_SIG])
        ccp2 = ClipCallback('Mu',[MIN_MU,MAX_MU])
        #ccp = ClipCallback('Sigma',[MIN_SIG,MAX_SIG])
        pr_1 = PrintLayerVariableStats("focus-1","Weights:0",stat_func_list,stat_func_name)
        pr_2 = PrintLayerVariableStats("focus-1","Sigma:0",stat_func_list,stat_func_name)
        pr_3 = PrintLayerVariableStats("focus-1","Mu:0",stat_func_list,stat_func_name)
        #pr_4 = PrintLayerVariableStats("focus-2","Sigma:0",stat_func_list,stat_func_name)
        #red_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=10, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=1e-5)
        #schedule = lambda epoch: np.exp()
        
        callbacks += [ccp1, ccp2, pr_1, pr_2, pr_3]    





    if not settings['augment']:
        print('Not using data augmentation.')
        history=model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_data=(x_test, y_test),
                  shuffle=True,
                  callbacks=callbacks)
    else:
        print('Using real-time data augmentation.')
        # This will do preprocessing and realtime data augmentation:
        datagen = ImageDataGenerator(
            # set input mean to 0 over the dataset
            featurewise_center=False,
            # set each sample mean to 0
            samplewise_center=False,
            # divide inputs by std of dataset
            featurewise_std_normalization=False,
            # divide each input by its std
            samplewise_std_normalization=False,
            # apply ZCA whitening
            zca_whitening=False,
            # epsilon for ZCA whitening
            zca_epsilon=1e-06,
            # randomly rotate images in the range (deg 0 to 180)
            rotation_range=0,
            # randomly shift images horizontally
            width_shift_range=0.1,
            # randomly shift images vertically
            height_shift_range=0.1,
            # set range for random shear
            shear_range=0.,
            # set range for random zoom
            zoom_range=0.,
            # set range for random channel shifts
            channel_shift_range=0.,
            # set mode for filling points outside the input boundaries
            fill_mode='nearest',
            # value used for fill_mode = "constant"
            cval=0.,
            # randomly flip images
            horizontal_flip=True,
            # randomly flip images
            vertical_flip=False,
            # set rescaling factor (applied before any other transformation)
            rescale=None,
            # set function that will be applied on each input
            preprocessing_function=None,
            # image data format, either "channels_first" or "channels_last"
            data_format='channels_last',
            # fraction of images reserved for validation (strictly between 0 and 1)
            validation_split=0.0)
    
        # Compute quantities required for featurewise normalization
        # (std, mean, and principal components if ZCA whitening is applied).
        datagen.fit(x_train)
    
        # Fit the model on the batches generated by datagen.flow().
        history=model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                            validation_data=(x_test, y_test),
                            epochs=epochs, verbose=1, workers=4,
                            callbacks=callbacks, 
                            steps_per_epoch=x_train.shape[0]//batch_size)
    
    
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    return score, history, model, callbacks


    

def repeated_trials(test_function=None, settings={}):
    
    list_scores =[]
    list_histories =[]
    list_sigmas = []
    sigmas = settings['focus_sigma_reg']
    sigmas = [None] if sigmas is None or sigmas is [] else sigmas 
    models = []
    
    import time 
    print("Delayed start ",delayed_start)
    time.sleep(delayed_start)
    from datetime import datetime
    now = datetime.now()
    timestr = now.strftime("%Y%m%d-%H%M%S")
    
    filename = 'outputs/'+settings['dset']+'/'+timestr+'_'+settings['neuron']+'.model_results.npz'
    #copyfile("Kfocusingtf2.py",filename+"code.py")
   
    for s in range(len(sigmas)): # sigmas loop, should be safe if it is empty
        for i in range(settings['repeats']):
            
            sigma_reg = sigmas[s] if sigmas else None
            print("REPEAT",i,"sigma regularization", sigma_reg)
            #run_settings = settings.copy()
            settings['focus_sigma_reg'] = sigma_reg
            sc, hs, ms, cb = test_function(random_sid=i*17,settings=settings)
            list_scores.append(sc)
            list_histories.append(hs)
            models.append(ms)
            # record current regularizer and final sigma 
            if settings['neuron']=='focused' and sigma_reg:
                list_sigmas.append([sigma_reg, np.mean(cb[4].record[-1])])
            
    print("Final scores", list_scores)
    print("History KEYS:",list_histories[i].history.keys())
    mx_scores = [np.max(list_histories[i].history['val_accuracy']) for i in range(len(list_histories))]
    
    histories = [m.history.history for m in models]
    print("Max sscores", mx_scores)
    
    try:
        np.savez_compressed(filename,mx_scores =mx_scores, list_scores=list_scores, 
                        modelz=histories, sigmas=list_sigmas)
    except:
        print("I could not save into",filename) 
    return mx_scores, list_scores, histories, list_sigmas


   
    
if __name__ == "__main__":
    import os
    if 'CUDA_VISIBLE_DEVICES' not in os.environ.keys():
        os.environ['CUDA_VISIBLE_DEVICES']="0"
        os.environ['TF_FORCE_GPU_ALLOW_GROWTH']="true"
    print("Run as main")
    #test()
    delayed_start = 0*3600
    import time 
    from shutil import copyfile
    print("Delayed start ",delayed_start)
    time.sleep(delayed_start)
    dset='mnist' 
    #dset='cifar10'  # ~64,5 cifar is better with batch 256, init_sigma =0.01 
    dset='mnist'
    #dset = 'mnist-clut'
    #dset = 'fashion'
    #dset='lfw_faces' # ~78,use batch_size = 32, augment=True, init_sigm=0.025, init_mu=spread
    sigma_reg_set = None
    nhidden = (800,800)
    #nhidden = (256,)
    #sigma_reg_set = [1e-10, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]

    
    mod={'dset':dset, 'neuron':'focused', 'nhidden':nhidden, 'cnn_model':False,
         'nfilters':(32,32), 'kn_size':(5,5),
         'focus_init_sigma':0.025, 'focus_init_mu':'spread','focus_train_mu':True, 
         'focus_train_si':True,'focus_train_weights':True,'focus_norm_type':2,
         'focus_sigma_reg':sigma_reg_set,'augment':False, 
         'Epochs':200, 'batch_size':512,'repeats':5,
         'lr_all':0.1}
    
    # lr_all 0.1 for MNIST
    # lr_all:0.01 for CIFAR-FACES-CLUT 
    # FACES OVERWRITES THIS IN THE CODE. lr_all =0.001
    # ALSO I USE GRAD CLIP 0.01. IT IS OK IF YOU DO GRAD_CLIP 1.0
    

    
    f = test_comp
    res = repeated_trials(test_function=f,settings=mod)
    try:
        fname ='outputs/'+'_'+mod['dset']+'_'+'cnn'+str(mod['cnn_model'])+'_regularization.npz'
        np.savez_compressed(fname, res)
    except:
        print("I could not save into",fname)
        
    import matplotlib.pyplot as plt
    if sigma_reg_set:
        
        plt.plot(np.log10(sigma_reg_set),np.reshape(res[0],(-1,5)),'o')
        plt.errorbar(x=np.log10(sigma_reg_set),y=np.mean(np.reshape(res[0],(-1,5)),axis=1),
                     yerr=np.std(np.reshape(res[0],(-1,5)),axis=1))
        
    else:
        plt.plot(np.reshape(res[0],(-1,mod['repeats'])),'o')
    # SOME RESULTS 
    # focused MNIST Augmented accuracy (200epochs): ~99.25-99.30
    # focused MNIST No Augment accuracy(200 epochs): ~99.25
    # Max sscores [0.9926999999046325, 0.9925999997138977, 0.9921999997138977, 0.9922999998092651, 0.9923999998092652]
    # if you put 0.5 decay to all Max sscores [0.9928999998092651, 0.9923999997138977, 0.9920999997138977, 0.9923999998092652, 0.9923999996185303] 
    # res = repeated_trials(dset='cifar10',N=1,epochs=200, augment=False, mod=mod, test_function=f)
    # focused CIFAR-10 Augmented accuracy(200epochs): ~0.675
    # focused CIFAR-10 NONAugmented 63.5
    #
    
    # CNN results CIFAR-10 (200 epochs) max: 74.16 no augmentation
    # CNN results CIFAR_10 (200 epochs)max: 76.32 with batch:32
    # focus mx_1 = [0.7368000005722046, 0.7413000006675721, 0.7416000005722045, 0.741200000667572, 0.7374999997138977]
    # dense mx_2 =[0.7257999999046326, 0.7290000000953675, 0.7257999997138977, 0.7214000000953674, 0.7229000000953675]
    # with batch 256 and maxpool(4,4) focus reaches 81
    
    # CNN results MNIST  max : 99.63 focus, 99.63 dense
    # focused mx_1 = [0.9958999999046325, 0.9958999998092651, 0.9959999999046326, 0.9960999999046326, 0.9960999999046326]
    # dense mx_2=[Max sscores [0.9958999999046325, 0.9958999999046325, 0.9962999998092651, 0.9959999999046326, 0.9954999998092652]]

    # lfw face simple  focus augment on.
    # FOCUS: Max sscores [0.7338582669656107, 0.7417322843093571, 0.7543307095062075, 0.7527559056995422, 0.7385826764144297]
    # DENSE:Max sscores [0.6960629923137154, 0.6992125992699871, 0.6992125992699871, 0.7055118109297565, 0.6929133860145028]
    # FIXED simple (sig=0.025) : Max sscores [0.6944881882254533, 0.6976377945246659, 0.7086614181676248, 0.6881889756270281, 0.6992125995515839]
    # FIXED simple (sig=0.08) :Max sscores [0.6992125986129281, 0.6850393702664713, 0.703937007123091, 0.6866141743547335, 0.6897637797152902]



    # lfw face cnn 
    # Focus:Max sscores [0.8944881883193189, 0.8881889766595495, 0.896062992407581, 0.8834645672107306, 0.8913385829587621]
    # Dense: Max sscores [0.8787401587005675, 0.8929133870470243, 0.8866141728528841, 0.8881889757208937, 0.8787401587005675]
    
    # FIXED simple init_si=0.08, init_mu:'spread', lr=0.1, train_mu=False, train_si=False
    # MNIST : Max sscores [0.9926999997138977, 0.9919999997138977, 0.9917999997138977, 0.9923999998092652, 0.9920999997138977]

    # FIXED: Max sscores [0.8913385838974179, 0.896062992407581, 0.8818897649997801, 0.8850393697032778, 0.9007874027950558] cnn sigma =0.025

    # FOCUSED-C (center starting sig=0.025): 
    # MNIST :[0.9913999997138977, 0.9909999996185302, 0.9911999995231628, 0.9907999997138977, 0.9910999997138977]
    # CLUT :  [0.6750000007629394, 0.668100000667572, 0.6753000002861023, 0.6707000003814697, 0.6701000000953674]
    # FASHION: [0.9082999998092651, 0.9091, 0.9068999997138977, 0.9084999999046326, 0.9088999996185303]
    # CIFAR: [0.6272, 0.6252, 0.6247, 0.6289, 0.6247]
    # FACES : [0.7401574802210951, 0.7511811013296833, 0.7401574792824392, 0.7385826761328329, 0.7370078748605383]




    