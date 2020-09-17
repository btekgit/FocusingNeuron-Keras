#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 15:36:24 2019

@author: btek

This file contains 1D focusing layer implementation: FocusedLayer1D 
in keras. 
There are some differerences from the early implementation in Theano. 

1) This file includes the model and tests (mnist,cifar-10, lfw-faces)
2) It requires my keras_utils.py for optimizers (that work with dictionaries and clips)
3) Just running the file would work mnist 2-hidden layer network. 
"""

from keras import backend as K
from keras.engine.topology import Layer
from keras import activations, regularizers, constraints
from keras import initializers
from keras.engine import InputSpec
import numpy as np
import tensorflow as tf


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
        
        
        from keras.initializers import constant
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
        MIN_SI = 1.0/self.input_dim  # this must be a function of the input size. aug 2020 not tested.
        MAX_SI = self.input_dim
        
        # create shared vars.
        self.MIN_SI = np.float32(MIN_SI)#, dtype='float32')
        self.MAX_SI = np.float32(MAX_SI)#, dtype='float32')
        
        w_init = initializers.get(self.kernel_initializer) if self.kernel_initializer else self.weight_initializer_fw_bg
        #w_init = initializers.glorot_uniform()
        #w_init = initializers.he_uniform()
        
        self.W = self.add_weight(shape=(self.input_dim, self.units),
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
        
        
    def call(self, inputs):
        u = self.calc_U()
        if self.verbose:
            print("weights shape", self.W.shape)
        self.kernel = self.W*u
        
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
            q = np.sqrt(W.shape[0])/np.max(kernel)
            W[mu[n], n] = np.random.choice([-q,q])
        
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(W[:,0:10])
        plt.show()
        input()
        
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
        # clipping scaler in range to prevent div by 0 or negative cov. 
        sigma = K.clip(self.sigma,self.MIN_SI,self.MAX_SI)
        dwn = K.expand_dims(2 * ( sigma ** 2), axis=1)
        result = K.exp(-up / dwn)
    
        # normalizations. self.normed==0 leaves Gauss as it is, max 1.0, min 0.0
        if self.normed==1:
            # nake the squared norm ==1
            result /= K.sqrt(K.sum(K.square(result), axis=-1,keepdims=True))
        
        elif self.normed==2:
            # make the squared norm = sqrt(n)
            result /= K.sqrt(K.sum(K.square(result), axis=-1,keepdims=True))
            result *= K.sqrt(K.constant(self.input_dim))
            print("result.shape",result.shape)
            
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
    from keras.models import  Model
    from keras.layers import Input, Dense, Dropout, Flatten, BatchNormalization, AlphaDropout
    from keras.layers import Activation # MaxPool2D
    act_func = 'relu'
    act_func = 'selu'
    act_func = 'relu'
    #drp_out = AlphaDropout
    drp_out = Dropout
    #from keras.regularizers import l2
    
    node_in = Input(shape=input_shape, name='inputlayer')

    node_fl = Flatten(data_format='channels_last')(node_in)
    
    node_ = Dropout(0.2)(node_fl) # changed for lfw_faces
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
                                   gain=1.0)(node_)
        else:
            node_ = Dense(nh,name='dense-'+str(h),activation='linear',
                          kernel_initializer=heu())(node_)
        
        node_ = BatchNormalization()(node_)
        node_ = Activation(act_func)(node_)
        node_ = drp_out(0.25)(node_) # changed for lfw_face, orig 0.25
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
        node_ = Dropout(0.25)(node_)
        h = h + 1
    
    node_fin = Dense(num_classes, name='softmax', activation='softmax', 
                     kernel_initializer=initializers.he_uniform(),
                    kernel_regularizer=None)(node_)
    
    model = Model(inputs=node_in, outputs=[node_fin])
    
    return model

    

def create_cnn_model(input_shape,  num_classes=10, settings={}):
    from keras.models import  Model
    from keras.layers import Input, Dense, Dropout, Flatten,Conv2D, BatchNormalization
    from keras.layers import Activation, MaxPool2D
    
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
    import keras
    #from keras.optimizers import SGD
    from keras.datasets import mnist,fashion_mnist, cifar10    
    #from skimage import filters
    from keras import backend as K
    #from keras_utils import WeightHistory as WeightHistory
    from keras_utils import RecordVariable, RecordOutput, \
    PrintLayerVariableStats, SGDwithLR, eval_Kdict, standarize_image_025,\
    standarize_image_01, AdamwithClip
    from keras_data import load_dataset
    
    from keras_preprocessing.image import ImageDataGenerator

    K.clear_session()
    
    epochs = settings['epochs']
    batch_size = settings['batch_size']

    sid = random_sid  
    np.random.seed(sid)
    tf.random.set_random_seed(sid)
    tf.compat.v1.random.set_random_seed(sid)
    
    # MINIMUM SIGMA CAN EFFECT THE PERFORMANCE.
    # BECAUSE NEURON CAN GET SHRINK TOO MUCH IN INITIAL EPOCHS WITH LARGER GRADIENTS
    #, and GET STUCK!
    MIN_SIG = 0.01
    MAX_SIG = 1.0
    MIN_MU = 0.0
    MAX_MU = 1.0
    
    GRADCLIP = 1.0   # set 1.0 for mnist 
    lr_dict = {'all':0.1, 'Mu':0.01, 'Sigma':0.01}
    mom_dict = {'all':0.9}
    decay_dict ={'all':0.9, 'Sigma':0.5}
    clip_dict = {'Sigma':(MIN_SIG,MAX_SIG), 'Mu':(MIN_MU,MAX_MU)}

    # update learning rates with multiplier set, e.g. 0.5 for CNN CIFAR
    for o in lr_dict.keys():
        lr_dict[o] *= settings['lr_mul']# lr_all is multiplier for all  

        
    print("Loading dataset")
    if settings['dset']=='mnist':
        # input image dimensions 
        img_rows, img_cols = 28, 28  
        # the data, split between train and test sets
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        n_channels=1
           
        e_i = x_train.shape[0] // batch_size
        
        GRADCLIP = 1.0  # set 1.0 for mnist 
        
        if settings['cnn_model']:
                   
           decay_epochs =[e_i*30,e_i*100]
        else:
            decay_dict = {'all':0.5} # best 0.5 for  99.29 with mnist
            decay_epochs =np.array([e_i*100, e_i*150], dtype='int64')
    
    elif settings['dset']=='cifar10':
        img_rows, img_cols = 32,32
        n_channels=3
        
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        
        e_i = x_train.shape[0] // batch_size
        decay_epochs =np.array([e_i*30,e_i*80,e_i*120,e_i*180], dtype='int64')
        
        if  settings['cnn_model']:
            # use sigma_reg=1e-10 for 79+ aug 2020, lr_mul=0.5
            print("using default lr with lr_mul")
        
    elif settings['dset']=='fashion':
        img_rows, img_cols = 28,28
        n_channels=1

        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
        #lr_dict['Sigma'] = 0.001
        e_i = x_train.shape[0] // batch_size
        decay_epochs =np.array([e_i*100, e_i*150], dtype='int64')
        if  settings['cnn_model']:
            decay_dict = {'all':0.9}
            # not sure about this.
            lr_dict = {'all':0.01, 'Sigma':0.1, 'Mu':0.1}
            
                     
    elif settings['dset']=='mnist-clut':
        
        img_rows, img_cols = 60, 60  
        # the data, split between train and test sets
        
        folder='/media/home/rdata/image/'
        try: 
            data = np.load(folder+"mnist_cluttered_60x60_6distortions.npz")
        except:
            folder='/home/btek/datasets/image/'
            data = np.load(folder+"mnist_cluttered_60x60_6distortions.npz")
        
        if not data:
            print("Unable to load mnist_cluttterd_60x60_6distortions.npz")
            
    
        x_train, y_train = data['x_train'], np.argmax(data['y_train'],axis=-1)
        x_valid, y_valid = data['x_valid'], np.argmax(data['y_valid'],axis=-1)
        x_test, y_test = data['x_test'], np.argmax(data['y_test'],axis=-1)
        x_train=np.vstack((x_train,x_valid))
        y_train=np.concatenate((y_train, y_valid))
        n_channels=1
        
        lr_dict = {'all':0.01}
        #lr_dict['mu']=0.001 #testing in laptop. 71.45 plain network
        
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
        X, y, test_size=0.25, random_state=sid) # some train/test sets are more
        # difficult than others. 
        
        print("Train dataset size:")
        print("n_samples: %d" % x_train.shape[0])
        print("n_features: %d" % x_train.shape[1:])
        print("n_classes: %d" % np.unique(y_train).shape[0])
        
        print("Test dataset size:")
        print("n_samples: %d" % x_test.shape[0])
        print("n_features: %d" % x_test.shape[1:])
        print("n_classes: %d" % np.unique(y_test).shape[0])
        #input('111')
        
        
        
        import matplotlib.pyplot as plt
        
        plt.imshow(X[0].reshape((img_rows,img_cols)))
        plt.show()
        
        #lr_dict['all']=0.01 # simple network best results for both focus and cnn+focus july 2020
        #if settings['cnn_model']:
        lr_dict = {'all':0.01}
        print(decay_dict)

        
        e_i = x_train.shape[0] // batch_size
        decay_epochs =np.array([e_i*50,e_i*100, e_i*150], dtype='int64')
        #decay_epochs =np.array([e_i*100], dtype='int64')

    
    
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
        
        #x_train, _, x_test = standarize_image_01(x_train, tst=x_test)
        x_train, _, x_test = standarize_image_025(x_train, tst=x_test)
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, n_channels)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, n_channels)
    
    input_shape = (img_rows, img_cols, n_channels)    
    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

#    from keras_data import load_dataset
##    dset = settings['dset']
#    normalize_data = True
#    if dset=='mnist-clut':
#        normalize_data=False
#
#    ld_data = load_dataset(dset,normalize_data,options=[])
#    x_train,y_train,x_test,y_test,input_shape,num_classes=ld_data
#    x_train,y_train,x_test,y_test,input_shape,num_classes=ld_data
    #print(num_classes)
    
    
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    
    sigma_reg = settings['focus_sigma_reg']
    sigma_reg = keras.regularizers.l2(sigma_reg) if sigma_reg is not None else sigma_reg
    settings['focus_sigma_reg'] = sigma_reg
    if settings['cnn_model']:
        model=create_cnn_model(input_shape,num_classes, settings=settings)
    else:
        model=create_simple_model(input_shape, num_classes, settings=settings)
    
 
    model.summary()
    
    print (lr_dict)
    print (mom_dict)
    print (decay_dict)
    print (clip_dict)
    
    opt= SGDwithLR(lr_dict, mom_dict, decay_dict, clip_dict, 
                   decay_epochs, clipvalue=GRADCLIP, verbose=1)#, decay=None)
    #opt = AdamwithClip(clips=clip_dict)
    #opt= SGDwithLR(lr_dict, mom_dict,decay_dict,clip_dict, 
    #                decay_epochs,update_clip=UPDATE_Clip)#, decay=None)
                   
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=opt,
                  metrics=['accuracy'])
    
    
        
    stat_func_name = ['max: ', 'mean: ', 'min: ', 'var: ', 'std: ']
    stat_func_list = [np.max, np.mean, np.min, np.var, np.std]
    #callbacks = [tb]
    callbacks = []
    
    if  settings['neuron']=='focused' and False:
        pr_1 = PrintLayerVariableStats("focus-1","Weights:0",stat_func_list,stat_func_name)
        pr_2 = PrintLayerVariableStats("focus-1","Sigma:0",stat_func_list,stat_func_name)
        pr_3 = PrintLayerVariableStats("focus-1","Mu:0",stat_func_list,stat_func_name)
        rv_weights_1 = RecordVariable("focus-1","Weights:0")
        rv_sigma_1 = RecordVariable("focus-1","Sigma:0")
        rv_mu_1 = RecordVariable("focus-1","Mu:0")
        #out_call = RecordOutput(model,["focus-1", "focus-2"],stat_func_list, stat_func_name)
        print_lr_rates_callback = keras.callbacks.LambdaCallback(
                on_epoch_end=lambda epoch, logs: print("iter: ", 
                                                       K.eval(model.optimizer.iterations),
                                                       " LR RATES :", 
                                                       eval_Kdict(model.optimizer.lr)))
    
        callbacks+=[pr_1,pr_2,pr_3,rv_weights_1,rv_sigma_1, rv_mu_1,
                    print_lr_rates_callback]
    
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
    
    filename = 'outputs/Kfocusing/'+settings['dset']+'/'+timestr+'_'+settings['neuron']+'.model_results.npz'
    copyfile("Kfocusing.py",filename+"code.py")
   
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
                last_mean_sigma = 0
                if len(cb)>4:
                    last_mean_sigma = np.mean(cb[4].record[-1])
                list_sigmas.append([sigma_reg, last_mean_sigma])
            
    print("Final scores", list_scores)
    mx_scores = [np.max(list_histories[i].history['val_acc']) for i in range(len(list_histories))]
    histories = [m.history.history for m in models]
    print("Max sscores", mx_scores)
    np.savez_compressed(filename,mx_scores =mx_scores, list_scores=list_scores, 
                        modelz=histories, sigmas=list_sigmas)
    return mx_scores, list_scores, histories, list_sigmas


   
    
if __name__ == "__main__":
    import os
    if 'CUDA_VISIBLE_DEVICES' not in os.environ.keys():
        os.environ['CUDA_VISIBLE_DEVICES']="0"
        os.environ['TF_FORCE_GPU_ALLOW_GROWTH']="true"
    
    import sys
    import time 
    from shutil import copyfile
    
    #dset='mnist' #
    #dset='cifar10'  # ~64,5 cifar is better with batch 256, init_sigma =0.01 
    #dset='mnist'
    #dset = 'mnist-clut'
    dset = 'fashion'
    #dset='lfw_faces' # simple network  ~max 77, lr 0.01, dec=0.9 (si0.5), use batch_size = 32, augment=True, init_sigm=0.025, init_mu=spread july 2020 confirmed
    #dset = 'lfw_faces' 
    sigma_reg_set = None
    
    # I replaced direct lr_all with lr_mul. which is multiplier. aug 2020
    # For plain network: lr_mul 1.0 MNIST, CIFAR, fashion, mnist clut. faces overwrite all learning rates =0.01
    # For cnn: lr_mul 1.0 for nhidden 256, batch512 nfilters 32,32 MNIST, gets ~99.63
    # I also use GRAD_CLIP 1.0 in code
    kwargs = {'dset':'mnist', 'neuron':'focused', 'nhidden':(784,784), 
              'cnn_model':False, 'nfilters':(32,32), 'repeats':5, 'epochs':200,
              'batch_size':512,'lr_mul':1.0, 'augment':False, 'delay':0,
              'kn_size':(5,5), 'focus_init_sigma':0.025,
              'focus_init_mu':'spread', 'focus_train_mu':True, 'focus_train_si':True,
              'focus_train_weights':True, 'focus_norm_type':2, 
              'focus_sigma_reg':[1e-10], 'ex_name':''}
    
    for s in sys.argv:
        print(":",s)
    
    # this is to read args in the form '(x,t,y)'
    def argtolist(s):
        s1 = s.replace('(','').replace(')','').split(',')
        r = [int(k) for k in s1 if k is not '']
        return r
        
   
    i = 1
    if len(sys.argv) > i:
            kwargs['dset'] = sys.argv[i]
            i+=1
    if len(sys.argv) > i:
        kwargs['neuron'] = sys.argv[2]
        i+=1
    if len(sys.argv) > i:
        kwargs['cnn_model'] = sys.argv[i]=="True"
        i+=1
    if len(sys.argv) > i:
        kwargs['nhidden'] = argtolist(sys.argv[i])
        i+=1
    if len(sys.argv) > i:
        kwargs['nfilters'] = argtolist(sys.argv[i])
        i+=1
        
    if len(sys.argv) > i:
        kwargs['repeats'] = int(sys.argv[i])
        i+=1
    
    if len(sys.argv) > i:
        kwargs['epochs'] = int(sys.argv[i])
        i+=1
        
    if len(sys.argv) > i:
        kwargs['batch_size'] = int(sys.argv[i])
        i+=1
             
    if len(sys.argv) > i:
        kwargs['lr_mul'] = float(sys.argv[i])
        i+=1
        
    if len(sys.argv) > i:
        kwargs['augment'] = sys.argv[i]=="True"
        i+=1
        
    if len(sys.argv) > i:
        kwargs['delay'] =  int(sys.argv[i])
        i+=1
        
    if len(sys.argv) > i:
        kwargs['kn_size'] = argtolist(sys.argv[i])
        i+=1
    if len(sys.argv) > i:
        kwargs['focus_init_sigma'] = float(sys.argv[i])
        i+=1
    if len(sys.argv) > i:
        kwargs['focus_init_mu'] = sys.argv[i]
        i+=1
    if len(sys.argv) > i:
        kwargs['focus_train_mu'] = sys.argv[i]=="True"
        i+=1
    if len(sys.argv) > i:
        kwargs['focus_train_si'] = sys.argv[i]=="True"
        i+=1
    if len(sys.argv) > i:
        kwargs['focus_norm_type'] = int(sys.argv[i])
        i+=1
    if len(sys.argv) > i:
        if sys.argv[i]== 'None':
            kwargs['focus_sigma_reg']=None
            sigma_reg_set  = None
        else:
            kwargs['focus_sigma_reg'] = [float(sys.argv[i])]
            sigma_reg_set = [kwargs['focus_sigma_reg']]
        i+=1
        
    if len(sys.argv) > i:
        kwargs['ex_name'] = sys.argv[i]
        i+=1
        
    if len(sys.argv) > i:
        kwargs['ex_name'] = sys.argv[i]
        i+=1
    tim = time.localtime()
    tim = str(tim[1])+'_'+str(tim[1])+'_'+str(tim[2])+str(tim[3])
    fname=kwargs['dset']+'_'+kwargs['neuron']+'_'+tim+'.npz'
    #test()
    delayed_start = kwargs['delay']

    print("Delayed start ",delayed_start)
    time.sleep(delayed_start)
    
    print(kwargs)

    f = test_comp
    res = repeated_trials(test_function=f,settings=kwargs)
    
    resdir = 'outputs/'+kwargs['ex_name']
    if not os.path.isdir(resdir):
        os.mkdir(resdir)
        
    np.savez_compressed(resdir+'/'+fname, res, kwargs)
    import matplotlib.pyplot as plt
    if sigma_reg_set:
        
        plt.plot(np.log10(sigma_reg_set),np.reshape(res[0],(-1,5)),'o')
        plt.errorbar(x=np.log10(sigma_reg_set),y=np.mean(np.reshape(res[0],(-1,5)),axis=1),
                     yerr=np.std(np.reshape(res[0],(-1,5)),axis=1))
        
    else:
        plt.plot(np.reshape(res[0],(-1,kwargs['repeats'])),'o')
    
    # USE SCRIPTS FOR PARAMS. 
    # run_mnist....
    # run_fashion_cnn_focus.sh
    # SOME RESULTS july-aug 2020
    # focused MNIST Augmented accuracy (200epochs): ~99.25-99.30
    # focused MNIST No Augment accuracy(200 epochs): ~99.25
    # CNN results MNIST  max : 99.63 focus, 99.63 dense
    # focused mx_1 = [0.9958999999046325, 0.9958999998092651, 0.9959999999046326, 0.9960999999046326, 0.9960999999046326]
    # dense mx_2=[Max sscores [0.9958999999046325, 0.9958999999046325, 0.9962999998092651, 0.9959999999046326, 0.9954999998092652]]
    # july 2020 with default lrs lr_mul = 1.0
    # [0.9960999999046326, 0.9961999999046326, 0.9962999999046326, 0.9963999999046326, 0.9958999999046325]
    
    
    # fASHION LR_mUl 1.0, lr-all 0.1, mu,si 0.01
    # [0.9124999999046326, 0.9105999994277955, 0.9114999995231629, 0.9122999999046326, 0.9101999999046325]
    
    # FASHION CNN Lr_mul=1.0 jul 2020 
    # [0.9367000004768372, 0.939300000667572, 0.9373000004768371, 0.9375000004768371, 0.9370000005722046]
    
    
    # produce MNIST 99.27
    #kwargs = {'dset':'mnist', 'neuron':'focused', 'nhidden':(784,784), 
    #          'cnn_model':False, 'nfilters':(32,32), 'repeats':5, 'epochs':200,
    #          'batch_size':512,'lr_mul':1.0, 'augment':False, 'delay':0,
    #          'kn_size':(5,5), 'focus_init_sigma':0.025,
    #          'focus_init_mu':'spread', 'focus_train_mu':True, 'focus_train_si':True,
    #          'focus_train_weights':True, 'focus_norm_type':2, 
    #          'focus_sigma_reg':[1e-10], 'ex_name':''}