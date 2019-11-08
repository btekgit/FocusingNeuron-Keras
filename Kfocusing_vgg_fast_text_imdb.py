#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import print_function
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES']="0"
os.environ['TF_FORCE_GPU_ALLOW_GROWTH']="true"

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten,Activation
from keras.layers import Embedding,Input,BatchNormalization, Permute
from keras.layers import GlobalAveragePooling1D
from keras.datasets import imdb
import tensorflow as tf
import keras
import keras.backend as K

from keras_utils import WeightHistory as WeightHistory
from keras_utils import RecordVariable, PrintLayerVariableStats, PrintAnyVariable, \
    SGDwithLR, AdamwithClip, eval_Kdict, standarize_image_025
from keras_preprocessing.image import ImageDataGenerator
from Kfocusing import FocusedLayer1D
from keras.engine.topology import Layer
from keras.engine.base_layer import InputSpec

class _GlobalPooling1D(Layer):
    """Abstract class for different global pooling 1D layers.
    """

    def __init__(self, data_format='channels_last', **kwargs):
        super(_GlobalPooling1D, self).__init__(**kwargs)
        self.input_spec = InputSpec(ndim=3)
        self.data_format = K.normalize_data_format(data_format)

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_first':
            return (input_shape[0], input_shape[1])
        else:
            return (input_shape[0], input_shape[2]*2)

    def call(self, inputs):
        raise NotImplementedError

    def get_config(self):
        config = {'data_format': self.data_format}
        base_config = super(_GlobalPooling1D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
class GlobalSTDPooling1D(_GlobalPooling1D):
    """Global average pooling operation for temporal data.

    # Arguments
        data_format: A string,
            one of `channels_last` (default) or `channels_first`.
            The ordering of the dimensions in the inputs.
            `channels_last` corresponds to inputs with shape
            `(batch, steps, features)` while `channels_first`
            corresponds to inputs with shape
            `(batch, features, steps)`.

    # Input shape
        - If `data_format='channels_last'`:
            3D tensor with shape:
            `(batch_size, steps, features)`
        - If `data_format='channels_first'`:
            3D tensor with shape:
            `(batch_size, features, steps)`

    # Output shape
        2D tensor with shape:
        `(batch_size, features)`
    """

    def __init__(self, data_format='channels_last', **kwargs):
        super(GlobalSTDPooling1D, self).__init__(data_format,
                                                     **kwargs)
        self.supports_masking = True

    def call(self, inputs, mask=None):
        steps_axis = 1 if self.data_format == 'channels_last' else 2
        if mask is not None:
            mask = K.cast(mask, K.floatx())
            input_shape = K.int_shape(inputs)
            broadcast_shape = [-1, input_shape[steps_axis], 1]
            mask = K.reshape(mask, broadcast_shape)
            inputs *= mask
            return K.sum(inputs, axis=steps_axis) / K.sum(mask, axis=steps_axis)
        else:
            return K.concatenate((K.mean(inputs, axis=steps_axis),
                                  K.std(inputs, axis=steps_axis)))

    def compute_mask(self, inputs, mask=None):
        return None
    
def create_ngram_set(input_list, ngram_value=2):
    """
    Extract a set of n-grams from a list of integers.

    >>> create_ngram_set([1, 4, 9, 4, 1, 4], ngram_value=2)
    {(4, 9), (4, 1), (1, 4), (9, 4)}

    >>> create_ngram_set([1, 4, 9, 4, 1, 4], ngram_value=3)
    [(1, 4, 9), (4, 9, 4), (9, 4, 1), (4, 1, 4)]
    """
    return set(zip(*[input_list[i:] for i in range(ngram_value)]))


def add_ngram(sequences, token_indice, ngram_range=2):
    """
    Augment the input list of list (sequences) by appending n-grams values.

    Example: adding bi-gram
    >>> sequences = [[1, 3, 4, 5], [1, 3, 7, 9, 2]]
    >>> token_indice = {(1, 3): 1337, (9, 2): 42, (4, 5): 2017}
    >>> add_ngram(sequences, token_indice, ngram_range=2)
    [[1, 3, 4, 5, 1337, 2017], [1, 3, 7, 9, 2, 1337, 42]]

    Example: adding tri-gram
    >>> sequences = [[1, 3, 4, 5], [1, 3, 7, 9, 2]]
    >>> token_indice = {(1, 3): 1337, (9, 2): 42, (4, 5): 2017, (7, 9, 2): 2018}
    >>> add_ngram(sequences, token_indice, ngram_range=3)
    [[1, 3, 4, 5, 1337, 2017], [1, 3, 7, 9, 2, 1337, 42, 2018]]
    """
    new_sequences = []
    for input_list in sequences:
        new_list = input_list[:]
        for ngram_value in range(2, ngram_range + 1):
            for i in range(len(new_list) - ngram_value + 1):
                ngram = tuple(new_list[i:i + ngram_value])
                if ngram in token_indice:
                    new_list.append(token_indice[ngram])
        new_sequences.append(new_list)

    return new_sequences

# Set parameters:
# ngram_range = 2 will add bi-grams features
ngram_range = 1
max_features = 20000
maxlen = 400
batch_size = 32
embedding_dims = 50
epochs = 7

print('Loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')
print('Average train sequence length: {}'.format(
    np.mean(list(map(len, x_train)), dtype=int)))
print('Average test sequence length: {}'.format(
    np.mean(list(map(len, x_test)), dtype=int)))

if ngram_range > 1:
    print('Adding {}-gram features'.format(ngram_range))
    # Create set of unique n-gram from the training set.
    ngram_set = set()
    for input_list in x_train:
        for i in range(2, ngram_range + 1):
            set_of_ngram = create_ngram_set(input_list, ngram_value=i)
            ngram_set.update(set_of_ngram)

    # Dictionary mapping n-gram token to a unique integer.
    # Integer values are greater than max_features in order
    # to avoid collision with existing features.
    start_index = max_features + 1
    token_indice = {v: k + start_index for k, v in enumerate(ngram_set)}
    indice_token = {token_indice[k]: k for k in token_indice}

    # max_features is the highest integer that could be found in the dataset.
    max_features = np.max(list(indice_token.keys())) + 1

    # Augmenting x_train and x_test with n-grams features
    x_train = add_ngram(x_train, token_indice, ngram_range)
    x_test = add_ngram(x_test, token_indice, ngram_range)
    print('Average train sequence length: {}'.format(
        np.mean(list(map(len, x_train)), dtype=int)))
    print('Average test sequence length: {}'.format(
        np.mean(list(map(len, x_test)), dtype=int)))

print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

   # Create a session with the above options specified.
#.tensorflow_backend.set_session(tf.Session(config=config))


mods=['focused', 'dense']
print('Build model...')
Repeats = 5
results_ = [[],[]]
for j, m in enumerate(mods):
    for r in range(Repeats):
        K.clear_session()
        model = Sequential()
        #model.add(BatchNormalization())
        model.add(Embedding(max_features,
                             embedding_dims,
                             input_length=maxlen))
        
        ### REPLACE GLOBAL AVERAGE 
# =============================================================================
#         tranzpose = True
#         if tranzpose:
#             model.add(Permute((2, 1)))
#         nu = 5
#         model.add(FocusedLayer1D(units=nu,
#                                name='focus-1',
#                                activation='linear',
#                                init_sigma=0.5,
#                                init_mu=np.linspace(0.4, 0.6, nu),
#                                kernel_initializer= keras.initializers.constant(1.0),
#                                kernel_regularizer = keras.regularizers.l2(1e-8),
#                                train_sigma=True,
#                                train_weights=False,
#                                train_mu = True,
#                                si_regularizer=keras.regularizers.l2(1e-8),
#                                normed=1))
# =============================================================================
        #model.add(BatchNormalization())
        #model.add(Input(shape=x_train.shape[1:]))
        #model.add(BatchNormalization(input_shape=x_train.shape[1:]))
        
        # we start off with an efficient embedding layer which maps
        # our vocab indices into embedding_dims dimensions
# =============================================================================
#        model.add(Embedding(max_features,
#                            embedding_dims,
#                            input_length=maxlen))
#         
#         # we add a GlobalAveragePooling1D, which will average the embeddings
#         # of all words in the document
#         #model.add(Flatten())
        model.add(GlobalSTDPooling1D())
# =============================================================================
        #model.add(Dropout(0.25))
        
#        nu = 10
#        model.add(FocusedLayer1D(units=nu,
#                                name='focus-1',
#                                activation='linear',
#                                init_sigma=0.5,
#                                init_mu='spread',
#                                train_sigma=True,
#                                train_weights=False,
#                                train_mu = True,
#                                si_regularizer=keras.regularizers.l2(1e-8),
#                                normed=2))
#        model.add(BatchNormalization())
#        model.add(Activation('relu'))
#        
#        model.add(Dropout(0.5))
        pad_input =False
        if pad_input:
            print("PADDING LAYER OUPUT")
            
            paddings = tf.constant([[0, 0,], [3, 3]])
        
            padding_layer = keras.layers.Lambda(lambda x: tf.pad(x,paddings,"CONSTANT"))
            model.add(padding_layer)
   
        
        #model.add(Flatten())
        if m=='focused':
            nf = 1#init_sigma=np.exp(-(np.linspace(0.1, 0.9, nf)-0.5)**2/0.07),
            model.add(FocusedLayer1D(units=nf,
                               name='focus-1',
                               activation='sigmoid',
                               init_sigma=1.0,
                               init_mu=0.5,
                               kernel_regularizer = None,
                               train_sigma=True,
                               train_weights=True,
                               train_mu = True,
                               si_regularizer=None,
                               normed=2))
        elif m=='dense':
            model.add(Dense(1, input_shape=x_train.shape[1:], activation='sigmoid'))
        else:
            print('unknown mod')
            
        # We project onto a single unit output layer, and squash it with a sigmoid:
        
        #model.add(Dense(1, activation='sigmoid'))
        model.summary()
        MIN_SIG = np.float32(0.01)
        MAX_SIG = np.float32(5.0)
        MIN_MU = np.float32(-0.1)
        MAX_MU =np.float32(1.1)
        clip_dict={'focus-1/Sigma:0':(MIN_SIG,MAX_SIG),
                    'focus-1/Mu:0':(MIN_MU,MAX_MU),
                    'focus-2/Sigma:0':(MIN_SIG,MAX_SIG),
                    'focus-2/Mu:0':(MIN_MU,MAX_MU)}
        opt = AdamwithClip(lr=0.001, clips=clip_dict)
        
        model.compile(loss='binary_crossentropy',
                      optimizer=opt,
                      metrics=['accuracy'])
        
        stat_func_name = ['max: ', 'mean: ', 'min: ', 'var: ', 'std: ']
        stat_func_list = [np.max, np.mean, np.min, np.var, np.std]
        #callbacks = [tb]
        callbacks = []
        if m=='focused':
            pr_1 = PrintLayerVariableStats("focus-1","Weights:0",stat_func_list,stat_func_name)
            pr_2 = PrintLayerVariableStats("focus-1","Sigma:0",stat_func_list,stat_func_name)
            pr_3 = PrintLayerVariableStats("focus-1","Mu:0",stat_func_list,stat_func_name)
            callbacks+=(pr_1,pr_2,pr_3)
            
        #x_train = np.float32(x_train) /np.float32(np.max(x_train))
        #x_train -= 0.5
        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_data=(x_test, y_test), callbacks=callbacks)
        score = model.evaluate(x_test, y_test, verbose=0)
        
        results_[j].append(score[1])
        
print("Focus: mean:",np.mean(results_[0]),np.std(results_[0]), "  max: ", np.max(results_[0]))
print("Dense: mean:",np.mean(results_[1]),np.std(results_[1]), "  max: ", np.max(results_[1]))
from scipy.stats import ttest_ind
print(ttest_ind(results_[1],results_[0]))