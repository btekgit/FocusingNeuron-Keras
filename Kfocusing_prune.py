#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 15:36:24 2019

@author: btek
# this code is written to demonstrate pruning of weights by using the focus function
# After the network is trained. (it trains from scratch)
# It calculates focus functions for each focusing layer. 
# In a loop which increments in-focus threshold, it zeros out-of-focus weights
# and sets them to zero. Then it computes sparsity and the evaluation score of
# the network. 
"""

import os

os.environ['CUDA_VISIBLE_DEVICES']="0"
os.environ['TF_FORCE_GPU_ALLOW_GROWTH']="true"

print('ADD early_stopping_cb = keras.callbacks.EarlyStopping(patience=10,restore_best_weights=True)')
from Kfocusing import test_comp, calculate_fi_and_weights, U_numeric, prune_out_of_focus_weights
import keras.backend as K
from keras.datasets import mnist, cifar10
from data_utils import standarize_image_025
import keras_utils
import matplotlib.pyplot as plt

import keras
import numpy as np

dset='mnist'
#dset='cifar10'
#mod='focused'
#mod='focused'
cnn_model = True
nhidden=[256]
#nhidden=[784,784]
# fnn use nhidden=[784,784], 
#nhidden=[32*32*3,32*32*3]
Epochs = 5
# using pre_trained_model does not work properly do to numpy.save
use_pre_trained_model=False 
save_records=False
if not use_pre_trained_model:

    from datetime import datetime
    now = datetime.now()
    timestr = now.strftime("%Y%m%d-%H%M%S")
    
    mod={'dset':dset, 'neuron':'focused', 'nhidden':nhidden, 'cnn_model':cnn_model,
         'nfilters':(32,32), 'kn_size':(5,5),
         'focus_init_sigma':0.025, 'focus_init_mu':'spread','focus_train_mu':True, 
         'focus_train_si':True,'focus_train_weights':True,'focus_norm_type':2,
         'focus_sigma_reg':None,'augment':False, 
         'Epochs':Epochs, 'batch_size':512,'repeats':1,
         'lr_all':0.1}

    sc, hs, model, cb = test_comp(mod,random_sid=0)
    callback_records =[]
    for i in range(len(cb)):
        if isinstance(cb[i],keras_utils.RecordVariable):
            callback_records.append(cb[i].record)
            then = datetime.now()
            timestr += then.strftime("%Y%m%d-%H%M%S")
            filename = 'outputs/Kfocusing/'+dset+'/'+timestr+'_'+mod['neuron']+'.trained_model.npz'
            import keras.utils.generic_utils as gen_util    
            from Kfocusing import FocusedLayer1D
    
            with gen_util.custom_object_scope({'FocusedLayer1D':FocusedLayer1D}):
                np.savez_compressed(filename,scores=sc, history=hs,  model=model)
            
            
    if mod=='focused' and save_records:
    #root_to_save ='/home/btek/Dropbox/code/pythoncode/FocusingNeuron/outputs/Kfocusing/vids/'
        file_to_save = filename = 'outputs/Kfocusing/'+dset+'/'+timestr+'_'+mod+'.trained_model_records.npz'
        np.savez_compressed(file_to_save, callbacks=callback_records)

else:
    # cifar-10 focused CNN 256 nhidden, 200 epochs
    pretrained=np.load('outputs/Kfocusing/'+dset+'/20191005-15455320191005-15465520191005-15471020191005-154726_focused.trained_model.npz')
    import keras.utils.generic_utils as gen_util
    from Kfocusing import FocusedLayer1D
    from keras_utils import SGDwithLR
    
    with gen_util.custom_object_scope({'FocusedLayer1D':FocusedLayer1D,
                                       'SGDwithLR':SGDwithLR}):

        sc = pretrained['scores']
        #hs = pretrained['history']
        model = pretrained['model']

print("Loading dataset")
if dset=='mnist':
    # input image dimensions
    img_rows, img_cols = 28, 28  
    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    n_channels=1
    
    class_names = [str(i) for i in range(10)]
    
    
elif dset=='cifar10':    
    img_rows, img_cols = 32,32
    n_channels=3
    
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    class_names = ['airplane','automobile', 
                   'bird','cat','deer','dog','frog','horse','ship','truck' ]
    

num_classes = np.unique(y_train).shape[0]

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], n_channels, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], n_channels, img_rows, img_cols)
    input_shape = (n_channels, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, n_channels)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, n_channels)
    input_shape = (img_rows, img_cols, n_channels)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train, _, x_test = standarize_image_025(x_train, tst=x_test)
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, n_channels)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, n_channels)
input_shape = (img_rows, img_cols, n_channels)

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])



print("Pruning")

# img_rows,img_cols,n_channels
in_shape = img_rows*img_cols

#import matplotlib.pyplot as plt
import plot_utils as pu
pu.paper_fig_settings(addtosize=2)

if cnn_model:
    in_rows, in_cols, in_channels = 16,16,32
    #in_rows, in_cols, in_channels = 32,32,32
else:
    in_rows, in_cols, in_channels = img_rows,img_cols,n_channels


layer_names = ['focus-1', 'focus-2']
foc_1 = model.get_layer('focus-1')
backup_weights_1 = foc_1.get_weights()


#%%
count_all_weights_layer_1 =0
count_non_zero_weights_layer_1=[]
hist_non_zero_weights_layer_1 =[]

foc_2 = None
if len(nhidden)>1:
    foc_2 = model.get_layer('focus-2')
    backup_weights_2 = foc_2.get_weights()
    count_all_weights_layer_2 =0
    count_non_zero_weights_layer_2=[]
    hist_non_zero_weights_layer_2 =[]
    
 
focus_thresholds = np.linspace(1e-7,1.5, 40)
focus_thresholds_0 = np.zeros(focus_thresholds.shape[0]+1)
focus_thresholds_0[1:] = focus_thresholds
focus_thresholds = focus_thresholds_0
scores = []
#hist_range = np.linspace(50,img_rows*img_cols)
hist_range = (50,img_rows*img_cols)
r_ix = np.random.permutation(nhidden[0])[0:20]
for fc_th in focus_thresholds:
  
    # prune layer-1 
    fi_p, in_f, fi_p_w = prune_out_of_focus_weights(foc_1, threshold=fc_th)
    count_all_weights_layer_1=np.prod(fi_p_w.shape)
    count_non_zero_weights_layer_1.append(np.sum(in_f))
    hist_non_zero_weights_layer_1.append(np.histogram(np.sum(in_f,axis=0),20,range=hist_range)[0])
    
 
    # prune layer-2
    if foc_2:
        fi_p, in_f, fi_p_w = prune_out_of_focus_weights(foc_2, threshold=fc_th)
        count_all_weights_layer_2=np.prod(fi_p_w.shape)
        count_non_zero_weights_layer_2.append(np.sum(in_f))
        hist_non_zero_weights_layer_2.append(np.histogram(np.sum(in_f,axis=0),20,range=hist_range)[0])
   
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    scores.append(score)
    
    # restore weights
    foc_1.set_weights(backup_weights_1)
    if foc_2:
        foc_2.set_weights(backup_weights_2)
    plt_results=False
    if plt_results:
        n_ix= [5,480,780]
        plt.figure()
        print('Plotting three candidates foci')
        plt.plot(fi_p[:,n_ix])
        plt.gca().set_prop_cycle(None)
        plt.plot(fi_p_w[:,n_ix],'--',alpha=0.7)
        plt.xlabel('Input index')
        plt.ylabel(r'Focused weights (& Focus)')
        plt.grid('on')
    
#%% plots
    
from plot_utils import save_fig, paper_fig_settings
paper_fig_settings(+4)
scores = np.array(scores)
# sparsity is the ratio of zeros to all weights
sparsity_1 = 1-np.array(count_non_zero_weights_layer_1)/count_all_weights_layer_1
plt.figure()
print('Plotting histogram of Sigma')
plt.plot(sparsity_1, scores[:,1])
plt.grid('on')
plt.xlabel('Sparsity in layer 1')
plt.ylabel('Test accuracy')

#scores = np.array(scores)
sparsity_2 = 1-np.array(count_non_zero_weights_layer_2)/count_all_weights_layer_2
plt.figure()
print('Plotting histogram of Sigma')
plt.plot(sparsity_2, scores[:,1])
#plt.semilogy(sparsity, scores[:,1])
plt.grid('on')
plt.xlabel('Sparsity in layer 2')
plt.ylabel('Test accuracy')


sparsity_network = 1.-(np.array(count_non_zero_weights_layer_1)+np.array(count_non_zero_weights_layer_2))/(count_all_weights_layer_1+count_all_weights_layer_2)

fig=plt.figure()
ax1 = fig.add_subplot(111)
ax2 = ax1.twiny()
ax1.plot(sparsity_network, scores[:,1],'-')
#plt.semilogy(sparsity, scores[:,1])
ax1.grid('on')
ax1.set_xlabel('Sparsity')
ax1.set_ylabel('Test accuracy')
ax1.set_xticks([0.0,0.3,0.5,0.7,0.8])
mx_scores= np.max(scores[:,1])
ix_mx = np.argmax(scores[:,1][::-1]) # from right to left, choose max th
th_mx = focus_thresholds[::-1][ix_mx]
sp_mx = sparsity_network[::-1][ix_mx]
ax1.plot(sp_mx, mx_scores,'^', markersize=12,alpha=0.8)
ax1.text(sp_mx+0.01,mx_scores-0.0005,str(mx_scores),fontdict={'fontsize':'16'})
last_better_dense = np.argmax(scores[:,1]<=0.9902)-1
ax1.plot(sparsity_network[last_better_dense],
         scores[last_better_dense,1],'d',markersize=12,alpha=0.8)
ax1.text(sparsity_network[last_better_dense]-0.18,
         scores[last_better_dense,1]-0.002,str('Dense mean'),fontdict={'fontsize':'16'})
ax2.set_xlim(ax1.get_xlim())
ax2.set_xticks(focus_thresholds[::12])
#ax2.set_xticklabels(tick_function(focus_thresholds))
ax2.set_xlabel(r"Focus threshold: t ")
plt.show()
print('Plotting histogram of Sigma')
print("Network")
print("Mx score: ", mx_scores, " at threshold ", th_mx, " provides sparsity", sp_mx)


fig = plt.figure()
ax1 = fig.add_subplot(111)
t = 1
bins = np.histogram(np.sum(in_f,axis=0),20,range=hist_range)[1]
print('Plotting non-zero weights')
ax1.bar(bins[:-1],hist_non_zero_weights_layer_1[t],width=14.0,alpha=0.5)
ax1.bar(bins[:-1],hist_non_zero_weights_layer_2[t],width=14.0,alpha=0.4)
ax1.bar(bins[:-1],hist_non_zero_weights_layer_1[27],width=14.0,alpha=0.7)
ax1.bar(bins[:-1],hist_non_zero_weights_layer_2[27],width=14.0,alpha=0.7)
ax1.grid('on')
ax1.set_xlabel('# Nonzero weights')
ax1.set_ylabel('# Neurons')
ax1.legend([r'layer-1 base','layer-2 base','layer-1 max accuracy','layer-2 max accuracy'])
print("Threshold:", focus_thresholds[t], focus_thresholds[27])
plt.show()
plt.figure()
#for n in n_ix:
#    plt.imshow(fi_1p_2d[n,:,:,0],alpha=0.3, cmap='jet')
#ax1.set_grid('on')
#plt.show()    