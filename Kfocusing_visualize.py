#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 15:36:24 2019

@author: btek
# Plots focus functions, weights, focused-weights, 2D projections. 
"""

import os



os.environ['CUDA_VISIBLE_DEVICES']="0"
os.environ['TF_FORCE_GPU_ALLOW_GROWTH']="true"


from Kfocusing import test_comp, calculate_fi_and_weights, U_numeric
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
mod='focused'
cnn_model = False
#nhidden=[256]
nhidden=[784,784]
# fnn use nhidden=[784,784], 
#nhidden=[32*32*3,32*32*3]
Epochs =20
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
         'Epochs':200, 'batch_size':256,'repeats':1,
         'lr_all':0.1}
    sc, hs, model, cb = test_comp(mod,random_sid=9)
    callback_records =[]
    for i in range(len(cb)):
        if isinstance(cb[i],keras_utils.RecordVariable):
            callback_records.append(cb[i].record)
            then = datetime.now()
            timestr += then.strftime("%Y%m%d-%H%M%S")
            filename = 'outputs/Kfocusing/'+dset+'/'+timestr+'_'+mod+'.trained_model.npz'
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



print("Visualizing layer weights")

# img_rows,img_cols,n_channels
in_shape = img_rows*img_cols

import matplotlib.pyplot as plt
import plot_utils as pu
pu.paper_fig_settings(addtosize=2)

if cnn_model:
    in_rows, in_cols, in_channels = 16,16,32
    #in_rows, in_cols, in_channels = 32,32,32
else:
    in_rows, in_cols, in_channels = img_rows,img_cols,n_channels

if mod=='focused':
    layer_names = ['focus-1', 'focus-2']
    foc_1 = model.get_layer('focus-1')
    if len(nhidden)>1:
        foc_2 = model.get_layer('focus-2')



#from matplotlib import rc
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
#rc('text', usetex=True)
    
    fi_1, we, fi_1_w = calculate_fi_and_weights(foc_1)
    fi_1_2d = np.reshape(fi_1.T, (fi_1.T.shape[0],in_rows,in_cols,in_channels))
    fi_1_w_2d = np.reshape(fi_1_w.T, (fi_1_w.T.shape[0],in_rows,in_cols,in_channels))


    n_ix = [8, nhidden[0]//2, nhidden[0]-10]
    all_mus = foc_1.get_weights()[0]
    sel_mus = foc_1.get_weights()[0][n_ix]
    mus_x = np.int32((sel_mus*in_cols*in_rows) % in_cols)
    mus_y = np.int32((sel_mus*in_cols*in_rows) // in_rows)

    all_sis = foc_1.get_weights()[1]
    sel_sis = np.int32(foc_1.get_weights()[1][n_ix]*100)/100.0

    plt.figure()
    print('Plotting three candidates foci')
    plt.plot(fi_1_w[:,n_ix])
    plt.gca().set_prop_cycle(None)
    plt.plot(fi_1[:,n_ix],'--',alpha=0.7)
    plt.xlabel('Input index')
    plt.ylabel(r'Focused weights (& Focus)')
    plt.grid('on')
    
    plt.figure()
    plt.plot(fi_1,alpha=0.5)
    plt.grid('on')
    plt.xlabel('index')
    plt.ylabel('Focus Magnititude')
    plt.show()
    
    plt.figure()
    plt.plot(fi_1_w, alpha=0.5)
    plt.grid('on')
    plt.xlabel('index')
    plt.ylabel('Focus*W Magnititude')
    plt.show()


    pltfoc2d = np.zeros(shape=(fi_1_w_2d[0].shape[0],fi_1_w_2d[0].shape[1]))
    pltfoc2d[:,:] = np.sum(fi_1_2d[n_ix[0]],axis=2)
    plt.figure()
    print('Plotting three projections foci')
    plt.imshow(pltfoc2d)
    plt.colorbar()
    
    
    plt.figure()
    print('Plotting histogram of Sigma')
    plt.hist(all_sis)
    plt.grid('on')
    plt.xlabel('Focus Sigma')
    plt.ylabel('Count')
    
    plt.figure()
    print('Plotting histogram of Mus')
    plt.hist(all_mus)
    plt.grid('on')
    plt.xlabel('Focus Mu')
    plt.ylabel('Count')
    # pruning
    #fig, axs = plt.subplots(15, sharex=True, sharey=True, gridspec_kw={'hspace': 0})
    #ax = fig.add_axes([0, 0, 1, 1],alpha=0.)
    #ax.axis('on')
    plt.figure()
    dom_ix = foc_1.get_input_shape_at(0)[1]
    mus_int = np.int32(all_mus*20)
    uniquq_mus =np.unique(mus_int)
    sis_hist2d = np.zeros(shape=(np.max(uniquq_mus)+1,10))
    leg_list=[]
    for i,u in enumerate(uniquq_mus):
        sis_hist2d[u,:] = np.histogram(all_sis[mus_int==u],np.linspace(0,0.4,11))[0]
        leg_list.append(str(u))
        #plt.subplot(axs[u])
    
    plt.plot(np.tile(np.linspace(0,0.5,10),(sis_hist2d.shape[0],1)).T,
             sis_hist2d.T,alpha=0.7)
    #plt.legend(leg_list)
        #axs[u].axis('off')    
    #fig.add_axes([0, 0, 1, 1], frameon=False, facecolor='g')
    locs, labels = plt.xticks()  
    print(locs,labels)
    #plt.xticks(np.linspace(0,0.5,11))
    #plt.yticks(np.linspace(0,1.0,sis_hist2d.shape[0]))
    plt.grid('on')
    plt.ylabel('Histogram per mu')
    plt.xlabel('Focus aperture (sigma)')
    plt.show()
    
    
    fi_1p = fi_1*(fi_1>0.3).astype('float')
    fi_1p_w = fi_1p*we
    fi_1p_2d = np.reshape(fi_1p.T, (fi_1p.T.shape[0],in_rows,in_cols,in_channels))
    fi_1p_w_2d = np.reshape(fi_1p_w.T, (fi_1p_w.T.shape[0],in_rows,in_cols,in_channels))
    
    if not cnn_model:
        pltfoc2d = np.zeros(shape=(fi_1_w_2d[0].shape[0],fi_1_w_2d[0].shape[1],3))
        if dset=='mnist':
            pltfoc2d[:,:,2] = (np.squeeze(fi_1_2d[n_ix[0]]))
            pltfoc2d[:,:,0] = (np.squeeze(fi_1_2d[n_ix[1]]))
            pltfoc2d[:,:,1] = (np.squeeze(fi_1_2d[n_ix[2]]))
        else:
            pltfoc2d[:,:,:] = (np.squeeze(fi_1_2d[n_ix[0]]))
        plt.figure()
        print('Plotting three projections foci')
        plt.imshow(pltfoc2d)
        k = 1
        for x,y,si in zip(mus_x,mus_y,sel_sis):
            plt.text(x,y,'+',fontsize=18,c='white')
            plt.text(x+2,y,'s='+str(si),fontsize=14,c='white')
        
        
        plt.figure()
        pltfoc2d = np.zeros(shape=(fi_1_w_2d[0].shape[0],fi_1_w_2d[0].shape[1],3))
        if dset=='mnist':
            pltfoc2d[:,:,2] = (np.squeeze(fi_1_w_2d[n_ix[0]]))*5
            pltfoc2d[:,:,0] = (np.squeeze(fi_1_w_2d[n_ix[1]]))*5
            pltfoc2d[:,:,1] = (np.squeeze(fi_1_w_2d[n_ix[2]]))*5
        else:
            pltfoc2d[:,:,:] = np.squeeze(fi_1_w_2d[n_ix[0]])*8
    
        plt.imshow(pltfoc2d)
        k = 1
        for x,y,si in zip(mus_x,mus_y,sel_sis):
            plt.text(x,y,'+',fontsize=18,c='white')
            plt.text(x+2,y,'s='+str(si),fontsize=14,c='white')
    
    
    
    if len(nhidden)>1:
        fi_2, we, fi_2_w = calculate_fi_and_weights(foc_2)
        plt.figure()
        plt.plot(fi_2_w[:,n_ix])
        plt.gca().set_prop_cycle(None)
        plt.plot(fi_2[:,n_ix],'--',alpha=0.7)
        plt.xlabel('Input index')
        plt.ylabel(r'Focused weights (& Focus)')
        plt.grid('on')
        
        plt.figure()
        all_sis2 = foc_2.get_weights()[1]
        print('Plotting histogram of Mus')
        plt.hist(all_sis,alpha=0.5)
        plt.hist(all_sis2, alpha=0.5)
        plt.grid('on')
        plt.xlabel('Focus Si')
        plt.ylabel('Count')
        plt.legend(['layer 1', 'layer 2'])

else:
    layer_names = ['dense-1', 'dense-2']
    
def normcam(cam):
    cam1 = cam - np.min(cam)
    cam_img = cam1 / np.max(cam1)
    cam_img = np.uint8(255 * cam_img)
    return cam_img

def get_cam_cnn(test_img,for_cls,channel_index, cnn_shape, foc_shape):


    input_layer = model.get_layer(index=0)
    foc_1_layer = model.get_layer('activation_1')
    cnn_layer = model.get_layer('max_pooling2d_1')
    cnn_shape = cnn_layer.get_output_shape_at(0)
    soft_layer = model.get_layer('softmax')
    output_layer_tensor = model.output[:, for_cls]
    
    grad_input = K.gradients(output_layer_tensor, input_layer.output)[0]
    grad_foc_1 = K.gradients(output_layer_tensor, foc_1_layer.output)[0]
    grad_cnn_2 = K.gradients(output_layer_tensor, cnn_layer.output)[0]
    
    iterate = K.function([model.input, K.learning_phase()], 
                          [foc_1_layer.output[0],
                           cnn_layer.output[0], 
                           grad_input,grad_foc_1,grad_cnn_2, output_layer_tensor])
    
    foc_1_value, cnn_2_value, grad_input_value, grad_foc_1_value, grad_cnn_2_value, cls_out= iterate([test_img, 0])
    
    foc_1_heatmap = foc_1_value*soft_layer.get_weights()[0][:,for_cls]
    cnn_2_value_flatten = cnn_2_value.flatten()
    conv_cam = np.zeros_like(cnn_2_value_flatten)
    for i,f in enumerate(foc_1_heatmap):
        conv_cam += f* fi_1_w[:,i]*cnn_2_value_flatten
    
    conv_cam2d = np.sum(np.reshape(conv_cam,cnn_shape[1:]),axis=2)
    
    conv_cam2dnormed = normcam(conv_cam2d)
    
    return conv_cam2dnormed, conv_cam2d, foc_1_heatmap


def get_cam_cnn_grad(test_img,for_cls,channel_index, cnn_shape, foc_shape):


    input_layer = model.get_layer(index=0)
    foc_1_layer = model.get_layer('activation_1')
    cnn_layer = model.get_layer('max_pooling2d_1')
    cnn_shape = cnn_layer.get_output_shape_at(0)
    soft_layer = model.get_layer('softmax')
    output_layer_tensor = model.output[:, for_cls]
    
    grad_input = K.gradients(output_layer_tensor, input_layer.output)[0]
    grad_foc_1 = K.gradients(output_layer_tensor, foc_1_layer.output)[0]
    grad_cnn_2 = K.gradients(output_layer_tensor, cnn_layer.output)[0]
    
    iterate = K.function([model.input, K.learning_phase()], 
                          [foc_1_layer.output[0],
                           cnn_layer.output[0], 
                           grad_input,grad_foc_1,grad_cnn_2, output_layer_tensor])
    
    foc_1_value, cnn_2_value, grad_input_value, grad_foc_1_value, grad_cnn_2_value, cls_out= iterate([test_img, 0])
    
    foc_1_heatmap = foc_1_value* grad_foc_1_value
    conv_heatmap = cnn_2_value* grad_cnn_2_value
    
    conv_cam2d = np.sum(np.reshape(conv_heatmap,cnn_shape[1:]),axis=2)
    
    conv_cam2dnormed = normcam(conv_cam2d)
    
    return conv_cam2dnormed, conv_cam2d, foc_1_heatmap


def get_cam_simple(test_img,for_cls,channel_index):
    in_channels = test_img.shape[channel_index]

    foc_1_layer = model.get_layer('activation_1')
    foc_2_layer = model.get_layer('activation_2')
    soft_layer = model.get_layer('softmax')
    output_layer_tensor = model.output[:, for_cls]
    
    
    iterate = K.function([model.input, K.learning_phase()], 
                          [foc_1_layer.output[0],
                           foc_2_layer.output[0], output_layer_tensor])
    
    foc_1_value, foc_2_value, cls_out= iterate([test_img, 0])
    input_value = test_img.flatten()
    foc_2_heatmap = foc_2_value*soft_layer.get_weights()[0][:,for_cls]
    foc_1_heatmap = np.zeros_like(foc_1_value)
    for i,f in enumerate(foc_2_heatmap):
        foc_1_heatmap += f * fi_2_w[:,i] * foc_1_value
    
    input_heatmap = np.zeros_like(input_value).flatten()
    for i,f in enumerate(foc_1_heatmap):
        input_heatmap += f * fi_1_w[:,i] * input_value
        
    foc_2_heatmap2d = np.reshape(foc_2_heatmap,(img_rows,img_cols))
    foc_1_heatmap2d = np.reshape(foc_1_heatmap,(img_rows,img_cols))
    
    input_heatmap2d = np.reshape(input_heatmap,(img_rows,img_cols,in_channels))
    if in_channels==3:
        input_heatmap2d= np.sum(input_heatmap2d, axis=2)
    
    input_heatmap2dnorm = normcam(input_heatmap2d)

    return input_heatmap2dnorm, input_heatmap2d, foc_1_heatmap2d, foc_2_heatmap2d


def get_cam_simple_grad(test_img,for_cls,channel_index, layer_names):
    in_channels = test_img.shape[channel_index]
    
    input_layer = model.get_layer(index=0)
    foc_1_layer = model.get_layer(layer_names[0])
    foc_2_layer = model.get_layer(layer_names[1])
    soft_layer = model.get_layer('softmax')
    output_layer_tensor = model.output[:, for_cls]
    
    grad_input = K.gradients(output_layer_tensor, input_layer.output)[0]
    grad_foc_1 = K.gradients(output_layer_tensor, foc_1_layer.output)[0]
    grad_foc_2 = K.gradients(output_layer_tensor, foc_2_layer.output)[0]
    
    iterate = K.function([model.input, K.learning_phase()], 
                          [foc_1_layer.output[0],
                           foc_2_layer.output[0], 
                           grad_input,grad_foc_1,grad_foc_2, output_layer_tensor])
    
    foc_1_value, foc_2_value, grad_input_value, grad_foc_1_value, grad_foc_2_value, cls_out= iterate([test_img, 0])
    
    foc_2_heatmap = foc_2_value*grad_foc_2_value[0,:]
    foc_1_heatmap = foc_1_value*grad_foc_1_value[0,:]
    input_value = test_img.flatten()
    input_heatmap = input_value*grad_input_value.flatten()
    print(input_heatmap.shape)
    foc_2_heatmap2d = np.reshape(foc_2_heatmap,(img_rows,img_cols))
    foc_1_heatmap2d = np.reshape(foc_1_heatmap,(img_rows,img_cols))
    
    input_heatmap2d = np.reshape(input_heatmap,(img_rows,img_cols,in_channels))
    if in_channels==3:
        input_heatmap2d= np.sum(input_heatmap2d, axis=2)
    
    input_heatmap2dnorm = normcam(input_heatmap2d)
    
    return input_heatmap2dnorm, input_heatmap2d, foc_1_heatmap2d, foc_2_heatmap2d
    

    

''' cifar-10 names
airplane : 0
automobile : 1
bird : 2
cat : 3
deer : 4
dog : 5
frog : 6
horse : 7
ship : 8
truck : 9
'''
ex_s=4
test_inp = x_test[ex_s][np.newaxis]
if n_channels==3:
    plt.imshow(np.uint8((test_inp[0]+2)*64))
else:
    plt.imshow(np.uint8((np.squeeze(test_inp[0])+2)*64))
_cls =np.argmax(y_test[ex_s])
print(_cls)
ci=3

import cv2
if cnn_model:
    
    cnn_map1normed, cnn_map1, foc_map_1 = get_cam_cnn_grad(test_inp, 
                                                      for_cls=_cls,
                                                      channel_index=ci,
                                                      cnn_shape=(16,16,32),
                                                      foc_shape=(16,16)
                                                      )
    cnn_map2normed, cnn_map2, foc_map_2= get_cam_cnn_grad(test_inp, 
                                                     for_cls=7,
                                                     channel_index=ci,
                                                     cnn_shape=(16,16,32),
                                                     foc_shape=(16,16))

    inp_map_1 = cv2.resize(cnn_map1normed, dsize=(img_rows, img_cols), interpolation=cv2.INTER_CUBIC)
    inp_map_1c = cv2.applyColorMap(inp_map_1, cv2.COLORMAP_JET)

    inp_map_2 = cv2.resize(cnn_map2normed, dsize=(img_rows, img_cols), interpolation=cv2.INTER_CUBIC)
    inp_map_2c = cv2.applyColorMap(inp_map_2, cv2.COLORMAP_JET)
    
else:
    inp_map_1norm, inp_map_1, foc_1_cam_1, foc_2_cam_1 = get_cam_simple_grad(test_inp, 
                                      for_cls=_cls,
                                      channel_index=ci,
                                      layer_names=layer_names)
    inp_map_2norm, inp_map_2, foc_1_cam_2,foc_2_cam_2 = get_cam_simple_grad(test_inp, 
                                      for_cls=0,
                                      channel_index=ci,
                                      layer_names=layer_names)
    
    inp_map_1 = cv2.resize(inp_map_1norm, dsize=(img_rows, img_cols), interpolation=cv2.INTER_CUBIC)
    inp_map_1c = cv2.applyColorMap(inp_map_1, cv2.COLORMAP_JET)

    inp_map_2 = cv2.resize(inp_map_2norm, dsize=(img_rows, img_cols), interpolation=cv2.INTER_CUBIC)
    inp_map_2c = cv2.applyColorMap(inp_map_2, cv2.COLORMAP_JET)


plt.figure(figsize=(12,4))
img = np.squeeze(np.uint8((test_inp[0,:,:,:]+2)*64))
plt.subplot(131)
plt.imshow(img,alpha=1.0, cmap='gray')
plt.axis('off')
plt.subplot(132)
plt.imshow(img, cmap='gray',alpha=0.8)
plt.imshow(inp_map_1c,cmap='jet',alpha=0.4)
plt.axis('off')
plt.subplot(133)
plt.imshow(img, cmap='gray',alpha=0.8)
plt.imshow(inp_map_2c,cmap='jet',alpha=0.4)
plt.axis('off')

##########################################SHAP
show_shap=True
if show_shap:
    import shap
    
    # select a set of background examples to take an expectation over
    background = x_train[np.random.choice(x_train.shape[0], 100, replace=False)]
    
    # explain predictions of the model on four images
    e = shap.DeepExplainer(model, background)
    # ...or pass tensors directly
    # mnist simple samps = [0,1,2,3,6]
    samps = [4,29,65]
    #samps = [3,5,6,9,12,13,15,23]
    # e = shap.DeepExplainer((model.layers[0].input, model.layers[-1].output), background)
    shap_values = e.shap_values(x_test[samps])
    
    # plot the feature attributions
    plt.figure()
    shap.image_plot(shap_values, (x_test[samps]+2)*64, 
                    np.tile(np.array(class_names),(len(samps),1)),sharetitles=True)

    X_s = x_test[0:50]
    samps = [3,6] # mnist
    samps = [29,65] # mnist(29,33)
    to_explain = x_test[samps]
    
    # explain how the input to the 7th layer of the model explains the top two classes
    def map2layer(x, layer):    
        feed_dict = dict(zip([model.layers[0].input], [x]))
        return K.get_session().run(model.layers[layer].input, feed_dict)
    lay = 2 # last cnn layer
    e = shap.GradientExplainer((model.layers[lay].input, 
                                model.layers[-1].output),
                                map2layer(X_s, lay),
                                local_smoothing=0) 
                                # std dev of smoothing noise)
    shap_values,indexes = e.shap_values(map2layer(to_explain, lay), 
                                        ranked_outputs=2)
    # get the names for the classes
    index_names = np.vectorize(lambda x: class_names[x])(indexes)
    if cnn_model:
        shap_values_=shap_values 
        #shap_values_ = [np.reshape(x,(to_explain.shape[0],16,16)) for x in shap_values]
    else:
        shap_values_ = [np.reshape(x,(to_explain.shape[0],28,28)) for x in shap_values]
    # plot the explanations
    to_explain_edges = [cv2.Canny(i) for i in to_explain]

    shap.image_plot(shap_values_, (to_explain+2)*64,index_names)
    
    ## try mapping focus functions
    lay = 7  # input to dropout.
    e = shap.GradientExplainer((model.layers[lay].input, 
                                model.layers[-1].output),
                                map2layer(X_s, lay),
                                local_smoothing=0) 
                                # std dev of smoothing noise)
    shap_values,indexes = e.shap_values(map2layer(to_explain, lay), 
                                        ranked_outputs=2)
    # get the names for the classes
    index_names = np.vectorize(lambda x: class_names[x])(indexes)
    if mod=='focused':
        c =np.dot(fi_1,shap_values[0][0])
    else:
        c =np.dot(model.layers[6].weights[0], shap_values[0][0]>0.)
    cc = np.reshape(c, (in_rows,in_cols,32)) 
    cc_mean = normcam(np.max(cc, axis=2))
    foc_map = cv2.resize(cc_mean, dsize=(img_rows, img_cols), interpolation=cv2.INTER_CUBIC)
    plt.figure()
    plt.imshow(np.uint8((to_explain[0]+2)*32),alpha=0.8)
    plt.imshow(foc_map,cmap='gray',alpha=0.5)
    plt.axis('off')
#    
#####################################################################################
root_to_save ='/home/btek/Dropbox/code/pythoncode/FocusingNeuron/outputs/Kfocusing/vids/'
file_to_save = root_to_save+'cifar10-records'
np.savez_compressed(file_to_save,mu_0 = cb[5].record,        si_0 = cb[4].record,w_0 = cb[3].record)
cb=np.load(file_to_save+'.npz')
record_fi_w_video= False
if record_fi_w_video:

    import matplotlib.animation as animation
    #FFMpegWriter = animation.writers['ffmpeg']
    metadata = dict(title='Movie Test', artist='Matplotlib',
                comment='Movie support!')
    writer = animation.FFMpegWriter(fps=4, codec='mpeg4',metadata=metadata)

    im_list=[]

    fig, ax = plt.subplots()
    

    
with writer.saving(fig, root_to_save+dset+'_cnn_'+str(cnn_model)+"_fi_all.mp4", 100):
    for e in range(Epochs):
        mu_0 = cb['mu_0'][e]
        si_0 = cb['si_0'][e]
        w_0 = cb['w_0'][e]
        x_v = np.linspace(0,1,w_0.shape[0])
        fay_0 = U_numeric(x_v,mu_0,si_0,1)
    
        lines_=[]
        ax.clear()
        
        # plots focus functions
        #ax.set_title('$\phi$s at iteration '+str(e))
        #for n in range(0,fay_0.shape[0],80):
        #    line_= ax.plot(x_v, fay_0[n])


        # plots focus times weight
        n_ix = [8, nhidden[0]//2, nhidden[0]-10]
        
        for n in range(0,fay_0.shape[1],10):
       
        #for n in n_ix:
            line_= ax.plot(x_v, fay_0[:,n],alpha=0.6)
        #ax.legend([str(l) for l in n_ix],loc='upper right')
        ax.set_ylim([-0.005,8.0])
        ax.set_xlabel('index')
        ax.set_ylabel('Focus magnitude')
        ax.grid('on')
        ax.set_title('Epoch:'+str(e))
        
        #plt.show()
        writer.grab_frame()

fig, ax = plt.subplots()
with writer.saving(fig, root_to_save+dset+'_cnn_'+str(cnn_model)+"_fi_w.mp4", 100):
    for e in range(Epochs):
        mu_0 = cb['mu_0'][e]
        si_0 = cb['si_0'][e]
        w_0 = cb['w_0'][e]
        x_v = np.linspace(0,1,w_0.shape[0])
        fay_0 = U_numeric(x_v,mu_0,si_0,1)
    
        lines_=[]
        ax.clear()
        
        # plots focus functions
        #ax.set_title('$\phi$s at iteration '+str(e))
        #for n in range(0,fay_0.shape[0],80):
        #    line_= ax.plot(x_v, fay_0[n])
        n_ix = [8, nhidden[0]//2, nhidden[0]-10]
        for n in range(0,fay_0.shape[1],10):
        #for n in n_ix:
            line_= ax.plot(x_v, fay_0[:,n]*w_0[:,n],alpha=0.6)
        ax.legend([str(l) for l in n_ix],loc='upper right')
        ax.set_ylim([-1.0,1.0])
        ax.set_xlabel('index')
        ax.set_ylabel('Weight magnitude')
        ax.set_title('Epoch:'+str(e))
        ax.grid('on')
        
        #plt.show()
        writer.grab_frame()

            
    
        