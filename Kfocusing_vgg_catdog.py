#!/usr/bin/env python3
# -*- coding: utf-8 -*-


'''This script goes along the blog post
"Building powerful image classification models using very little data"
from blog.keras.io.
It uses data that can be downloaded at:
https://www.kaggle.com/c/dogs-vs-cats/data
In our setup, we:
- created a data/ folder
- created train/ and validation/ subfolders inside data/
- created cats/ and dogs/ subfolders inside train/ and validation/
- put the cat pictures index 0-999 in data/train/cats
- put the cat pictures index 1000-1400 in data/validation/cats
- put the dogs pictures index 12500-13499 in data/train/dogs
- put the dog pictures index 13500-13900 in data/validation/dogs
So that we have 1000 training examples for each class, and 400 validation examples for each class.
In summary, this is our directory structure:
```
data/
    train/
        dogs/
            dog001.jpg
            dog002.jpg
            ...
        cats/
            cat001.jpg
            cat002.jpg
            ...
    validation/
        dogs/
            dog001.jpg
            dog002.jpg
            ...
        cats/
            cat001.jpg
            cat002.jpg
            ...
```
'''

import time 
from shutil import copyfile
delayed_start = 0  * 3600 * 4
print("Delayed start ",delayed_start)
time.sleep(delayed_start)

import os
os.environ['CUDA_VISIBLE_DEVICES']="0"
os.environ['TF_FORCE_GPU_ALLOW_GROWTH']="true"

import keras    
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, BatchNormalization
from keras.layers import Activation, Dropout, Flatten, Dense, Lambda, Input, Permute
from keras.optimizers import RMSprop, SGD
from keras import backend as K

from keras_utils import WeightHistory as WeightHistory
from keras_utils import RecordVariable, \
PrintLayerVariableStats, PrintAnyVariable, \
SGDwithLR, eval_Kdict, standarize_image_025
#from keras_preprocessing.image import ImageDataGenerator
from Kfocusing import FocusedLayer1D
import numpy as np

from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input
np.random.seed(9)

#config = K.tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.3
#   # Create a session with the above options specified.
#K.tensorflow_backend.set_session(K.tf.Session(config=config))



Reps = 5

list_hist_trn_acc=[[]]*Reps
list_hist_val_acc=[[]]*Reps
list_hist_trn_loss=[[]]*Reps
list_hist_val_loss=[[]]*Reps
    
list_val_sc=[[]]*Reps
list_tst_sc=[[]]*Reps

# dimensions of our images.
img_width, img_height = 125, 125
root = '/media/home/rdata/dogs-vs-cats/'
#root = '/home/btek/datasets/dogs-vs-cats/'
train_data_dir = root+'train/'
validation_data_dir = root+'validation/'
nb_train_samples = 9500 #9500
nb_validation_samples = 3000 #300
epochs = 50
batch_size = 64
focusing='focused'

from datetime import datetime
now = datetime.now()



lr_dict = {'all':1e-3,
          'focus-1/Sigma:0': 1e-2,'focus-1/Mu:0': 1e-3,'focus-1/Weights:0':1e-3,
          'focus-2/Sigma:0': 1e-2,'focus-2/Mu:0': 1e-3,'focus-2/Weights:0': 1e-3,
          'dense_1/Weights:0':1e-3}

#lr_dict = {'all':1e-3}

mom_dict = {'all':0.9,'focus-1/Sigma:0': 0.9,'focus-1/Mu:0': 0.9,
           'focus-2/Sigma:0': 0.9,'focus-2/Mu:0': 0.9}
# simple
decay_dict = {'all':0.9, 'focus-1/Sigma:0': 0.9,'focus-1/Mu:0':0.9,
              'focus-2/Sigma:0': 0.9,'focus-2/Mu:0': 0.9}
# dec-2.0
#decay_dict = {'all':0.9, 'focus-1/Sigma:0': 2.0,'focus-1/Mu:0':0.9,
#              'focus-2/Sigma:0': 1.0,'focus-2/Mu:0': 0.9}

#decay_dict = {'all':0.9, 'focus-1/Sigma:0': 2.0,'focus-1/Mu:0':2.0,
#              'focus-2/Sigma:0': 1.0,'focus-2/Mu:0': 0.9}
clip_dict = {'focus-1/Sigma:0':(0.01,1.0),'focus-1/Mu:0':(0.0,1.0),
         'focus-2/Sigma:0':(0.01,1.0),'focus-2/Mu:0':(0.0,1.0)}

e_i = nb_train_samples // batch_size

#decay_epochs =np.array([e_i*10], dtype='int64') #for 20 epochs
#decay_epochs =np.array([e_i*10,e_i*80,e_i*120,e_i*160], dtype='int64')
decay_epochs =np.array([e_i*10, e_i*20, e_i*120, e_i*180], dtype='int64')

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)


for r in range(Reps):

    K.clear_session()
    #K.get_session().close()
    #K.get_session()
    
    base_in = Input(shape=input_shape, name='inputlayer')
    
    
    #base_model = resnet.ResNet50(weights='imagenet', include_top=False,
    #                         input_shape=input_shape,input_tensor=base_in)
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape,
                      input_tensor=base_in)
    #base_in = Input(shape=FRAME_SIZE, name='inputlayer')
    #base_model = Conv2D(16,3,padding='valid')(base_in)
    
    #base_model = InceptionV3(weights='imagenet', include_top=False,
    #                         input_shape=FRAME_SIZE)
    
    # add a global spatial average pooling layer
    #x = base_model.output
    x=base_model.output
    
    #x = Permute((3,1,2))(x)
    #x = Flatten()(x)
    x = GlobalAveragePooling2D()(x)
    
    pad_input =False
    if pad_input:
        print("PADDING LAYER OUPUT")
        
        paddings = K.tf.constant([[0, 0,], [3, 3]])
    
        padding_layer = Lambda(lambda x: K.tf.pad(x,paddings,"CONSTANT"))
        x = padding_layer(x)
    #x = Dropout(0.1)(x)
    # let's add a fully-connected layer
    
    if focusing== 'focused':
        nf = 40#init_sigma=np.exp(-(np.linspace(0.1, 0.9, nf)-0.5)**2/0.07),
        x = FocusedLayer1D(units=nf,
                           name='focus-1',
                           activation='linear',
                           init_sigma=0.025,
                           init_mu=np.linspace(0.1, 0.9, nf),
                           init_w= None,
                           train_sigma=True,
                           train_weights=True,
                           train_mu = True,
                           normed=2)(x)
    elif focusing=='dense':
        x = Dense(40, activation='linear')(x)
    else:
        print('unknown mod')
        
        
    x = BatchNormalization()(x)
    #from functools import partial
    #act = partial(keras.activations.relu, alpha=0.01, max_value=None, threshold=0.)
    #x = Activation(act)(x)
    x = Activation('relu')(x)
    x = Dropout(0.2)(x)
    # and a logistic layer -- let's say we have 200 classes
    predictions = Dense(2, activation='softmax')(x)
    
    # this is the model we will train
    model = Model(inputs=base_in, outputs=[predictions])
    #model = Model(inputs=base_in, outputs=[predictions])
    
    stat_func_name = ['max: ', 'mean: ', 'min: ', 'var: ', 'std: ']
    stat_func_list = [np.max, np.mean, np.min, np.var, np.std]
    callbacks=[]
    recordvars =False
    if focusing=='focused' and recordvars:
        pr_1 = PrintLayerVariableStats("focus-1","Weights:0",stat_func_list,stat_func_name)
        pr_2 = PrintLayerVariableStats("focus-1","Sigma:0",stat_func_list,stat_func_name)
        pr_3 = PrintLayerVariableStats("focus-1","Mu:0",stat_func_list,stat_func_name)
        rv_weights_1 = RecordVariable("focus-1","Weights:0")
        rv_sigma_1 = RecordVariable("focus-1","Sigma:0")
        rv_mu_1 = RecordVariable("focus-1","Mu:0")
        print_lr_rates_callback = keras.callbacks.LambdaCallback(
                on_epoch_end=lambda epoch, logs: print("iter: ", 
                                                       K.eval(model.optimizer.iterations),
                                                       " LR RATES :", 
                                                       eval_Kdict(model.optimizer.lr)))
    
        callbacks+=[pr_1,pr_2,pr_3,rv_weights_1,rv_sigma_1, rv_mu_1,
                    print_lr_rates_callback]
    
    
    
    optimizer_s = 'SGDwithLR'
    if optimizer_s == 'SGDwithLR':
        opt = SGDwithLR(lr_dict, mom_dict,decay_dict,clip_dict, decay_epochs)#, decay=None)
    elif optimizer_s=='RMSprob':
        opt = RMSprop(lr=0.01, rho=0.9, epsilon=None, decay=0.0)
    else:
    #opt= SGDwithLR({'all': 0.01},{'all':0.9})#, decay=None)
        opt= SGD(lr=0.01, momentum=0.9)#, decay=None)
    
    
    # compile the model (should be done *after* setting layers to non-trainable)
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=opt,
                  metrics=['accuracy'])
        
    model.summary()
    
    print("Repeat: ",r)
    
    # this is the augmentation configuration we will use for training
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
    
    # this is the augmentation configuration we will use for testing:
    # only rescaling
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    
    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical')
    
    validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical')
    
    history = model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples // batch_size,
        verbose=2,callbacks=callbacks)
    test_generator = test_datagen.flow_from_directory(
        directory=validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False,
        seed=9
    )
    mx_score = np.max(history.history['val_acc'])
    
    sc = model.evaluate_generator(test_generator, test_generator.n // test_generator.batch_size)
    
    list_hist_trn_acc[r]=history.history['acc']
    list_hist_val_acc[r]=history.history['val_acc']
    list_hist_trn_loss[r]=history.history['loss']
    list_hist_val_loss[r]=history.history['val_loss']
    
    list_val_sc[r]=mx_score
    list_tst_sc[r]=sc

then = datetime.now()
from scipy.stats import ttest_ind

timestr = then.strftime("%Y%m%d-%H%M%S")+now.strftime("%H%M%S")
print("Val mean", np.mean(list_val_sc))
print("Val std", np.std(list_val_sc))
print("Val max", np.max(list_val_sc))
print("test mean ", np.mean(list_tst_sc,axis=0))
print("test std ", np.std(list_tst_sc,axis=0))
print("test max ", np.max(list_tst_sc,axis=0))
filename = 'outputs/transfer-cifar10/'+'catdog_'+timestr+'_'+focusing+'_n_results.npz'
np.savez_compressed(filename, list_hist_trn_acc=list_hist_trn_acc,
                    list_hist_val_acc = list_hist_val_acc,
                    list_hist_trn_loss = list_hist_trn_loss,
                    list_hist_val_loss = list_hist_val_loss,
                    ls_val=list_val_sc,ls_tst=list_tst_sc)


# paper results 
# focus
# 50 epochs test results
'''
fs: [0.97883065 0.97530242 0.97731855 0.97009409 0.97160618]
ds: [0.97748656 0.97143817 0.97530242 0.97849462 0.97832661]
Focus mean 97.46303763440861
Focus std 0.3317946515143079
Focus max 97.88306451612904
Dense mean 97.62096774193549
Dense std 0.2642847056752107
Dense max 97.84946236559139
Ttest_indResult(statistic=-0.7446259943662625, pvalue=0.4777989481817849)
'''
plt_losses=False
if plt_losses:
    import numpy as np
    from scipy.stats import ttest_ind

    root = 'outputs/transfer-cifar10/'
    f1 = np.load(root+'catdog_20191101-220030175139_focused_n_results.npz')['ls_tst']
    d1 = np.load(root+'catdog_20191101-222315170543_dense_n_results.npz')['ls_tst']

    #d1=np.load(root+'experiment20190826-114828095952_densen_results.npz')['ls_mx_sc']
    #f1=np.load(root+'experiment20190905-133509041531_focusedn_results.npz')['ls_mx_sc']
    
    fs=np.array(f1)[:,1]
    ds=np.array(d1)[:,1]
    print("fs:",fs)
    print("ds:",ds)
    #fs = np.concatenate((f1,f2,f3))
    #ds = np.concatenate((d1,d2))
    print("Focus mean",100*np.mean(fs,axis=0))
    print("Focus std",100*np.std(fs,axis=0))
    print("Focus max",100*np.max(fs,axis=0))
    
    print("Dense mean",100*np.mean(ds,axis=0))
    print("Dense std",100*np.std(ds,axis=0))
    print("Dense max",100*np.max(ds,axis=0))
    
    
    ttest_ind(fs,ds,axis=0)
    
    # plot val accuracies
    f1 = np.load(root+'catdog_20191101-220030175139_focused_n_results.npz')['list_hist_val_acc']
    d1 = np.load(root+'catdog_20191101-222315170543_dense_n_results.npz')['list_hist_val_acc']

    fvals=np.array(f1).T
    dvals=np.array(d1).T
    f_val = np.mean(fvals,axis=1)
    f_error = np.std(fvals,axis=1)
    
    d_val = np.mean(dvals,axis=1)
    d_error = np.std(dvals,axis=1)
    
    x =np.arange(0,f_val.shape[0])+1
    import matplotlib.pyplot as plt
    from plot_utils import save_fig, paper_fig_settings
    paper_fig_settings(+4)
    plt.figure()
    plt.plot(f_val,'r-', alpha=0.8)
    plt.fill_between(x, f_val-f_error, f_val+f_error,alpha=0.1, edgecolor='red', facecolor='#FF1010')
    plt.plot(d_val,'g--', alpha=0.8)
    plt.fill_between(x, d_val-d_error, d_val+d_error, alpha=0.1,edgecolor='g', facecolor='#10FF10')
    plt.grid('on')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy')
    plt.legend(['Vgg+Fcs','Vgg+Dns'], loc=4)
    plt.show()
    
visualize=False
if visualize:
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt
    import cv2
    import numpy as np
    from keras.preprocessing import image
    
    def plot_cam(img_path,for_cls):


        img = image.load_img(img_path, target_size=(img_width, img_height))
    
        test_img = image.img_to_array(img)
        test_img = np.expand_dims(test_img, axis=0)
        #from keras.applications.vgg16 import preprocess_input
        #test_img = preprocess_input(test_img)
        test_img *= 1./ 255
        plt.imshow(test_img[0,:,:,0])
    
        #mx_cls_prob = model.output[0][for_cls]
    
        last_conv_layer = model.get_layer('block5_conv3')
        output_layer_tensor = model.output[:, for_cls]
    
        global_grads = K.gradients(output_layer_tensor, model.get_layer('global_average_pooling2d_1').output)[0]
        iterate = K.function([model.input, K.learning_phase()], [global_grads, last_conv_layer.output[0]])
    
        pl_grads,conv_layer_output_value = iterate([test_img,0])

        for i in range(512):
            conv_layer_output_value[:, :, i] *= pl_grads[0,i]

        def normcam(cam):
            cam1 = cam - np.min(cam)
            cam_img = cam1 / np.max(cam1)
            cam_img = np.uint8(255 * cam_img)
            return cam_img
        
        
        heatmap = normcam(np.sum(conv_layer_output_value,axis=2))
        #heatmap = np.sum(conv_layer_output_value,axis=2)
        print(np.mean(heatmap), np.max(heatmap), np.min(heatmap))
        plt.matshow(heatmap)
        plt.show()
    
        inp_img = img
        heatmap2 = cv2.resize(heatmap, (img_width, img_height))
        #heatmap2c = np.uint8(255 * heatmap2)
        heatmap2c = cv2.applyColorMap(heatmap2, cv2.COLORMAP_JET)
        #hif = .8
        #superimposed_img = heatmap2c * hif + test_img
        #plt.imshow(inp_img, alpha=0.8), 
        #plt.imshow(heatmap2c, alpha=0.5)
        #plt.axis('off')
        return heatmap2c, heatmap2, inp_img

    
    hmapc_1, hmap_1,inp_img = plot_cam(img_path=root+'train/dogs/dog.117.jpg', for_cls=0)
    hmapc_2, hmap_2,_ = plot_cam(img_path=root+'train/dogs/dog.117.jpg', for_cls=1)
    
    
    plt.figure(figsize=(12,4))
    img = inp_img
    plt.subplot(131)
    plt.imshow(img,alpha=1.0, cmap='gray')
    plt.axis('off')
    plt.subplot(132)
    plt.imshow(img, cmap='gray',alpha=0.8)
    plt.imshow(hmapc_1,cmap='jet',alpha=0.4)
    plt.axis('off')
    plt.subplot(133)
    plt.imshow(img, cmap='gray',alpha=0.8)
    plt.imshow(hmapc_2,cmap='jet',alpha=0.4)
    plt.axis('off')
    
    hmapc_1, hmap_1,inp_img2 = plot_cam(img_path=root+'catpic/20190925_170837.jpg', for_cls=1)
    hmapc_2, hmap_2,_ = plot_cam(img_path=root+'catpic/20190925_170837.jpg', for_cls=0)
    
    plt.figure(figsize=(12,4))
    img = inp_img2
    plt.subplot(131)
    plt.imshow(img,alpha=1.0, cmap='gray')
    plt.axis('off')
    plt.subplot(132)
    plt.imshow(img, cmap='gray',alpha=0.8)
    plt.imshow(hmapc_1,cmap='jet',alpha=0.4)
    plt.axis('off')
    plt.subplot(133)
    plt.imshow(img, cmap='gray',alpha=0.8)
    plt.imshow(hmapc_2,cmap='jet',alpha=0.4)
    plt.axis('off')
    
    
    from Kfocusing import calculate_fi_and_weights
    
show_shap=False
if show_shap:
    import shap
    import matplotlib.pyplot as plt
    from matplotlib.image import  imread as imread
    from matplotlib.image import  resample as resample
    import cv2
    img_path=root+'catpic/20190925_170837.jpg'
    im_in = imread(img_path)
    im_rs = np.zeros((img_width, img_height,3), dtype=im_in.dtype)
    img =  cv2.resize(im_in, dsize=(img_width, img_height),interpolation=cv2.INTER_CUBIC)
    test_img = np.zeros((tuple([2])+img.shape))
    test_img[0] =img
    img_path=root+'train/dogs/dog.117.jpg'
    im_in = imread(img_path)
    im_rs = np.zeros((img_width, img_height,3), dtype=im_in.dtype)
    img =  cv2.resize(im_in, dsize=(img_width, img_height),interpolation=cv2.INTER_CUBIC)
    test_img[1]=img
        #from keras.applications.vgg16 import preprocess_input
        #test_img = preprocess_input(test_img)
    test_img *= 1./ 255
    # select a set of background examples to take an expectation over
    train_generator.batch_size = 64
    background = train_generator.next()[0]
    #background2 = train_generator.next()[0]
    #background3 = background+background2
    
    # explain predictions of the model on four images
    e = shap.DeepExplainer(model, background)
    # ...or pass tensors directly
    # mnist simple samps = [0,1,2,3,6]
    
    # e = shap.DeepExplainer((model.layers[0].input, model.layers[-1].output), background)
    shap_values = e.shap_values(test_img)
    
    # plot the feature attributions
    plt.figure()
    shap.image_plot(shap_values, test_img)

    
    
    # explain how the input to the 7th layer of the model explains the top two classes
    def map2layer(x, layer):    
        feed_dict = dict(zip([model.layers[0].input], [x]))
        return K.get_session().run(model.layers[layer].input, feed_dict)
    lay = 19  # last cnn layer
    e = shap.GradientExplainer((model.layers[lay].input, 
                                model.layers[-1].output),
                                map2layer(background, lay),
                                local_smoothing=0) 
                                # std dev of smoothing noise)
    shap_values,indexes = e.shap_values(map2layer(test_img, lay), 
                                        ranked_outputs=2)
    # get the names for the classes
    #index_names = np.vectorize(lambda x: class_names[x])(indexes)
    
    shap_values_=shap_values 
    shap_values_ = np.zeros((2,2)+(img_width, img_height))
    for i,s in enumerate(shap_values):
        for j,k in enumerate(s):
            hmap = np.mean(k,axis=2)    
            heatmap2 = cv2.resize(hmap, (img_width, img_height))
            #heatmap2c = np.uint8(255 * heatmap2)
            #heatmap2c = cv2.applyColorMap(heatmap2c, cv2.COLORMAP_JET)
            shap_values_[i][j] = heatmap2
    # plot the explanations
    shap.image_plot(list(shap_values_), test_img)
    
    
    
    def get_cmap_binary(functor, test_inp, dense_units, glob_units, conv_size, 
                        forcls_=0, out_layer_index=-1, 
                        pre_act_index=-3, dense_layer_index=-5 ):
        
        layer_outs = functor([test_inp, 0.])
        print(layer_outs[-1][0])
        mx_ix= np.argmax(layer_outs[-1][0])
        mx_sc = layer_outs[-1][0][mx_ix]

        sft_max = model.layers[out_layer_index]
        we_soft_max = sft_max.get_weights()[0]
        relu_act_2 = np.squeeze(layer_outs[pre_act_index])
        dense_out = np.squeeze(layer_outs[dense_layer_index])
        dense_cam = np.squeeze(we_soft_max[:,forcls_] * mx_sc * relu_act_2)
        
        glob_1_cam = np.zeros(shape=(glob_units,))
        
        dense_lay = model.layers[dense_layer_index]
        
        if focusing:
            fi_2, we, fi_2_w = calculate_fi_and_weights(dense_lay)
        else:
            fi_2_w = dense_lay.get_weights()[0].shape
            
        for r in range(glob_units):
            sm = 0
            for k in range(dense_units):
                sm+=np.squeeze(dense_cam[k]*fi_2_w[r,k])
            glob_1_cam[r] = sm

        inp_map = np.zeros(shape=(conv_size[0],conv_size[1], glob_units))
        for j in range(in_shape):
            sm = 0
            for r in range(n_units_1):
                sm+=np.squeeze(foc_1_cam[r]*fi_1_w[j,r])
            inp_map[j] = sm

        # flatten color axies
        inp_map = inp_map.reshape((img_rows,img_cols,n_channels))
        inp_map = np.sum(inp_map,axis=2)
        return inp_map, dense_cam, conv_1_cam
    
    ex_s = 0
    test_inp = x_train[ex_s][np.newaxis]
    _cls =np.argmax(y_train[ex_s])+2
    #_cls=2
    print(_cls)
    
    inp_map_1,_,_ =get_cmap(functor, test_inp, np.argmax(y_train[ex_s]))  # true label
    inp_map_2,_,_ =get_cmap(functor, test_inp, 0) 
    
    
    output = model.output[:, 0]