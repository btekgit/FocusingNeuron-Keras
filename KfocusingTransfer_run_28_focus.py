# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 17:52:19 2019

@author: btek
this file tests focusing neuron in transfer learning
of CIFAR-10 dataset
"""
import numpy as np
from skimage.transform import rescale
from keras.applications.resnet50 import preprocess_input

def padsingleimage(x, frame_size,random_pos=False, pre_scale=None):
    x_p = np.zeros(shape=(frame_size[0],frame_size[1],frame_size[2]),dtype=x.dtype)
    # accepts tensor 3 dims, channels last
    #print("input dims:",x.ndim)
    if pre_scale:
      x = rescale(x, pre_scale)
    if x.ndim < 3:
        n_channels = 1
    else:
        n_channels = x.shape[2]
    nrows,ncols = x.shape[0],x.shape[1]
    hh = nrows//2
    hw = ncols//2
    ncx,ncy = frame_size[0]//2, frame_size[1]//2
    if random_pos:
        ncx = np.random.randint(hw,frame_size[1]//2-hw-1) 
        ncy = np.random.randint(hh,frame_size[0]//2-hw-1)
    if n_channels==1 and frame_size[2]==1:
        x_p[ncy-hh:ncy+hh, ncx-hw:ncx+hw] = x
    elif n_channels==1 and frame_size[2]==3:
        x_p[ncy-hh:ncy+hh, ncx-hw:ncx+hw,0] = np.squeeze(x)
        x_p[ncy-hh:ncy+hh, ncx-hw:ncx+hw,1] = np.squeeze(x)
        x_p[ncy-hh:ncy+hh, ncx-hw:ncx+hw,2] = np.squeeze(x)
    else:
        x_p[ncy-hh:ncy+hh, ncx-hw:ncx+hw,:] = x
    
    print("REMOVED PREPROCESS FROM HERE")
    #x_p=preprocess_input(x_p)
    return x_p

def paddataset(trn, val, tst, frame_size, random_pos=False,pre_scale=None):
    ntrn = trn.shape[0]
    nrows,ncols = frame_size[0],frame_size[1]
    nchan = frame_size[2]
    if trn.ndim==4:
        nchan = frame_size[2]
    trn2 = np.zeros(shape=(ntrn,nrows,ncols,nchan),dtype=trn.dtype)
    for i in range(ntrn):
        trn2[i] = padsingleimage(trn[i], frame_size, random_pos,pre_scale)
    
    val2=None
    tst2=None    
    if val is not None:
        nval = val.shape[0]
        val2 = np.zeros(shape=(nval,nrows,ncols,nchan),dtype=val.dtype)
        for i in range(nval):
            val2[i] = padsingleimage(val[i], frame_size, random_pos,pre_scale)
    if tst is not None:
        ntst = tst.shape[0]
        tst2 = np.zeros(shape=(ntst,nrows,ncols,nchan),dtype=tst.dtype)
        for i in range(ntst):
            tst2[i] = padsingleimage(tst[i], frame_size, random_pos,pre_scale)
    
    return trn2,val2,tst2

    def create_directory_structure(ylabels,root='.'):
            import os
            labels = np.unique(ylabels)
            trn_root = root+'data/train'
            val_root = root+'data/validation'
            os.makedirs(trn_root)
            os.makedirs(val_root)
            for l in labels:
                os.makedirs(trn_root+'/'+str(l))
                os.makedirs(val_root+'/'+str(l))
            
    def dump_data_to_folders(trn_root, x_train,root='.'):
        for x in range(x_train.shape[0]):
            pass

def test_transfer(dset='mnist', random_seed=9, epochs=10, 
                  data_augmentation=False,
                  batch_size = 512,ntrn=None,ntst=None,mod='focusing'):
    import os
    import numpy as np
    #os.environ['CUDA_VISIBLE_DEVICES']="0"
    #os.environ['TF_FORCE_GPU_ALLOW_GROWTH']="true"
    import keras
    from keras.losses import mse
    from keras.optimizers import SGD, RMSprop
    from keras.datasets import mnist,fashion_mnist, cifar10
    from keras.models import Sequential, Model
    from keras.layers import Input, Dense, Dropout, Flatten,Conv2D, BatchNormalization
    from keras.layers import Activation, Permute,Concatenate,GlobalAveragePooling2D
    from skimage import filters
    from keras import backend as K
    from keras_utils import WeightHistory as WeightHistory
    from keras_utils import RecordVariable, \
    PrintLayerVariableStats, PrintAnyVariable, \
    SGDwithLR, eval_Kdict, standarize_image_025
    from keras_preprocessing.image import ImageDataGenerator
    from Kfocusing import FocusedLayer1D
    
    from keras.engine.topology import Layer
    from keras import activations, regularizers, constraints
    from keras import initializers
    from keras.engine import InputSpec
    import tensorflow as tf
    from keras.applications.inception_v3 import InceptionV3
    #import keras.applications.resnet50 as resnet
    #from keras.applications.resnet50 import preprocess_input
    from keras.applications import VGG16
    from keras.applications.vgg16 import preprocess_input
    
    #Load the VGG model



    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.3
   # Create a session with the above options specified.
    K.tensorflow_backend.set_session(tf.Session(config=config))
    K.clear_session()


    sid = random_seed
 
    
    #test_acnn = True
    
    np.random.seed(sid)
    tf.random.set_random_seed(sid)
    tf.compat.v1.random.set_random_seed(sid)
    
    from datetime import datetime
    now = datetime.now()
    
    if dset=='mnist':
        # input image dimensions
        img_rows, img_cols = 28, 28  
        # the data, split between train and test sets
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        NTRN = ntrn if ntrn else x_train.shape[0]
        NTST = ntst if ntst else x_test.shape[0]
        n_channels=1
        
        lr_dict = {'all':0.1,
                   'focus-1/Sigma:0': 0.01,'focus-1/Mu:0': 0.01,'focus-1/Weights:0': 0.1,
                   'focus-2/Sigma:0': 0.01,'focus-2/Mu:0': 0.01,'focus-2/Weights:0': 0.1}

        mom_dict = {'all':0.9,'focus-1/Sigma:0': 0.9,'focus-1/Mu:0': 0.9,
                   'focus-2/Sigma:0': 0.9,'focus-2/Mu:0': 0.9}
    
        decay_dict = {'all':0.9, 'focus-1/Sigma:0': 0.1,'focus-1/Mu:0':0.1,
                  'focus-2/Sigma:0': 0.1,'focus-2/Mu:0': 0.1}

        clip_dict = {'focus-1/Sigma:0':(0.01,1.0),'focus-1/Mu:0':(0.0,1.0),
                 'focus-2/Sigma:0':(0.01,1.0),'focus-2/Mu:0':(0.0,1.0)}
        
        e_i = x_train.shape[0] // batch_size
        decay_epochs =np.array([e_i*100], dtype='int64')
    
    elif dset=='cifar10':    
        img_rows, img_cols = 32,32
        n_channels=3
        
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        
        NTRN = ntrn if ntrn else x_train.shape[0]
        NTST = ntst if ntst else x_test.shape[0]
        
        lr_dict = {'all':1e-3,
                  'focus-1/Sigma:0': 1e-3,'focus-1/Mu:0': 1e-3,'focus-1/Weights:0':1e-3,
                  'focus-2/Sigma:0': 1e-3,'focus-2/Mu:0': 1e-3,'focus-2/Weights:0': 1e-3,
                  'dense_1/Weights:0':1e-3}
        
        # 1e-3 'all' reaches 91.43 at 250 epochs
        # 1e-3 'all' reached 90.75 at 100 epochs

        mom_dict = {'all':0.9,'focus-1/Sigma:0': 0.9,'focus-1/Mu:0': 0.9,
                   'focus-2/Sigma:0': 0.9,'focus-2/Mu:0': 0.9}
        #decay_dict = {'all':0.9}
        decay_dict = {'all':0.9, 'focus-1/Sigma:0': 0.9,'focus-1/Mu:0':0.9,
                      'focus-2/Sigma:0': 0.9,'focus-2/Mu:0': 0.9}

        clip_dict = {'focus-1/Sigma:0':(0.01,1.0),'focus-1/Mu:0':(0.0,1.0),
                 'focus-2/Sigma:0':(0.01,1.0),'focus-2/Mu:0':(0.0,1.0)}
        
        e_i = NTRN // batch_size
        
        #decay_epochs =np.array([e_i*10], dtype='int64') #for 20 epochs
        #decay_epochs =np.array([e_i*10,e_i*80,e_i*120,e_i*160], dtype='int64')
        decay_epochs =np.array([e_i*10, e_i*80, e_i*120, e_i*180], dtype='int64')
    
    num_classes = np.unique(y_train).shape[0]
    
    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], n_channels, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], n_channels, img_rows, img_cols)
        input_shape = (n_channels, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, n_channels)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, n_channels)
        input_shape = (img_rows, img_cols, n_channels)

   
    

    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, n_channels)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, n_channels)
    input_shape = (img_rows, img_cols, n_channels)
    
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    #FRAME_SIZE=(299,299,3)
    FRAME_SIZE = (224,224,3)

    idx = np.random.permutation(x_train.shape[0])  
    x_train = x_train[idx[0:NTRN]]
    y_train = y_train[idx[0:NTRN]]
    idx = np.random.permutation(x_test.shape[0])
    x_test = x_test[idx[0:NTST]]
    y_test = y_test[idx[0:NTST]]
    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    #x_train, _, x_test = paddataset(x_train,None,x_test,FRAME_SIZE,False,5)
    
    
    #x_train, _, x_test = standarize_image_025(x_train,tst=x_test)
    x_train=preprocess_input(x_train)
    x_test=preprocess_input(x_test)
    import matplotlib.pyplot as plt
    plt.imshow(x_train[0])
    print(np.max(x_train[0]),np.mean(x_train[0]))
    plt.show()
    plt.imshow(x_test[0])
    print(np.max(x_train[0]),np.mean(x_train[0]))
    plt.show()
    
    print(x_train.shape, 'train samples')
    print(np.mean(x_train))
    print(np.var(x_train))
    
    print(x_test.shape, 'test samples')
    print(np.mean(x_test))
    print(np.var(x_test))
    
   

    # create the base pre-trained model
    base_in = Input(shape=input_shape, name='inputlayer')
    
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape,
                      input_tensor=base_in)
    x=base_model.output
    x = GlobalAveragePooling2D()(x)
    
    pad_input =True
    if pad_input:
        print("PADDING LAYER OUPUT")
        
        paddings = tf.constant([[0, 0,], [3, 3]])
    
        padding_layer = keras.layers.Lambda(lambda x: tf.pad(x,paddings,"CONSTANT"))
        x = padding_layer(x)
    #x = Dropout(0.1)(x)
    # let's add a fully-connected layer
    focusing=mod=='focused'
    if focusing:
        nf = 40#init_sigma=np.exp(-(np.linspace(0.1, 0.9, nf)-0.5)**2/0.07),
        x = FocusedLayer1D(units=nf,
                           name='focus-1',
                           activation='linear',
                           init_sigma=0.08,
                           init_mu='spread',
                           init_w= None,
                           train_sigma=True,
                           train_weights=True,
                           train_mu = True,
                           si_regularizer=None,
                           normed=2)(x)
    elif mod=='dense':
        x = Dense(40, activation='linear')(x)
    else:
        print('unknown mod')
        return
        
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.2)(x)
    # and a logistic layer -- let's say we have 200 classes
    predictions = Dense(10, activation='softmax')(x)
    model = Model(inputs=base_in, outputs=[predictions])
    # opt= SGDwithLR(lr_dict, mom_dict,decay_dict,clip_dict, decay_epochs)#, decay=None)
    optimizer_s = 'SGDwithLR'
    if optimizer_s == 'SGDwithLR':
        opt = SGDwithLR(lr_dict, mom_dict,decay_dict,clip_dict, decay_epochs)#, decay=None)
    elif optimizer_s=='RMSprob':
        opt = RMSprop(lr=0.01, rho=0.9, epsilon=None, decay=0.0)
    else:
    # opt= SGDwithLR({'all': 0.01},{'all':0.9})#, decay=None)
        opt= SGD(lr=0.01, momentum=0.9)#, decay=None)
    
    # compile the model (should be done *after* setting layers to non-trainable)
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=opt,
                  metrics=['accuracy'])
        
    model.summary()
    
    stat_func_name = ['max: ', 'mean: ', 'min: ', 'var: ', 'std: ']
    stat_func_list = [np.max, np.mean, np.min, np.var, np.std]
    
    callbacks = []
    
    if focusing:
        pr_1 = PrintLayerVariableStats("focus-1","Weights:0",stat_func_list,stat_func_name)
        pr_2 = PrintLayerVariableStats("focus-1","Sigma:0",stat_func_list,stat_func_name)
        pr_3 = PrintLayerVariableStats("focus-1","Mu:0",stat_func_list,stat_func_name)
        callbacks+=[pr_1,pr_2,pr_3]
    
    recordvariables=False
    if recordvariables:
            
        rv_weights_1 = RecordVariable("focus-1","Weights:0")
        rv_sigma_1 = RecordVariable("focus-1","Sigma:0")
        callbacks+=[rv_weights_1,rv_sigma_1]

    if optimizer_s =='SGDwithLR': 
        print_lr_rates_callback = keras.callbacks.LambdaCallback(
            on_epoch_end=lambda epoch, logs: print("iter: ", 
                                                   K.eval(model.optimizer.iterations),
                                                   " LR RATES :", 
                                                   eval_Kdict(model.optimizer.lr)))
    
        callbacks.append(print_lr_rates_callback)
    
        
    if not data_augmentation:
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
            width_shift_range=0.2,
            # randomly shift images vertically
            height_shift_range=0.2,
            # set range for random shear
            shear_range=0.1,
            # set range for random zoom
            zoom_range=0.1,
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
        
        #x_test,_,_ = paddataset(x_test,None, None,frame_size=FRAME_SIZE, random_pos=False)
        # Fit the model on the batches generated by datagen.flow().
        history=model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                            validation_data=(x_test, y_test),
	                   workers=4, use_multiprocessing=False,epochs=epochs, verbose=2,
                            callbacks=callbacks, 
                            steps_per_epoch=x_train.shape[0]//batch_size)
    
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    return score,history,model

def repeated_ttrials(dset='mnist',N=1, epochs=2, augment=False,
                    batch_size=32,delayed_start=0,
                    ntrn=None, ntst=None,random_seed=17,mod='focused'):
    
    list_scores =[]
    list_histories =[]
    models = []
    
    import time 
    from shutil import copyfile
    print("Delayed start ",delayed_start)
    time.sleep(delayed_start)
    from datetime import datetime
    now = datetime.now()
    timestr = now.strftime("%Y%m%d-%H%M%S")
    
    filename = 'outputs/transfer-'+dset+'/'+timestr+'_'+mod+'.model_results.npz'
    #copyfile("Kfocusing.py",filename+"code.py")
               
    for i in range(N):
        print("Training with ", ntrn," samples")
        sc,hs,ms = test_transfer(dset=dset,epochs=epochs,
                                 data_augmentation=augment,
                                 batch_size=batch_size,
                                 ntrn=ntrn,ntst=ntst,
                                 random_seed=random_seed*i,mod=mod)
        list_scores.append(sc)
        list_histories.append(hs)
        models.append([ms])
    
    
    print("Final scores", list_scores,)
    mx_scores = [np.max(list_histories[i].history['val_acc']) for i in range(len(list_histories))]
    print("Max sscores", mx_scores)
    np.savez_compressed(filename,mx_scores =mx_scores, list_scores=list_scores)
    return mx_scores, list_scores, list_histories


import os
os.environ['CUDA_VISIBLE_DEVICES']="0"
os.environ['TF_FORCE_GPU_ALLOW_GROWTH']="true"

mod='focused'
#mod='focused'
# ntrn limits the number of training samples
#ntrn =[250,1000,10000] #
ntrn=[None]
N=5 # number of repeats for each ntrn[r]
ls_mx_sc =[]
ls_ls_sc =[]
ls_hs = []
import time 
from shutil import copyfile
delayed_start=0*3600
print("Delayed start ",delayed_start)
time.sleep(delayed_start)
from datetime import datetime
then = datetime.now()
for r in range(len(ntrn)):
    mx_sc,ls_sc,ls_hs = repeated_ttrials(dset='cifar10',N=N,
                                        epochs=250, 
                                        augment=True,delayed_start = 0,                         
                                        batch_size=32, 
                                        ntrn=ntrn[r],ntst=10000,
                                        random_seed = 17*r, mod=mod)
    ls_mx_sc.append(mx_sc)
    ls_ls_sc.append(ls_sc)
    ls_hs.append(ls_hs)


from datetime import datetime
now = datetime.now()
timestr = then.strftime("%Y%m%d-%H%M%S")+now.strftime("%H%M%S")
    
filename = 'outputs/transfer-cifar10/'+'experiment'+timestr+'_'+mod+'_n_results.npz'
np.savez_compressed(filename,ls_mx_sc =ls_mx_sc, ls_ls_sc=ls_ls_sc, ls_hs=ls_hs)


summarize_results = False
if summarize_results:
    import numpy as np
    from scipy.stats import ttest_ind
    root = 'outputs/transfer-cifar10/'
    # paper results below.
    f1 = np.load(root+'experiment20190916-171726052929_focused_n_results.npz')['ls_mx_sc']
    d1 = np.load(root+'experiment20190917-021154140905_dense_n_results.npz')['ls_mx_sc']

    #d1=np.load(root+'experiment20190826-114828095952_densen_results.npz')['ls_mx_sc']
    #f1=np.load(root+'experiment20190905-133509041531_focusedn_results.npz')['ls_mx_sc']
    
    fs=f1
    ds=d1
    print("fs:",fs)
    print("ds:",ds)
    #fs = np.concatenate((f1,f2,f3))
    #ds = np.concatenate((d1,d2))
    print("Focus mean",100*np.mean(fs,axis=1))
    print("Focus std",100*np.std(fs,axis=1))
    print("Focus max",100*np.max(fs,axis=1))
    
    print("Dense mean",100*np.mean(ds,axis=1))
    print("Dense std",100*np.std(ds,axis=1))
    print("Dense max",100*np.max(ds,axis=1))
    
    
    ttest_ind(fs,ds,axis=1)


# paper result ~2 November
# uses extended augmentation
# Focus [0.9184, 0.9163, 0.9176, 0.919, 0.9187]
#Dense [91.61,91.92,91.79]

    
'''
# plots results
ntrnr= ntrn[np.newaxis,:].repeat(R,axis=0)

import matplotlib.pyplot as plt
import seaborn as sns
plt.figure()
#sns.lineplot(x='N',y='Dense',data=a)
#plt.loglog(a['N'],a['Dense'],'r+')
#plt.loglog(a['N'],a['Focus'],'go')
#plt.loglog(a['N'],a['Focus']-a['Dense'],'bs')
#plt.show()

plt.figure()
plt.errorbar(ntrn,np.mean(ls_mx_dense,axis=0),yerr=np.std(ls_mx_dense,axis=0),color='r', marker='+')
plt.errorbar(ntrn,np.mean(ls_mx_focus,axis=0),yerr=np.std(ls_mx_focus,axis=0),color='g', marker='o')
#plt.xlim([0,5000])
#plt.ylim([0.5,1.0])
plt.yscale('log')

plt.figure()
plt.plot(ntrn,np.mean(ls_mx_dense,axis=0)-np.mean(ls_mx_focus,axis=0),color='r', marker='+')
'''
