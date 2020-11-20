export CUDA_VISIBLE_DEVICES=0
export TF_FORCE_GPU_ALLOW_GROWTH="true"
#kwargs = {'dset':'mnist', 'neuron':'focused', 'cnn_model':False, 'nhidden':(800,800), 
#         'nfilters':(32,32), 'repeats':5, 'epochs':200, 'batch_size':512,
#         'lr_mul':0.1, 'augment':False, 'delay':0,'kn_size':(5,5),
#         'focus_init_sigma':0.025, 'focus_init_mu':'spread','focus_train_mu':True, 
#         'focus_train_si':True,'focus_norm_type':2,
#         'focus_sigma_reg':None, 'ex_name':''}
# sigma_reg_set = (1e-10, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1)
dset='mnist-clut'
neuron=focused
cnn_model=False
nhidden='(800,800)'
nfilters='(32,32)'
repeats=5
epochs=200
batch_size=512
lr_mul=1.0
augment=False
delay=0
kn_size='(5,5)'
focus_init_sigma=0.025
focus_init_mu=spread
focus_train_mu=True
focus_train_si=True
focus_norm_type=2
focus_sigma_reg=None




nhidden='(800,800)'
ex_name=test_mnist_clut_lr001_glorot_uni_nhidden_800_29_july_2020                  
python Kfocusing.py $dset $neuron $cnn_model $nhidden $nfilters $repeats $epochs $batch_size $lr_mul $augment $delay $kn_size $focus_init_sigma\  $focus_init_mu $focus_train_mu $focus_train_si $focus_norm_type $focus_sigma_reg $ex_name




