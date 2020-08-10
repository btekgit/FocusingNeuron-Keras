export CUDA_VISIBLE_DEVICES=0
export TF_FORCE_GPU_ALLOW_GROWTH="true"
#kwargs = {'dset':'mnist', 'neuron':'focused', 'cnn_model':False, 'nhidden':(800,800), 
#         'nfilters':(32,32), 'repeats':5, 'epochs':200, 'batch_size':512,
#         'lr_all':0.1, 'augment':False, 'delay':0,'kn_size':(5,5),
#         'focus_init_sigma':0.025, 'focus_init_mu':'spread','focus_train_mu':True, 
#         'focus_train_si':True,'focus_norm_type':2,
#         'focus_sigma_reg':None, 'ex_name':''}
# sigma_reg_set = (1e-10, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1)

# note aug 2020
# lrmul 0.5 batch 512: init_sigma=0.025 augment=false, norm=2, 
# spread, epochs 200 78.90
# set focus_sigma_reg=1e-10 for 79.something

dset=cifar10
neuron=focused
cnn_model=True
nhidden='(256,)'
nfilters='(32,32)'
repeats=5
epochs=200
batch_size=512
lr_mul=0.5 # 0.5 batch 512: init_sigma=0.025 78.90
#lr_mul=1.0
augment=False
delay=0
kn_size='(5,5)'
focus_init_sigma=0.025
focus_init_mu=spread
focus_train_mu=True
focus_train_si=True
focus_norm_type=2
focus_sigma_reg=1e-10


nhidden='(256,)'
ex_name=test_cifar_cnn_lrmu05_isi_0025_b512_9_aug_2020_sigma_reg
python Kfocusing.py $dset $neuron $cnn_model $nhidden $nfilters $repeats $epochs $batch_size $lr_mul $augment $delay $kn_size $focus_init_sigma\  $focus_init_mu $focus_train_mu $focus_train_si $focus_norm_type $focus_sigma_reg $ex_name
#python Kfocusing.py $dset dense $cnn_model $nhidden $nfilters $repeats $epochs $batch_size $lr_mul $augment $delay $kn_size $focus_init_sigma\  $focus_init_mu $focus_train_mu $focus_train_si $focus_norm_type $focus_sigma_reg $ex_name
