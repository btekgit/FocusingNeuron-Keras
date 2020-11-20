export CUDA_VISIBLE_DEVICES=0
export TF_FORCE_GPU_ALLOW_GROWTH="true"
#kwargs = {'dset':'mnist', 'neuron':'focused', 'cnn_model':False, 'nhidden':(800,800), 
#         'nfilters':(32,32), 'repeats':5, 'epochs':200, 'batch_size':512,
#         'lr_mul':0.1, 'augment':False, 'delay':0,'kn_size':(5,5),
#         'focus_init_sigma':0.025, 'focus_init_mu':'spread','focus_train_mu':True, 
#         'focus_train_si':True,'focus_norm_type':2,
#         'focus_sigma_reg':None, 'ex_name':''}
# sigma_reg_set = (1e-10, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1)
dset=lfw_faces
neuron=focused
cnn_model=False
nfilters='(32,32)'
repeats=7
epochs=200
batch_size=32
lr_mul=1.0
augment=True
delay=0
kn_size='(5,5)'
focus_init_sigma=0.025
focus_init_mu=spread
focus_train_mu=True
focus_train_si=True
focus_norm_type=2
focus_sigma_reg=None
# PLAIN NETWORK
# focus-s
nhidden='(784,784)'
ex_name=test_faces_nhidden_fixed_vs_focus_s_784_july_28_2020_focus_s_run                  
python Kfocusing.py $dset $neuron $cnn_model $nhidden $nfilters $repeats $epochs $batch_size $lr_mul $augment $delay $kn_size $focus_init_sigma\  $focus_init_mu $focus_train_mu $focus_train_si $focus_norm_type $focus_sigma_reg $ex_name

# focus-c
focus_init_mu=middle
focus_train_mu=True
focus_train_si=True
focus_norm_type=2
ex_name=test_faces_nhidden_fixed_vs_focus_s_784_july_28_2020_focus_c_run
python Kfocusing.py $dset $neuron $cnn_model $nhidden $nfilters $repeats $epochs $batch_size $lr_mul $augment $delay $kn_size $focus_init_sigma\  $focus_init_mu $focus_train_mu $focus_train_si $focus_norm_type $focus_sigma_reg $ex_name


focus_init_mu=spread
focus_train_mu=False
focus_train_si=False
focus_norm_type=2
focus_init_sigma=0.1
ex_name=test_faces_nhidden_fixed_vs_focus_s_784_july_28_2020_fixed_s_run
python Kfocusing.py $dset $neuron $cnn_model $nhidden $nfilters $repeats $epochs $batch_size $lr_mul $augment $delay $kn_size $focus_init_sigma\  $focus_init_mu $focus_train_mu $focus_train_si $focus_norm_type $focus_sigma_reg $ex_name




# CNN NETWORK
cnn_model=True
nhidden='(256)'
lr_mul=0.01

ex_name=test_faces_nhidden_fixed_vs_focus_s_784_july_28_2020_focus_s_run                  
python Kfocusing.py $dset $neuron $cnn_model $nhidden $nfilters $repeats $epochs $batch_size $lr_mul $augment $delay $kn_size $focus_init_sigma\  $focus_init_mu $focus_train_mu $focus_train_si $focus_norm_type $focus_sigma_reg $ex_name
neuron=dense
python Kfocusing.py $dset $neuron $cnn_model $nhidden $nfilters $repeats $epochs $batch_size $lr_mul $augment $delay $kn_size $focus_init_sigma\  $focus_init_mu $focus_train_mu $focus_train_si $focus_norm_type $focus_sigma_reg $ex_name
