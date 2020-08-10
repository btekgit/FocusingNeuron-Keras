export CUDA_VISIBLE_DEVICES=0
export TF_FORCE_GPU_ALLOW_GROWTH="true"
#kwargs = {'dset':'mnist', 'neuron':'focused', 'cnn_model':False, 'nhidden':(800,800), 
#         'nfilters':(32,32), 'repeats':5, 'epochs':200, 'batch_size':512,
#         'lr_mul':0.1, 'augment':False, 'delay':0,'kn_size':(5,5),
#         'focus_init_sigma':0.025, 'focus_init_mu':'spread','focus_train_mu':True, 
#         'focus_train_si':True,'focus_norm_type':2,
#         'focus_sigma_reg':None, 'ex_name':''}
# sigma_reg_set = (1e-10, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1)
dset=mnist
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



nhidden='(64,64)'
epochs=200
ex_name=test_nhidden_64_july_25_2020                  
python Kfocusing.py $dset $neuron $cnn_model $nhidden $nfilters $repeats $epochs $batch_size $lr_mul $augment $delay $kn_size $focus_init_sigma\  $focus_init_mu $focus_train_mu $focus_train_si $focus_norm_type $focus_sigma_reg $ex_name
python Kfocusing.py $dset dense $cnn_model $nhidden $nfilters $repeats $epochs $batch_size $lr_mul $augment $delay $kn_size $focus_init_sigma\  $focus_init_mu $focus_train_mu $focus_train_si $focus_norm_type $focus_sigma_reg $ex_name

nhidden='(128,128)'
ex_name=test_nhidden_128_july_25_2020                  
python Kfocusing.py $dset $neuron $cnn_model $nhidden $nfilters $repeats $epochs $batch_size $lr_mul $augment $delay $kn_size $focus_init_sigma\  $focus_init_mu $focus_train_mu $focus_train_si $focus_norm_type $focus_sigma_reg $ex_name
python Kfocusing.py $dset dense $cnn_model $nhidden $nfilters $repeats $epochs $batch_size $lr_mul $augment $delay $kn_size $focus_init_sigma\  $focus_init_mu $focus_train_mu $focus_train_si $focus_norm_type $focus_sigma_reg $ex_name

nhidden='(256,256)'
ex_name=test_nhidden_256_july_25_2020                  
python Kfocusing.py $dset $neuron $cnn_model $nhidden $nfilters $repeats $epochs $batch_size $lr_mul $augment $delay $kn_size $focus_init_sigma\  $focus_init_mu $focus_train_mu $focus_train_si $focus_norm_type $focus_sigma_reg $ex_name
python Kfocusing.py $dset dense $cnn_model $nhidden $nfilters $repeats $epochs $batch_size $lr_mul $augment $delay $kn_size $focus_init_sigma\  $focus_init_mu $focus_train_mu $focus_train_si $focus_norm_type $focus_sigma_reg $ex_name

nhidden='(512,512)'
ex_name=test_nhidden_512_july_25_2020                  
python Kfocusing.py $dset $neuron $cnn_model $nhidden $nfilters $repeats $epochs $batch_size $lr_mul $augment $delay $kn_size $focus_init_sigma\  $focus_init_mu $focus_train_mu $focus_train_si $focus_norm_type $focus_sigma_reg $ex_name
python Kfocusing.py $dset dense $cnn_model $nhidden $nfilters $repeats $epochs $batch_size $lr_mul $augment $delay $kn_size $focus_init_sigma\  $focus_init_mu $focus_train_mu $focus_train_si $focus_norm_type $focus_sigma_reg $ex_name

nhidden='(800,800)'
ex_name=test_nhidden_800_july_25_2020                  
python Kfocusing.py $dset $neuron $cnn_model $nhidden $nfilters $repeats $epochs $batch_size $lr_mul $augment $delay $kn_size $focus_init_sigma\  $focus_init_mu $focus_train_mu $focus_train_si $focus_norm_type $focus_sigma_reg $ex_name
python Kfocusing.py $dset dense $cnn_model $nhidden $nfilters $repeats $epochs $batch_size $lr_mul $augment $delay $kn_size $focus_init_sigma\  $focus_init_mu $focus_train_mu $focus_train_si $focus_norm_type $focus_sigma_reg $ex_name

nhidden='(1024,1024)'
ex_name=test_nhidden_1024_july_25_2020                  
python Kfocusing.py $dset $neuron $cnn_model $nhidden $nfilters $repeats $epochs $batch_size $lr_mul $augment $delay $kn_size $focus_init_sigma\  $focus_init_mu $focus_train_mu $focus_train_si $focus_norm_type $focus_sigma_reg $ex_name
python Kfocusing.py $dset dense $cnn_model $nhidden $nfilters $repeats $epochs $batch_size $lr_mul $augment $delay $kn_size $focus_init_sigma\  $focus_init_mu $focus_train_mu $focus_train_si $focus_norm_type $focus_sigma_reg $ex_name

# results with plt_Kfocusing_results.py
# plot_graphs(mnist_200_dec_09_lists,ylims=[0.97,1.0050] )
#dense_64 : 0.9805000005722045 0.9796200005149842 0.0004874422870610656
#focused_64 : 0.9803000006675721 0.9794400005912781 0.0007255343111260256
#Folder results: foc [0.9801 0.9796 0.9803 0.9787 0.9785]  dense : [0.9793 0.9793 0.9805 0.9792 0.9798]
#Ttest_indResult(statistic=0.4118657058424029, pvalue=0.6912521262230584)
#dense_128 : 0.987299999332428 0.9866599994277954 0.0004409081234192891
#focused_128 : 0.9878999996185303 0.9873599994659423 0.00036660617942119774
#Folder results: foc [0.987  0.9871 0.9877 0.9879 0.9871]  dense : [0.987  0.9866 0.9863 0.9861 0.9873]
#Ttest_indResult(statistic=-2.441530267564615, pvalue=0.04046779929998762)
#dense_256 : 0.9889999995231629 0.9887599995613098 0.000257681980456089
#focused_256 : 0.9913999997138977 0.9909799995994568 0.0003655133992255896
#Folder results: foc [0.9914 0.9909 0.9905 0.9914 0.9907]  dense : [0.989  0.9887 0.9888 0.989  0.9883]
#Ttest_indResult(statistic=-9.928140797702275, pvalue=8.957511998692913e-06)
#dense_512 : 0.9900999995231629 0.9897599995422363 0.00027276366470754083
#focused_512 : 0.9923999997138977 0.9920199997138978 0.00027129322041675086
#Folder results: foc [0.9919 0.9922 0.9916 0.9924 0.992 ]  dense : [0.9895 0.99   0.9894 0.9901 0.9898]
#Ttest_indResult(statistic=-11.749180306902598, pvalue=2.517892139735151e-06)

#dense_800 : 0.9902999997138977 0.9898199996185303 0.00046647620065376645
#focused_800 : 0.9927999997138977 0.9923799997329711 0.00035999998516505246
#Folder results: foc [0.9926 0.9928 0.9926 0.9919 0.992 ]  dense : [0.9899 0.9895 0.9903 0.9903 0.9891]
#Ttest_indResult(statistic=-8.68920613554924, pvalue=2.3976503448486637e-05)

#dense_1024 : 0.9902999995231628 0.990019999599457 0.00023151675618050843
#focused_1024 : 0.9922999996185303 0.9920599997329713 0.00011999994277956596
#Folder results: foc [0.992  0.992  0.9923 0.992  0.992 ]  dense : [0.9903 0.99   0.9901 0.9896 0.9901]
#Ttest_indResult(statistic=-15.646087410897682, pvalue=2.7776794831716005e-07)