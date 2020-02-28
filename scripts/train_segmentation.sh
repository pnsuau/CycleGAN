gpu='1'

######################
# loss weight params #
######################
lr=5e-6
momentum=0.99
lambda_d=1
lambda_g=0.1

################
# train params #
################
batch='2'
crop='256'
load='256'


########
# Data #
########
src='/data1/pnsuau/cityscapes2gta/'
name='segmentation_test_1'
dataset_mode='unaligned_labeled_mask'
#max_dataset_size='1'


output_nc='3'
input_nc='3'
nclasses='8'
# init with pre-trained cyclegta5 model
#model='cycle_gan_semantic_mask'
model='segmentation'



#base_model="base_models/${model}-${src}-iter${baseiter}.pth"
#outdir="${resdir}/${model}/lr${lr}_crop${crop}_ld${lambda_d}_lg${lambda_g}_momentum${momentum}"

display_freq=1000

# Run python script #
#CUDA_VISIBLE_DEVICES=${gpu}
python3 ../train.py \
    --dataroot ${src} --name $name \
    --gpu ${gpu} \
    --dataset_mode ${dataset_mode} \
    --model ${model} --semantic_nclasses ${nclasses} \
    --crop_size ${crop} --load_size ${load} --batch_size ${batch} \
    --save_epoch_freq 10\
    --no_flip --input_nc ${input_nc} --output_nc ${output_nc} --display_freq ${display_freq} --continue_train --epoch_count 9




