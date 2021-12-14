CUDA_VISIBLE_DEVICES=0 python ../../../inference_pptod.py\
    --data_path_prefix data/dataset/multiwoz20/\
    --model_name t5-base\
    --pretrained_path model/pptod/ckpt/base/full_training/\
    --output_save_path ../../../inference_result/base/full_training/\
    --number_of_gpu 1\
    --batch_size_per_gpu 32