CUDA_VISIBLE_DEVICES=2,3 python ../../../inference_pptod.py\
    --data_path_prefix ../../../../data/multiwoz/MultiWOZ_2.0/\
    --model_name t5-small\
    --pretrained_path ../../../ckpt/small/full_training/\
    --output_save_path ../../../inference_result/small/full_training/\
    --number_of_gpu 2\
    --batch_size_per_gpu 64