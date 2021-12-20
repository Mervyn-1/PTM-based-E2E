CUDA_VISIBLE_DEVICES=6,7 python ../../../learn.py\
    --data_path_prefix ../../../../../data/dataset/multiwoz/MultiWOZ_2.1\
    --model_name t5-base\
    --pretrained_path ../../../checkpoints/base/\
    --ckpt_save_path ../../../ckpt/base/full_training/\
    --epoch_num 60\
    --gradient_accumulation_steps 16\
    --number_of_gpu 2\
    --batch_size_per_gpu 8\
    --only_use_PLM