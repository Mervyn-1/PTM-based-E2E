CUDA_VISIBLE_DEVICES=4,5 python ../../../learn.py\
    --data_path_prefix ../../../../../data/dataset/multiwoz/MultiWOZ_2.1\
    --model_name t5-small\
    --pretrained_path ../../../checkpoints/small/\
    --ckpt_save_path ../../../ckpt/small/full_training/\
    --epoch_num 60\
    --gradient_accumulation_steps 2\
    --number_of_gpu 2\
    --batch_size_per_gpu 16