CUDA_VISIBLE_DEVICES=0,1,2,3 python ../../../learn.py\
    --dataset_name MultiWOZ_2.0\
    --model_name t5-base\
    --pretrained_path ../../../../checkpoints/base/\
    --ckpt_save_path ../../../ckpt/t5_base/full_training/\
    --epoch_num 60\
    --gradient_accumulation_steps 16\
    --number_of_gpu 4\
    --batch_size_per_gpu 2\
    --only_use_PLM