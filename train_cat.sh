if [ $2 == "pretrain" ]; then
    PRE_TRAIN_MODEL="--pretrain_model_path /home/linghuxiongkun/workspace/models/ffhq_10m.pt"
else
    PRE_TRAIN_MODEL="--pretrain_model_path 1"
fi
MODEL_FLAGS="--attention_resolutions 16 --class_cond False
--diffusion_steps 1000 --image_size 256 --learn_sigma True
--noise_schedule linear --num_channels 128 --num_head_channels 64
--num_res_blocks 1 --resblock_updown True --use_fp16 False
--use_scale_shift_norm True"
DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule linear"
TRAIN_FLAGS="--lr 1e-4 --batch_size $3"
python -m ipdb scripts/image_train.py \
--data_dir /home/linghuxiongkun/workspace/guided-diffusion/datasets/cat$1 \
--log_dir /home/linghuxiongkun/workspace/guided-diffusion/output_debug2/cat$1/$2_bs$3_randomcrop$4_randomflip$5 \
--log_interval 100 \
--save_interval 50000 \
--random_crop $4 \
--random_flip $5 \
--instance_id $6 \
$MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS $PRE_TRAIN_MODEL
    