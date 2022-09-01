MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond False
--diffusion_steps 1000 --image_size 256 --learn_sigma True
--noise_schedule linear --num_channels 256 --num_head_channels 64
--num_res_blocks 2 --resblock_updown True --use_fp16 True
--use_scale_shift_norm True"
DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule linear"
TRAIN_FLAGS="--lr 1e-4 --batch_size 16"
python -m ipdb scripts/image_train.py \
--data_dir /home/linghuxiongkun/workspace/SNGAN-image-generation/data/anime \
--pretrain_model_path /home/linghuxiongkun/workspace/guided-diffusion/models/256x256_diffusion_uncond.pt \
--log_dir /home/linghuxiongkun/workspace/guided-diffusion/output/debug \
--log_interval 1 \
--save_interval 50 \
$MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS
    