MODEL_FLAGS="--attention_resolutions 16 --class_cond False
--diffusion_steps 1000 --image_size 256 --learn_sigma True
--noise_schedule linear --num_channels 128 --num_head_channels 64
--num_res_blocks 1 --resblock_updown True --use_fp16 False
--use_scale_shift_norm True"
# MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond False
# --diffusion_steps 1000 --image_size 256 --learn_sigma True
# --noise_schedule linear --num_channels 256 --num_head_channels 64
# --num_res_blocks 2 --resblock_updown True --use_fp16 False
# --use_scale_shift_norm True"
SAMPLE_FLAGS="--batch_size 32 --num_samples 100 --timestep_respacing ddim25 --use_ddim True"
SAVE_NAME="$2_epoch$4_ft_type$2bs$3_randomcrop$5_randomflip$6"
# SAVE_NAME="$2_epoch$4_ft_type$2"
SCALE="$1"
python scripts/image_sample.py $MODEL_FLAGS \
--K $7 \
--output_dir /home/linghuxiongkun/workspace/guided-diffusion/sampling_output_debug2/scale$SCALE/$SAVE_NAME \
--model_path /home/linghuxiongkun/workspace/guided-diffusion/output_debug2/cat$1/$2_bs$3_randomcrop$5_randomflip$6/ema_0.9999_$4.pt $SAMPLE_FLAGS
# --model_path /home/linghuxiongkun/workspace/guided-diffusion/output/cat$1/$2_bs$3_randomcrop$5_randomflip$6_structure2/ema_0.9999_$4.pt $SAMPLE_FLAGS
# --model_path /home/linghuxiongkun/workspace/guided-diffusion/output/cat$1/$2_bs$3_randomcrop$5_randomflip$6/ema_0.9999_$4.pt $SAMPLE_FLAGS