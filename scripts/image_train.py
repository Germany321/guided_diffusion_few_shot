"""
Train a diffusion model on images.
"""

import argparse
import sys
import os
sys.path.append("/home/linghuxiongkun/workspace/guided-diffusion")
from guided_diffusion import dist_util, logger
from guided_diffusion.image_datasets import load_data
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.train_util import TrainLoop
from IPython import embed


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure(format_strs=args.log_format_strs, dir=args.log_dir)

    argsDict = args.__dict__
    with open(os.path.join(args.log_dir, 'config.txt'), 'w') as f:
        f.writelines('-----------------config-----------------'+'\n')
        for arg, val in argsDict.items():
            f.writelines(arg + ' : ' + str(val) + '\n')

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )


    if len(args.pretrain_model_path) > 1:
        model.load_state_dict(
            dist_util.load_state_dict(args.pretrain_model_path, map_location="cpu")
        )
    # import ipdb
    # ipdb.set_trace()

    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")
    data = load_data(
        random_crop=args.random_crop,
        random_flip=args.random_flip,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
        instance_id=args.instance_id,
    )

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        ft_type=args.ft_type,
    ).run_loop()


def create_argparser():
    defaults = dict(
        data_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        save_interval=10000,
        resume_checkpoint="",
        pretrain_model_path="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        log_format_strs="json,log,tensorboard",
        log_dir="",
        ft_type="",
        random_flip=False,
        random_crop=False,
        instance_id=False
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()

    add_dict_to_argparser(parser, defaults)

    return parser


if __name__ == "__main__":
    main()
