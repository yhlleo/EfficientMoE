
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os

from diffusion import create_diffusion
from diffusion.rectified_flow import RectifiedFlow
from diffusers.models import AutoencoderKL
from download import find_model

from utils import update_ema, requires_grad, cleanup, setup_ddp0, setup_exp_dir, setup_data, instantiate_from_config, get_lr_scheduler_config, wandb_initialize, wandb_log
from omegaconf import OmegaConf

from torch.nn.utils import clip_grad_norm_

def main(config, use_wandb=True, resume_path=None):
    config = OmegaConf.load(config)
    rank, device, seed = setup_ddp0(config.basic)
    #rank, world_size, local_rank, device = setup_ddp(config.basic)
    logger, writer, checkpoint_dir = setup_exp_dir(rank, config)
    
    model = instantiate_from_config(config.model)
    if config.model.ckpt is not None:
        state_dict = find_model(config.model.ckpt, is_train=True)
        model.load_state_dict(state_dict)
        logger.info(f"load model ckpt from {config.model.ckpt}")
    logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # resume from previous training steps
    cur_epoch = 0
    cur_steps = 0
    if resume_path:
        if os.path.exists(resume_path):
            basename = os.path.basename(resume_path)
            cur_steps = int(basename.split(".")[0])

            cur_epoch = int(cur_steps // 2500)
            ckpt = torch.load(resume_path, weights_only=False, map_location="cpu")["model"]
            model.load_state_dict(ckpt)
            print(f"Loaded pretrained model from: {resume_path} ...")

    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    requires_grad(ema, False)
    model = DDP(model.to(device), device_ids=[rank], find_unused_parameters=True)

    if 'rf' not in config.basic:
        config.basic.rf = False  

    if config.basic.rf:
        logger.info("train with rectified flow")
        diffusion = RectifiedFlow(model)
    else:
        diffusion = create_diffusion(timestep_respacing="")  # default: 1000 steps, linear noise schedule


    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    opt = torch.optim.AdamW(model.parameters(),
                            lr=config.optim.base_learning_rate,
                            weight_decay=config.optim.weight_decay,
                            betas=config.optim.betas,
                            )
                            
    max_grad_norm = config.basic.clip_grad_norm


    # Setup warm-up and training lr scheduler
    warmup_steps = -1
    if config.lr_sheduler.get("warmup", None):
        lr_sheduler_warmup_config = get_lr_scheduler_config(config.lr_sheduler.warmup, opt)
        lr_sheduler_warmup = instantiate_from_config(lr_sheduler_warmup_config)
        warmup_steps = config.lr_sheduler.warmup.params.warmup_steps

    use_epoch_lr_scheduler = False
    if config.lr_sheduler.get("train_epoch", None):
        milestones = []
        for milestone in config.lr_sheduler.train_epoch.params["milestones"]:
            milestones.append(int(milestone*config.basic.epochs))
        config.lr_sheduler.train_epoch.params["milestones"] = milestones

        lr_sheduler_train_epoch_config = get_lr_scheduler_config(config.lr_sheduler.train_epoch, opt)
        lr_sheduler_train_epoch = instantiate_from_config(lr_sheduler_train_epoch_config)
        use_epoch_lr_scheduler = True


    dataset, sampler, loader = setup_data(rank, config.basic)
    logger.info(f"Dataset contains {len(dataset):,} images ({config.basic.data_path})")

    # VAE
    use_latent = hasattr(dataset, "use_latent")
    if not use_latent:
        logger.info(f"Using VAE Online Processing Latent")
        vae = AutoencoderKL.from_pretrained(config.basic.vae_path).to(device)
    else:
        logger.info(f"Using offline ProcessedLatent")
    # Prepare models for training:
    update_ema(ema, model.module, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    # set diffusion timestep range
    if not (config.basic.rf or config.basic.rf_ori):
        timestep_start = config.basic.get("timestep_start", 0)
        timestep_end = config.basic.get("timestep_end", diffusion.num_timesteps)
        logger.info(f"Training Diffusion among Timestep begin at: {timestep_start}, end at: {timestep_end}")

    # Variables for monitoring/logging purposes:
    global_step = cur_steps
    train_steps = cur_steps
    log_steps = cur_steps
    running_loss = {}
    accum_iter = config.basic.accum_iter
    log_every = config.basic.log_every
    ckpt_every = config.basic.ckpt_every
    start_time = time()

    logger.info(f"Training for {config.basic.epochs} epochs...")
    for epoch in range(cur_epoch, config.basic.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        for x, y in loader:
            if hasattr(model.module, "training_iters"):
                model.module.training_iters += 1
            x = x.to(device)
            y = y.to(device)
            if not use_latent:
                with torch.no_grad():
                    x = vae.encode(x).latent_dist.sample().mul_(0.18215)

            with torch.autocast(device_type="cuda", dtype=torch.float32):
                if config.basic.rf:
                    loss_dict = diffusion.forward(x, y, train_step=train_steps)
                    loss = loss_dict["loss"].mean()
                else:
                    if 'same_t_per_batch' not in config.basic:
                        config.basic.same_t_per_batch = False 
                    if config.basic.same_t_per_batch:
                        t = torch.randint(0, diffusion.num_timesteps, (1,), device=device)
                        t = t.expand(x.shape[0])
                    else:
                        t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)

                    model_kwargs = dict(y=y)
                    loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
                    loss = loss_dict["loss"].mean()

            loss.backward()

            if (global_step + 1) % accum_iter == 0:
                if max_grad_norm:
                    clip_grad_norm_(model.parameters(), max_grad_norm)
                opt.step()
                opt.zero_grad()
                update_ema(ema, model.module)
                train_steps += 1
                if train_steps <= warmup_steps:
                    lr_sheduler_warmup.step()
            # Log loss values:
            for k, v in loss_dict.items():
                if k not in running_loss:
                    running_loss[k] = 0
                running_loss[k] += loss_dict[k].mean().item() / accum_iter

            log_steps += 1
            global_step += 1
            if global_step % (config.basic.log_every*accum_iter) == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)

                
                log_msg = f"(Global Step={global_step:08d}, Train Step={train_steps:08d}), "  #Train Loss: {avg_loss:.4f},
                if use_wandb and rank == 0:
                    wandb_info = {"global_step": global_step, "train_steps": train_steps}
                
                for k,v in running_loss.items():
                    avg_loss = torch.tensor(v / log_every, device=device)
                    dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                    avg_loss = avg_loss.item() / dist.get_world_size()
                    log_msg += f" {k} : {avg_loss:.4f} ,"
                    if use_wandb and rank == 0:
                        wandb_info[k] = avg_loss
                    
                    if rank == 0:
                        writer.add_scalar(k, avg_loss, train_steps)
                log_msg += f" LR: {opt.param_groups[0]['lr']}, Train Steps/Sec: {steps_per_sec:.2f}"
                logger.info(log_msg)
                if use_wandb and rank==0:
                    wandb_info["lr"] = opt.param_groups[0]['lr']
                    wandb_info["Train Steps/Sec"] = steps_per_sec

                    wandb_log(wandb_info, step=train_steps)

                if rank == 0:
                    writer.add_scalar('lr', opt.param_groups[0]["lr"], train_steps)
                for k,v in running_loss.items():
                    running_loss[k] = 0
                log_steps = 0
                start_time = time()
            # Save DiT checkpoint:
            if global_step % (ckpt_every*accum_iter) == 0:
                if rank == 0:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {os.path.abspath(checkpoint_path)}")
                    logger.info(f"single device batch size = {y.size(0)}")

                dist.barrier()
        if use_epoch_lr_scheduler:
            lr_sheduler_train_epoch.step()
            logger.info(f"Adjust lr to {opt.param_groups[0]['lr']} . ")


    model.eval()  # important! This disables randomized embedding dropout

    logger.info("Done!")
    cleanup()


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--resume_path", type=str)
    parser.add_argument("--exp_name", type=str, default="train_s_e16")
    args = parser.parse_args()
    config_yml_path = args.config
    if args.wandb:
        os.environ["WANDB_API_KEY"] = "your-wandb-api-key"
        os.environ["WANDB_ENTITY"] = "your-wandb-entity"
        os.environ["WANDB_PROJECT"] = "your-wandb-project"

        entity = os.environ["WANDB_ENTITY"]
        project = os.environ["WANDB_PROJECT"]
        experiment_name = args.exp_name
        wandb_initialize(entity, experiment_name, project)

    main(config_yml_path, resume_path=args.resume_path)
