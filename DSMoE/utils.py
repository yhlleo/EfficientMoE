import copy
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

import logging
import os

from torch.utils.tensorboard import SummaryWriter
import importlib
import datetime
from omegaconf import OmegaConf

import wandb
import hashlib

# FSDP
from torch.distributed.fsdp import (
    CPUOffload,
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
    FullStateDictConfig,
    StateDictType,
)
import torch.distributed.fsdp._traversal_utils as traversal_utils
from models.models_DSMoE import DiTBlock
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
import functools

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)

    ema_buffers = OrderedDict(ema_model.named_buffers())
    model_buffers = OrderedDict(model.named_buffers())

    for name, buffer in model_buffers.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_buffers[name].mul_(decay).add_(buffer.data, alpha=1 - decay)


#--------------------------#
#           FSDP
#--------------------------#

class FSDPConfig:
    def __init__(
        self,
        sharding_strategy, 
        backward_prefetch, 
        cpu_offload, 
        num_replicate,
        num_shard=8,
    ):
        self.sharding_strategy = sharding_strategy
        self.backward_prefetch = backward_prefetch
        self.cpu_offload = cpu_offload
        self.num_replicate = num_replicate
        self.num_shard = num_shard

def fsdp_wrapper(original_model, fsdp_config, ignored_modules=[], dtype="fp32"):
    float_type = torch.bfloat16 if dtype == "bf16" else torch.float32
    if fsdp_config.sharding_strategy == 'HYBRID_SHARD':
        device_mesh = init_device_mesh(
            "cuda", 
            mesh_shape=(fsdp_config.num_replicate, fsdp_config.num_shard),
            mesh_dim_names=("replicate", "shard")
        )
    else:
        device_mesh = None
    return FSDP(
        original_model,
        auto_wrap_policy=functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={
                DiTBlock,
            },
        ),
        ignored_modules=ignored_modules,
        mixed_precision=MixedPrecision(
            param_dtype=float_type,
            reduce_dtype=float_type,
            buffer_dtype=float_type,
        ),
        device_id=dist.get_rank() % torch.cuda.device_count(),
        sharding_strategy=ShardingStrategy[fsdp_config.sharding_strategy],
        backward_prefetch=BackwardPrefetch[fsdp_config.backward_prefetch],
        cpu_offload=CPUOffload(offload_params=fsdp_config.cpu_offload),
        device_mesh=device_mesh,
        use_orig_params=True,
    )

def fsdp_ema_setup(ema_model, fsdp_config, ignored_modules=[]):
    for param in ema_model.parameters():
        param.requires_grad = False

    ema_model = fsdp_wrapper(ema_model, fsdp_config, ignored_modules=ignored_modules)
    return ema_model

@torch.no_grad()
def fsdp_ema_update(ema_model, model, decay=0.9999):
    ema_handles = traversal_utils._get_fsdp_handles(ema_model)
    new_handles = traversal_utils._get_fsdp_handles(model)
    assert len(ema_handles) == len(new_handles)
    ema_params = []
    new_params = []

    for ema_handle, new_handle in zip(ema_handles, new_handles):
        if ema_handle.flat_param is not None and new_handle.flat_param.requires_grad:
            ema_params.append(ema_handle.flat_param.data)
            new_params.append(new_handle.flat_param.data.to(dtype=ema_handle.flat_param.dtype))

    torch._foreach_mul_(ema_params, decay)
    torch._foreach_add_(ema_params, new_params, alpha=1 - decay)

def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger



def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


def setup_ddp(args):
    # Setup DDP:
    #dist.init_process_group("nccl")
    if torch.cuda.is_available() and torch.distributed.is_nccl_available():
        dist.init_process_group("nccl")
    else:
        dist.init_process_group("gloo")
    
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    #rank = dist.get_rank()
    #device = rank % torch.cuda.device_count()
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    #return rank, device, seed
    return rank, world_size, local_rank, device

def setup_ddp0(args):
    # Setup DDP:
    #dist.init_process_group("nccl")
    if torch.cuda.is_available() and torch.distributed.is_nccl_available():
        dist.init_process_group("nccl")
    else:
        dist.init_process_group("gloo")
    
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."

    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    #torch.cuda.set_device(local_rank)
    #device = torch.device(f"cuda:{local_rank}")
    seed = args.global_seed * dist.get_world_size() + rank 
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")
    return rank, device, seed

def setup_exp_dir(rank, args):
    config = copy.deepcopy(args)
    args = args.basic
    # Setup an experiment folder:
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)

        experiment_index = len(glob(f"{args.results_dir}/*"))
        temp = glob(f"{args.results_dir}/*")
        temp = [int(os.path.basename(_).split("-")[0]) for _ in temp]
        experiment_index = max(temp) + 1 if len(temp) > 0 else 0

        # model_string_name = args.model.replace("/", "-")  # e.g., DiT-XL/2 --> DiT-XL-2 (for naming folders)
        model_string_name = args.exp_name

        experiment_dir = f"{args.results_dir}/{experiment_index:04d}-{model_string_name}"  # Create an experiment folder
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints

        tensorboard_dir = f"{experiment_dir}/tensorboard"  # tensorboard log dir 
        os.makedirs(tensorboard_dir, exist_ok=True)

        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")

        writer = SummaryWriter(log_dir=tensorboard_dir)
        logger.info("Set up tensorboard SummaryWriter")

        now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        OmegaConf.save(config, os.path.join(experiment_dir, f"config_{now}.yaml"))

        return logger, writer, checkpoint_dir
    else:
        logger = create_logger(None)
        return logger, None, None





def setup_data(rank, args):
    dataset_config = args.get("dataset_config", None)

    if dataset_config:
        dataset = instantiate_from_config(dataset_config)

    else:
        transform = transforms.Compose([
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])
        dataset = ImageFolder(args.data_path, transform=transform)

    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // dist.get_world_size()),
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    return dataset, sampler, loader



def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def instantiate_from_config(config):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))



def get_lr_scheduler_config(config, optimizer):
    new_dict = {}
    new_dict["target"] = config["target"]
    new_dict["params"] = {"optimizer": optimizer}
    for k,v in config.params.items():
        new_dict["params"][k] = v
    return new_dict

#------ wandb log ------#
def is_main_process():
    return dist.get_rank() == 0

def wandb_log(stats, step=None):
    if is_main_process():
        wandb.log({k: v for k, v in stats.items()}, step=step)

def generate_run_id(exp_name):
    # https://stackoverflow.com/questions/16008670/how-to-hash-a-string-into-8-digits
    return str(int(hashlib.sha256(exp_name.encode('utf-8')).hexdigest(), 16) % 10 ** 8)

def wandb_initialize(entity, exp_name, project_name):
    #config_dict = namespace_to_dict(args)
    wandb.login(key=os.environ["WANDB_API_KEY"])
    wandb.init(
        entity=entity,
        project=project_name,
        name=exp_name,
        id=generate_run_id(exp_name),
        resume=False,
    )


