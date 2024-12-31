import os
import functools

import torch
from torch import nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch.distributed as dist
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.fsdp.api import (
    BackwardPrefetch,
    MixedPrecision,
    CPUOffload,
    ShardingStrategy,
)
from torchvision.models.vision_transformer import EncoderBlock

from heterogeneous_distributed_sampler import HeterogeneousDistributedSampler
from logger import setup_logging

logger = setup_logging("utils")


def setup():
    """
    Set up the distributed process group and CUDA device for multi-GPU training.
    Initializes the NCCL backend for efficient communication.
    """
    logger.debug("Initializing distributed process group...")
    dist.init_process_group(
        backend="nccl",
        world_size=int(os.environ['WORLD_SIZE']),
        rank=int(os.environ['RANK'])
    )
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    logger.debug("Distributed process group initialized successfully.")


def cleanup():
    """
    Clean up the distributed process group after training.
    Ensures that resources are properly released.
    """
    logger.debug("Cleaning up distributed process group...")
    dist.destroy_process_group()
    logger.debug("Distributed process group cleaned up successfully.")


def get_model_specific_vars(model, world_size, rank, batch_sizes, num_workers):
    """
    Prepares model-specific variables, dataset, and configurations for distributed training.

    Args:
        model (str): The model name, e.g., "VIT_L_32" or "VIT_L_16".
        world_size (int): Total number of distributed processes (e.g., GPUs or nodes).
        rank (int): Rank of the current process in the distributed setup.
        batch_sizes (list of int): List of batch sizes for each process, where index corresponds to rank.
        num_workers (int): Number of worker threads for data loading.

    Returns:
        tuple: Contains:
            - dataset (Dataset): The dataset object with the appropriate transform applied.
            - dataloader (DataLoader): The dataloader for the dataset, distributed by rank.
            - model (nn.Module): The initialized model for the given name.
            - optimizer_cls (callable): A partially initialized optimizer class.
            - loss (nn.Module): The loss function.
    """
    # Check and load the specified model along with its pretrained weights
    if model == "VIT_L_32":
        # Import and initialize Vision Transformer Large-32 model
        from torchvision.models.vision_transformer import vit_l_32, ViT_L_32_Weights
        model = vit_l_32(weights=None)

        # Define transformation pipeline for image preprocessing
        transform = transforms.Compose([
            transforms.Resize(256),  # Resize to 256 pixels on the shortest edge
            transforms.CenterCrop(224),  # Crop a 224x224 region at the center
            transforms.ToTensor(),  # Convert image to tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize image
        ])
    elif model == "VIT_L_16":
        # Import and initialize Vision Transformer Large-16 model
        from torchvision.models.vision_transformer import vit_l_16, ViT_L_16_Weights
        model = vit_l_16(weights=None)

        # Define transformation pipeline for image preprocessing
        transform = transforms.Compose([
            transforms.Resize(242),  # Resize to 242 pixels on the shortest edge
            transforms.CenterCrop(224),  # Crop a 224x224 region at the center
            transforms.ToTensor(),  # Convert image to tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize image
        ])
    else:
        # Raise an error if the specified model name is invalid
        raise RuntimeError("Invalid model")

    # Prepare the CIFAR10 dataset with the specified transformation
    dataset = datasets.CIFAR10(
        root='../data',  # Path to the dataset
        train=True,  # Use the training split
        transform=transform,  # Apply the transformation pipeline
        download=True,  # Download the dataset if not already available
    )

    # Set up the distributed sampler with heterogeneous batch sizes
    sampler = HeterogeneousDistributedSampler(
        dataset, 
        num_replicas=world_size,  # Total number of distributed processes
        participate_ratios=batch_sizes,  # Batch sizes for each rank
        rank=rank,  # Current process's rank
        drop_last=False  # Do not drop the last incomplete batch
    )

    # Create a DataLoader to load data for the current rank
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_sizes[rank],  # Batch size for the current rank
        sampler=sampler,  # Use the distributed sampler
        pin_memory=True,  # Enable pinned memory for faster data transfer
        num_workers=num_workers  # Number of worker threads
    )

    # Define the optimizer class with partial initialization
    optimizer_cls = functools.partial(AdamW, lr=1e-4)  # AdamW optimizer with a learning rate of 1e-4

    # Define the loss function
    loss = nn.CrossEntropyLoss()  # Cross-entropy loss for classification

    # Return all the components for distributed training
    return dataset, dataloader, model, optimizer_cls, loss


def get_model_size(model):
    """
    Calculate the total size of a PyTorch model in megabytes (MB).
    
    Args:
        model (torch.nn.Module): The model to calculate size for.

    Returns:
        float: Total size of the model in MB.
    """
    logger.debug("Calculating model size...")
    total_size = 0

    # Sum up the size of all parameters
    for param in model.parameters():
        total_size += param.nelement() * param.element_size()

    # Sum up the size of all buffers
    for buffer in model.buffers():
        total_size += buffer.nelement() * buffer.element_size()

    total_size_in_mb = total_size / (1024 ** 2)
    logger.debug(f"Model size calculated: {total_size_in_mb:.2f} MB")
    return total_size_in_mb


def get_fsdp_kwargs(mode):
    """
    Returns configuration arguments for Fully Sharded Data Parallel (FSDP) based on the given mode.
    
    Args:
        mode (str): The mode for distributed training. Can be one of "DDP", "DDP+FP", "FSDP", "FSDP+OFFLOAD", or "LSHDP".
    
    Returns:
        dict or None: A dictionary of FSDP configuration arguments.
    
    Raises:
        RuntimeError: If an invalid mode is provided.
    """
    if mode == "DDP":
        # Configuration for Distributed Data Parallel mode
        fsdp_kwargs = {
            "sharding_strategy": ShardingStrategy.NO_SHARD,  # Disable parameter sharding
            "backward_prefetch": None,  # Disable backward prefetch
            "mixed_precision": MixedPrecision(
                param_dtype=torch.float16,  # Use FP16 for model parameters
                reduce_dtype=torch.float16,  # Use FP16 for gradient reduction
                buffer_dtype=torch.float16,  # Use FP16 for buffers
            ),
            "forward_prefetch": False,  # Disbale prefetching during forward pass
        }
        return fsdp_kwargs
    elif mode == "DDP+FP":
        # Configuration for Distributed Data Parallel mode + Full Precision
        fsdp_kwargs = {
            "sharding_strategy": ShardingStrategy.NO_SHARD,  # Disable parameter sharding
            "backward_prefetch": None,  # Disable backward prefetch
            "mixed_precision": None,  # Use full precision
            "forward_prefetch": False,  # Disbale prefetching during forward pass
        }
        return fsdp_kwargs
    elif mode == "FSDP":
        # Configuration for standard Fully Sharded Data Parallel mode
        fsdp_kwargs = {
            "sharding_strategy": ShardingStrategy.FULL_SHARD,  # Use full parameter sharding
            "auto_wrap_policy": functools.partial(
                transformer_auto_wrap_policy, 
                transformer_layer_cls=(EncoderBlock,)
            ),  # Automatically wrap transformer layers
            "backward_prefetch": BackwardPrefetch.BACKWARD_PRE,  # Optimize backward pass execution
            "mixed_precision": MixedPrecision(
                param_dtype=torch.float16,  # Use FP16 for model parameters
                reduce_dtype=torch.float16,  # Use FP16 for gradient reduction
                buffer_dtype=torch.float16,  # Use FP16 for buffers
            ),
            "forward_prefetch": True,  # Enable prefetching during forward pass
        }
        return fsdp_kwargs
    elif mode == "FSDP+OFFLOAD":
        # Configuration for FSDP with CPU parameter offloading
        fsdp_kwargs = {
            "sharding_strategy": ShardingStrategy.FULL_SHARD,  # Use full parameter sharding
            "cpu_offload": CPUOffload(offload_params=True),  # Offload parameters to CPU
            "auto_wrap_policy": functools.partial(
                transformer_auto_wrap_policy, 
                transformer_layer_cls=(EncoderBlock,)
            ),  # Automatically wrap transformer layers
            "backward_prefetch": BackwardPrefetch.BACKWARD_PRE,  # Optimize backward pass execution
            "mixed_precision": MixedPrecision(
                param_dtype=torch.float16,  # Use FP16 for model parameters
                reduce_dtype=torch.float16,  # Use FP16 for gradient reduction
                buffer_dtype=torch.float16,  # Use FP16 for buffers
            ),
            "forward_prefetch": True,  # Enable prefetching during forward pass
        }
        return fsdp_kwargs
    elif mode == "LSHDP":
        # Configuration for Locally-Sharded Heterogeneous Data Parallel mode
        fsdp_kwargs = {
            "sharding_strategy": ShardingStrategy.NO_SHARD,  # Disable parameter sharding
            "cpu_offload": CPUOffload(offload_params=True),  # Offload parameters to CPU
            "auto_wrap_policy": functools.partial(
                transformer_auto_wrap_policy, 
                transformer_layer_cls=(EncoderBlock,)
            ),  # Automatically wrap transformer layers
            "backward_prefetch": BackwardPrefetch.BACKWARD_PRE,  # Optimize backward pass execution
            "mixed_precision": MixedPrecision(
                param_dtype=torch.float16,  # Use FP16 for model parameters
                reduce_dtype=torch.float16,  # Use FP16 for gradient reduction
                buffer_dtype=torch.float16,  # Use FP16 for buffers
            ),
            "forward_prefetch": True,  # Enable prefetching during forward pass
        }
        return fsdp_kwargs
    else:
        # Raise an error if the mode is invalid
        raise RuntimeError("Invalid mode")
