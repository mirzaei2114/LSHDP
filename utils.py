import os

import torch
import torch.distributed as dist

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
