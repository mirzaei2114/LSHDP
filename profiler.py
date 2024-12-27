import os
import gc
import time
import json
import torch.distributed
from tqdm.auto import tqdm

import torch
from torch import distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from utils import setup, cleanup, get_model_size

from model_specific import get_model_and_optim, get_dataset_and_dataloader
from config import (
    results_dir,
    fsdp_kwargs,
    num_workers
)
from logger import logging, setup_logging

logger = setup_logging("profiler")


def simulate_training(num_iterations):
    """
    Simulates the training loop with FSDP, tracking memory and execution time.

    Args:
        num_iterations (int): Total number of iterations to simulate.

    Returns:
        dict: Metrics including model size, execution time, and memory usage.
    """
    logger.debug("Starting simulation of training...")
    device = torch.device("cuda", torch.cuda.current_device())

    # Get model and optimizer
    logger.debug("Loading model and optimizer...")
    model, optimizer_cls = get_model_and_optim()
    model_size = get_model_size(model)
    logger.debug(f"Model loaded. Size: {model_size:.2f} MB")

    # Get dataset and dataloader
    logger.debug("Setting up dataset and dataloader...")
    _, dataloader = get_dataset_and_dataloader(
        world_size=int(os.environ['WORLD_SIZE']),
        rank=int(os.environ['RANK']),
        batch_sizes=args.batch_sizes,
        num_workers=num_workers
    )

    # Wrap model with FSDP and initialize optimizer
    logger.debug("Wrapping model with FSDP...")
    fsdp_model = FSDP(model, **fsdp_kwargs)
    optimizer = optimizer_cls(fsdp_model.parameters())

    iteration = 0
    epoch = 0
    completed = False
    start_time = None

    if logger.level <= logging.DEBUG:
        progress_bar = tqdm(total=num_iterations, desc="Simulate training", unit="batch")
    while not completed:
        logger.debug(f"Starting epoch {epoch}...")
        dataloader.sampler.set_epoch(epoch)  # Shuffle data for each epoch
        for batch in dataloader:
            inputs, targets = batch
            targets = targets.cuda(non_blocking=True)
            
            optimizer.zero_grad()
            outputs = fsdp_model(inputs)
            loss = nn.functional.cross_entropy(outputs, targets)
            loss.backward()
            optimizer.step()

            iteration += 1
            if logger.level <= logging.DEBUG:
                progress_bar.update(1)

            # Skip the initial warm-up batches for timing
            if iteration == num_workers:
                start_time = time.perf_counter()  # Start timing after warm-up

            if iteration >= num_iterations:
                logger.debug("Target number of iterations reached.")
                completed = True
                break

        epoch += 1
    if logger.level <= logging.DEBUG:
        progress_bar.close()

    end_time = time.perf_counter()  # End timing
    execution_time = (end_time - start_time) * 1000 if start_time else -1
    peak_memory = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    peak_reserved = torch.cuda.max_memory_reserved(device) / (1024 ** 2)

    logger.debug(f"Training simulation completed in {execution_time:.4f} ms.")
    logger.debug(f"Peak GPU memory usage: Allocated {peak_memory:.2f} MB, Reserved {peak_reserved:.2f} MB")

    return {
        "model_size": model_size,
        "execution_time": execution_time,
        "peak_memory": peak_memory,
        "peak_reserved": peak_reserved
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_sizes", nargs="+", type=int, help="Batch sizes per rank")
    parser.add_argument("--num_iterations", default=num_workers * 2, type=int, help="Number of iterations")
    args = parser.parse_args()

    # Validate input arguments
    assert args.num_iterations > num_workers, (
        "Number of iterations should be greater than number of workers "
        "to avoid including dataloader warm-up overhead."
    )

    os.makedirs(results_dir, exist_ok=True)
    setup()

    try:
        simulation_stats = simulate_training(args.num_iterations)

    except torch.cuda.OutOfMemoryError as gpu_oom:
        logger.debug("GPU Out of Memory Error occurred.")
        simulation_stats = {
            "model_size": -1,
            "execution_time": -1,
            "peak_memory": -1,
            "peak_reserved": -1
        }
    except MemoryError as cpu_oom:
        logger.debug("CPU Out of Memory Error occurred.")
        simulation_stats = {
            "model_size": -1,
            "execution_time": -1,
            "peak_memory": -1,
            "peak_reserved": -1
        }
    except RuntimeError as runtime_err:
        if "CUBLAS_STATUS_ALLOC_FAILED" in str(runtime_err) or "Failed to CUDA calloc" in str(runtime_err):
            logger.debug("RuntimeError related to GPU memory allocation.")
            simulation_stats = {
                "model_size": -1,
                "execution_time": -1,
                "peak_memory": -1,
                "peak_reserved": -1
            }
        else:
            logger.error("Unexpected RuntimeError occurred.", exc_info=True)
            raise runtime_err
    finally:
        torch.cuda.empty_cache()  # Clear GPU cache
        gc.collect()  # Trigger garbage collection
        logger.debug("Cleaned up memory and caches.")

    # Save the simulation stats
    result_path = os.path.join(results_dir, "profiler.json")
    with open(result_path, 'w') as result_file:
        json.dump(simulation_stats, result_file)
    logger.debug(f"Simulation results saved to {result_path}.")

    # Ensure all processes finish before cleanup
    dist.barrier()
    cleanup()
    logger.debug("Training simulation process completed and resources cleaned up.")
