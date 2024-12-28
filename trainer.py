import os
import gc
import time
import json
import torch.distributed
from tqdm.auto import tqdm

import torch
from torch import distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.nn.parallel import DistributedDataParallel as DDP

from utils import setup, cleanup, get_model_size

from model_specific import get_model_optim_loss, get_dataset_and_dataloader
from config import (
    results_dir,
    fsdp_kwargs,
    num_workers
)
from logger import setup_logging

logger = setup_logging("trainer")


def train(batch_sizes, num_iterations):
    """
    The training loop with FSDP, tracking memory and execution time.

    Args:
        num_iterations (int): Total number of iterations.

    Returns:
        dict: Metrics including model size, execution time, and memory usage.
    """
    logger.debug("Starting training...")
    device = torch.device("cuda", torch.cuda.current_device())

    # Get model and optimizer
    logger.debug("Loading model and optimizer...")
    model, optimizer_cls, loss_fn = get_model_optim_loss()
    model_size = get_model_size(model)
    logger.debug(f"Model loaded. Size: {model_size:.2f} MB")

    # Get dataset and dataloader
    logger.debug("Setting up dataset and dataloader...")
    _, dataloader = get_dataset_and_dataloader(
        world_size=int(os.environ['WORLD_SIZE']),
        rank=int(os.environ['RANK']),
        batch_sizes=batch_sizes,
        num_workers=num_workers
    )

    # Calculate loss coefficient
    logger.debug("Calculating loss coefficient...")
    participate_ratios = torch.Tensor(batch_sizes).to(device)
    participate_ratios /= participate_ratios.sum()
    loss_coef = int(os.environ['WORLD_SIZE']) * participate_ratios[int(os.environ['RANK'])]
    logger.debug(f"Loss coefficient: {loss_coef}")

    # Wrap model and initialize optimizer
    if fsdp_kwargs is not None:
        logger.debug("Wrapping model with FSDP...")
        if "cpu_offload" not in fsdp_kwargs or not fsdp_kwargs["cpu_offload"].offload_params:
            model = model.to(device=device)
        wrapped_model = FSDP(model, **fsdp_kwargs)
    else:
        logger.debug("Wrapping model with DDP...")
        model = model.to(device=device)
        wrapped_model = DDP(model, device_ids=[device.index])
    optimizer = optimizer_cls(wrapped_model.parameters())

    iteration = 0
    epoch = 0
    completed = False
    start_time = None
    losses = []

    progress_bar = tqdm(total=num_iterations, desc="Training", unit="batch")
    while not completed:
        logger.debug(f"Starting epoch {epoch}...")
        dataloader.sampler.set_epoch(epoch)  # Shuffle data for each epoch
        for batch in dataloader:
            inputs, targets = batch
            targets = targets.cuda(non_blocking=True)
            
            optimizer.zero_grad()
            outputs = wrapped_model(inputs)
            loss = loss_fn(outputs, targets)
            losses.append(loss.item())
            (loss*loss_coef).backward()
            optimizer.step()

            iteration += 1
            progress_bar.update(1)

            # Skip the initial warm-up batches for timing
            if iteration == num_workers:
                start_time = time.perf_counter()  # Start timing after warm-up

            if iteration >= num_iterations:
                logger.debug("Target number of iterations reached.")
                completed = True
                break

        epoch += 1
    progress_bar.close()

    end_time = time.perf_counter()  # End timing
    execution_time = (end_time - start_time) * 1000 if start_time else -1
    peak_memory = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    peak_reserved = torch.cuda.max_memory_reserved(device) / (1024 ** 2)

    logger.debug(f"Training completed in {execution_time:.4f} ms.")
    logger.debug(f"Peak GPU memory usage: Allocated {peak_memory:.2f} MB, Reserved {peak_reserved:.2f} MB")

    return {
        "model_size": model_size,
        "execution_time": execution_time,
        "peak_memory": peak_memory,
        "peak_reserved": peak_reserved,
        "losses": losses
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_sizes", nargs="+", type=int, help="Batch sizes per rank")
    parser.add_argument("--batch_sizes_path", default="temp_results/best_batch_sizes.pt", type=str, help="Path to batch sizes per rank")
    parser.add_argument("--num_iterations", default=100, type=int, help="Number of iterations")
    args = parser.parse_args()

    # Validate input arguments
    assert args.num_iterations > num_workers, (
        "Number of iterations should be greater than number of workers "
        "to avoid including dataloader warm-up overhead."
    )
    if args.batch_sizes is not None and len(args.batch_sizes) > 0:
        batch_sizes = args.batch_sizes
    else:
        assert os.path.isfile(args.batch_sizes_path), (
            "Cannot find batch sizes file."
        )
        batch_sizes = torch.load(args.batch_sizes_path, weights_only=True, map_location="cpu").tolist()

    os.makedirs(results_dir, exist_ok=True)
    setup()
    device = torch.device("cuda", torch.cuda.current_device())


    try:
        training_stats = train(batch_sizes, args.num_iterations)

    except torch.cuda.OutOfMemoryError as gpu_oom:
        logger.debug("GPU Out of Memory Error occurred.")
        training_stats = {
            "model_size": -1,
            "execution_time": -1,
            "peak_memory": -1,
            "peak_reserved": -1,
            "losses": []
        }
    except MemoryError as cpu_oom:
        logger.debug("CPU Out of Memory Error occurred.")
        training_stats = {
            "model_size": -1,
            "execution_time": -1,
            "peak_memory": -1,
            "peak_reserved": -1,
            "losses": []
        }
    except RuntimeError as runtime_err:
        if "CUBLAS_STATUS_ALLOC_FAILED" in str(runtime_err) or "Failed to CUDA calloc" in str(runtime_err):
            logger.debug("RuntimeError related to GPU memory allocation.")
            training_stats = {
                "model_size": -1,
                "execution_time": -1,
                "peak_memory": -1,
                "peak_reserved": -1,
                "losses": []
            }
        else:
            logger.error("Unexpected RuntimeError occurred.", exc_info=True)
            raise runtime_err
    finally:
        torch.cuda.empty_cache()  # Clear GPU cache
        gc.collect()  # Trigger garbage collection
        logger.debug("Cleaned up memory and caches.")

    # Save the training stats
    result_path = os.path.join(results_dir, "trainer.json")
    with open(result_path, 'w') as result_file:
        json.dump(training_stats, result_file)
    logger.debug(f"Training results saved to {result_path}.")

    # Ensure all processes finish before cleanup
    dist.barrier()
    cleanup()
    logger.debug("Training process completed and resources cleaned up.")
