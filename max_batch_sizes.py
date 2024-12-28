import subprocess
import os
import json

import torch
from torch import distributed as dist

from config import results_dir
from utils import setup, cleanup
from logger import setup_logging

logger = setup_logging("max_batch_sizes")


# Initial starting batch size
starting_batch_size = 64

if __name__ == "__main__":
    setup()

    # Retrieve distributed configuration from environment variables
    world_size = int(os.environ['WORLD_SIZE'])
    local_world_size = int(os.environ['LOCAL_WORLD_SIZE'])
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    master_port = int(os.environ['MASTER_PORT'])

    logger.info("Distributed configuration loaded.")
    logger.debug(f"world_size: {world_size}, local_world_size: {local_world_size}, "
                  f"rank: {rank}, local_rank: {local_rank}, master_port: {master_port}")
    logger.info(f"Using master port {master_port + 1} for the profiler.")

    # Configure GPU device and check memory stats
    device = torch.device("cuda", torch.cuda.current_device())
    init_gpu_free_mem, gpu_total_mem = torch.cuda.mem_get_info()
    init_gpu_free_mem /= (1024 ** 2)  # Convert to MB
    gpu_total_mem /= (1024 ** 2)  # Convert to MB
    logger.info(f"Initial GPU memory: (free: {init_gpu_free_mem:.2f} MB, total: {gpu_total_mem:.2f} MB)")

    # Initialize batch sizes
    current_batch_sizes = torch.ones((local_world_size,), dtype=torch.float32).to(device) * starting_batch_size
    previous_batch_sizes = current_batch_sizes.clone()
    logger.info(f"Starting batch sizes: {current_batch_sizes.cpu().tolist()}")

    os.makedirs(results_dir, exist_ok=True)

    low, high = None, None
    while True:
        logger.info("Launching profiler subprocess with current batch sizes...")
        subprocess.run(
            [
                "torchrun",
                "--nproc_per_node",
                f"{local_world_size}",
                "--master_port",
                f"{master_port + 1}",
                "profiler.py",
                "--batch_sizes",
            ] + [str(int(batch_size.item())) for batch_size in current_batch_sizes],
            check=True
        )

        # Read results from the profiler output
        result_path = os.path.join(results_dir, f"profiler.json")
        with open(result_path) as result_file:
            simulation_stats = json.load(result_file)

        # Check GPU memory usage and adjust batch sizes
        current_peak_reserved = torch.tensor([simulation_stats["peak_reserved"]], dtype=torch.float32).to(device)
        logger.info(f"Peak GPU memory reserved: {current_peak_reserved.item():.2f} MB")

        if current_peak_reserved == -1 or ((init_gpu_free_mem - current_peak_reserved * 1.1) <= 0):
            logger.info("Memory limit reached. Adjusting batch sizes downward.")
            high = current_batch_sizes.clone()
            if low is None:
                current_batch_sizes /= 2
            else:
                current_batch_sizes = (low + high) / 2
        else:
            logger.info("Memory usage within limits. Adjusting batch sizes upward.")
            low = current_batch_sizes.clone()
            if high is None:
                current_batch_sizes *= 2
            else:
                current_batch_sizes = (low + high) / 2

        logger.info(f"Batch sizes updated to {current_batch_sizes.cpu().tolist()} (was: {previous_batch_sizes.cpu().tolist()})")

        # Break if batch sizes are non-integer or unchanged
        if current_batch_sizes != current_batch_sizes.int():
            current_batch_sizes = low
            logger.info(f"Non-integer batch size detected. Finalizing batch sizes: {current_batch_sizes.cpu().tolist()}. Breaking...")
            break

        if torch.equal(previous_batch_sizes, current_batch_sizes):
            logger.info("Batch sizes unchanged. Finalizing...")
            break

        previous_batch_sizes = current_batch_sizes.clone()

    # Synchronize across all nodes
    logger.info("Waiting for all nodes to complete...")
    dist.barrier()

    # Gather maximum runnable batch sizes across all nodes
    gathered_batch_sizes = torch.zeros(world_size, dtype=torch.float32, device=current_batch_sizes.device)
    dist.all_gather_into_tensor(gathered_batch_sizes, current_batch_sizes)

    logger.info(f"Maximum runnable batch sizes gathered: {gathered_batch_sizes.cpu().tolist()}")

    # Save results
    result_path = os.path.join(results_dir, f"max_batch_sizes.pt")
    torch.save(gathered_batch_sizes.cpu(), result_path)
    logger.info(f"Saved maximum batch sizes to {result_path}.")

    cleanup()
    logger.info("Cleanup completed. Process exiting.")
