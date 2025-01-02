import subprocess
import os
import json

import torch
from torch import distributed as dist

from utils import setup, cleanup
from logger import setup_logging

# Set up logging
logger = setup_logging("batch_size_optimizer")


def update_batch_sizes(current_batch_sizes, gathered_times, max_batch_sizes, total_batch_size, learning_rate):
    """
    Adjust batch sizes dynamically based on execution times and constraints.

    Args:
        current_batch_sizes (torch.Tensor): Current batch sizes for each rank.
        gathered_times (torch.Tensor): Execution times for each rank.
        max_batch_sizes (torch.Tensor): Maximum batch sizes for each rank.
        total_batch_size (int): Current total batch size across all ranks.
        learning_rate (float): Learning rate for adjusting batch sizes.

    Returns:
        torch.Tensor: Updated batch sizes.
    """
    logger.debug("Updating batch sizes...")

    # Calculate relative speeds and balance adjustments
    relative_speeds = 1 / gathered_times
    relative_speeds /= relative_speeds.sum()
    change_speeds = relative_speeds - relative_speeds.mean()

    # Apply learning rate
    change_speeds *= learning_rate

    # Adjust to avoid exceeding max batch sizes
    max_change_speeds = (max_batch_sizes - current_batch_sizes) / total_batch_size
    min_change_speeds = (1 - current_batch_sizes) / total_batch_size

    differences_with_max = torch.clamp(change_speeds - max_change_speeds, min=0)
    if differences_with_max.count_nonzero():
        top_max_difference_index = differences_with_max.argmax()
        top_scale = max_change_speeds[top_max_difference_index] / change_speeds[top_max_difference_index]
    else:
        top_scale = 1.0

    differences_with_min = torch.clamp(change_speeds - min_change_speeds, max=0)
    if differences_with_min.count_nonzero():
        bottom_max_difference_index = differences_with_min.argmin()
        bottom_scale = min_change_speeds[bottom_max_difference_index] / change_speeds[bottom_max_difference_index]
    else:
        bottom_scale = 1.0

    scale_factor = min(top_scale, bottom_scale)
    if scale_factor < 1.0:
        if scale_factor == bottom_scale:
            logger.warning(f"With the current settings, we cannot find the most efficient batch sizes due to speed constraints "
                  f"of rank {top_max_difference_index}. Try a bigger batch size.")
        else:
            logger.warning(f"With the current settings, we cannot find the most efficient batch sizes due to memory constraints "
                  f"of rank {top_max_difference_index}. Try a smaller batch size.")

    change_speeds *= scale_factor

    # Update batch sizes
    current_batch_sizes += change_speeds * total_batch_size
    current_batch_sizes = current_batch_sizes.round()

    logger.debug(f"Updated batch sizes: {current_batch_sizes.tolist()}")
    return current_batch_sizes


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=["VIT_L_32", "VIT_L_16"], help="The model to use", required=True)
    parser.add_argument("--mode", type=str, choices=["DDP", "DDP+FP", "FSDP", "FSDP+OFFLOAD", "LSDP"], help="Training mode", required=True)
    parser.add_argument("--num_workers", default=4, type=int, help="Number of dataloader workers", required=True)
    parser.add_argument("--num_iterations", default=8, type=int, help="Number of iterations", required=True)
    parser.add_argument("--total_batch_size", type=int, help="Sum of all nodes batch sizes", required=True)
    parser.add_argument("--results_dir", type=str, help="Path to results dir", required=True)
    args = parser.parse_args()

    model = args.model
    mode = args.mode
    num_workers = args.num_workers
    num_iterations = args.num_iterations
    total_batch_size = args.total_batch_size
    results_dir = args.results_dir

    setup()

    # Load distributed configuration
    world_size = int(os.environ['WORLD_SIZE'])
    local_world_size = int(os.environ['LOCAL_WORLD_SIZE'])
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    master_port = int(os.environ['MASTER_PORT'])

    # Log the distributed setup
    logger.debug("Loaded distributed configuration.")
    logger.debug(f"world_size: {world_size}, local_world_size: {local_world_size}, "
                  f"rank: {rank}, local_rank: {local_rank}, master_port: {master_port}")
    logger.debug(f"Using master port {master_port + 1} for the profiler.")

    # Configure GPU device
    device = torch.device("cuda", torch.cuda.current_device())

    # Load maximum batch sizes
    max_batch_sizes_path = os.path.join(results_dir, f"max_batch_sizes.pt")
    if os.path.isfile(max_batch_sizes_path):
        max_batch_sizes = torch.load(max_batch_sizes_path, weights_only=True, map_location=device).float()
    else:
        logger.error("Max batch sizes file not found. Run find_max_batch_sizes.py first.")
        raise RuntimeError("Max batch sizes file not found.")
    
    logger.debug(f"Loaded max batch sizes: {max_batch_sizes.tolist()}")

    if total_batch_size > max_batch_sizes.sum():
        logger.error(f"The sum of max batch sizes ({max_batch_sizes.sum()}) is less than total batch size ({total_batch_size}).")
        raise RuntimeError(f"The sum of max batch sizes ({max_batch_sizes.sum()}) is less than total batch size ({total_batch_size}).")

    if total_batch_size < world_size:
        logger.error(f"Total batch size ({total_batch_size}) is less than world size ({world_size}).")
        raise RuntimeError(f"Total batch size ({total_batch_size}) is less than world size ({world_size}).")

    # Initialize current batch sizes
    start_batch_size_ratios = max_batch_sizes / max_batch_sizes.sum()
    current_batch_sizes = torch.ones((world_size,), dtype=torch.float32).to(device) + start_batch_size_ratios * (total_batch_size - world_size)
    current_batch_sizes = current_batch_sizes.round()
    previous_batch_sizes = current_batch_sizes.clone()

    logger.info(f"Starting batch sizes: {current_batch_sizes.tolist()}")

    os.makedirs(results_dir, exist_ok=True)

    learning_rate = 1.0
    max_iterations = 10
    for iteration in range(max_iterations):
        logger.debug(f"Iteration {iteration + 1}/{max_iterations} starting...")
        subprocess.run(
            [
                "torchrun",
                "--nproc_per_node",
                f"{local_world_size}",
                "--master_port",
                f"{master_port + 1}",
                "profiler.py",
                "--model",
                f"{model}",
                "--mode",
                f"{mode}",
                "--num_workers",
                f"{num_workers}",
                "--num_iterations",
                f"{num_iterations}",
                "--results_dir",
                f"{results_dir}",
                "--batch_sizes",
            ] + [str(int(batch_size.item())) for index, batch_size in enumerate(current_batch_sizes)
                 if index // local_world_size == rank // local_world_size],
            check=True
        )

        # Wait for all processes
        dist.barrier()

        # Gather simulation stats
        result_path = os.path.join(results_dir, f"profiler.json")
        with open(result_path) as result_file:
            simulation_stats = json.load(result_file)

        local_time_tensor = torch.tensor([simulation_stats["execution_time"]], dtype=torch.float32).to(device)
        gathered_times = torch.zeros(world_size, dtype=torch.float32, device=local_time_tensor.device)
        dist.all_gather_into_tensor(gathered_times, local_time_tensor)
        logger.debug(f"Execution times from nodes: {gathered_times.tolist()}")

        # Update batch sizes
        current_batch_sizes = update_batch_sizes(
            current_batch_sizes=current_batch_sizes,
            gathered_times=gathered_times,
            max_batch_sizes=max_batch_sizes,
            total_batch_size=total_batch_size,
            learning_rate=learning_rate
        )
        logger.debug(f"Batch sizes updated to {current_batch_sizes.tolist()} (was: {previous_batch_sizes.tolist()})")

        if torch.equal(previous_batch_sizes, current_batch_sizes):
            logger.debug("Batch sizes converged. Exiting loop.")
            break

        previous_batch_sizes = current_batch_sizes.clone()
        learning_rate = 1 / (iteration + 1)
        logger.debug(f"Learning rate updated to: {learning_rate}")

    logger.info(f"Final batch sizes with sum={total_batch_size} are {current_batch_sizes.tolist()}.")

    # Save results
    result_path = os.path.join(results_dir, f"best_batch_sizes_fixed_total.pt")
    torch.save(current_batch_sizes.cpu().int(), result_path)
    logger.info(f"Saved final batch sizes to {result_path}.")

    cleanup()
    logger.debug("Cleanup completed. Exiting process.")
