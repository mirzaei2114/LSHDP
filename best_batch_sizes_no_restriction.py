import subprocess
import os
import json

import torch
from torch import distributed as dist

from utils import setup, cleanup
from logger import setup_logging

logger = setup_logging("best_batch_sizes")


def update_batch_sizes(current_batch_sizes, gathered_times, max_batch_sizes, total_batch_size):
    """
    Adjust batch sizes dynamically based on execution times and constraints.

    Args:
        current_batch_sizes (torch.Tensor): Current batch sizes for each rank.
        gathered_times (torch.Tensor): Execution times for each rank.
        max_batch_sizes (torch.Tensor): Maximum batch sizes for each rank.
        total_batch_size (float): Current total batch size across all ranks.

    Returns:
        tuple: Updated current_batch_sizes and total_batch_size.
    """
    logger.debug("Updating batch sizes...")
    
    # Calculate relative speeds and balance adjustments
    relative_speeds = 1 / gathered_times
    relative_speeds /= relative_speeds.sum()
    change_speeds = relative_speeds - relative_speeds.mean()

    # Adjust to avoid exceeding max batch sizes
    max_change_speeds = (max_batch_sizes - current_batch_sizes) / total_batch_size
    differences_with_max = change_speeds - max_change_speeds
    top_scale = 1.0
    change_speeds -= differences_with_max.max()

    # Adjust total batch size based on change speeds
    total_batch_size += total_batch_size * change_speeds.sum()
    total_batch_size = total_batch_size.round()

    # Avoid falling below minimum allowed batch sizes
    min_change_speeds = (1 - current_batch_sizes) / total_batch_size
    differences_with_min = torch.clamp(change_speeds - min_change_speeds, max=0)
    if differences_with_min.count_nonzero():
        bottom_max_difference_index = differences_with_min.argmin()
        bottom_scale = min_change_speeds[bottom_max_difference_index] / change_speeds[bottom_max_difference_index]
    else:
        bottom_scale = 1.0

    # Scale adjustments
    scale_factor = min(top_scale, bottom_scale)
    change_speeds *= scale_factor

    # Update batch sizes
    current_batch_sizes += change_speeds * total_batch_size
    current_batch_sizes = current_batch_sizes.round()

    logger.debug(f"Updated batch sizes: {current_batch_sizes.tolist()}, total_batch_size: {total_batch_size}")
    return current_batch_sizes, total_batch_size


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=["VIT_L_32", "VIT_L_16"], help="The model to use", required=True)
    parser.add_argument("--mode", type=str, choices=["DDP", "DDP+FP", "FSDP", "FSDP+OFFLOAD", "LSHDP"], help="Training mode", required=True)
    parser.add_argument("--num_workers", default=4, type=int, help="Number of dataloader workers", required=True)
    parser.add_argument("--num_iterations", default=8, type=int, help="Number of iterations", required=True)
    parser.add_argument("--results_dir", type=str, help="Path to results dir", required=True)
    args = parser.parse_args()

    model = args.model
    mode = args.mode
    num_workers = args.num_workers
    num_iterations = args.num_iterations
    results_dir = args.results_dir

    setup()

    # Load distributed configuration
    world_size = int(os.environ['WORLD_SIZE'])
    local_world_size = int(os.environ['LOCAL_WORLD_SIZE'])
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    master_port = int(os.environ['MASTER_PORT'])

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

    total_batch_size = max_batch_sizes.sum()
    current_batch_sizes = max_batch_sizes.clone()
    previous_batch_sizes = current_batch_sizes.clone()
    logger.info(f"Starting batch sizes: {current_batch_sizes.tolist()}")

    os.makedirs(results_dir, exist_ok=True)

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
        current_batch_sizes, total_batch_size = update_batch_sizes(
            current_batch_sizes=current_batch_sizes,
            gathered_times=gathered_times,
            max_batch_sizes=max_batch_sizes,
            total_batch_size=total_batch_size
        )
        logger.debug(f"Batch sizes updated to {current_batch_sizes.tolist()} (was: {previous_batch_sizes.tolist()})")

        if torch.equal(previous_batch_sizes, current_batch_sizes):
            logger.debug("Batch sizes converged. Exiting loop.")
            break

        previous_batch_sizes = current_batch_sizes.clone()

    logger.info(f"Final batch sizes with sum={total_batch_size.item()} are {current_batch_sizes.tolist()}.")

    # Save results
    result_path = os.path.join(results_dir, f"best_batch_sizes_no_restriction.pt")
    torch.save(current_batch_sizes.cpu().int(), result_path)
    logger.info(f"Saved final batch sizes to {result_path}.")

    cleanup()
    logger.debug("Cleanup completed. Exiting process.")
