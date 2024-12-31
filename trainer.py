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

from utils import setup, cleanup, get_model_specific_vars, get_model_size, get_fsdp_kwargs
from logger import logging, setup_logging

logger = setup_logging("trainer")


def train(model, batch_sizes, num_workers, use_loss_coef, fsdp_kwargs):
    """
    The training loop with FSDP, tracking memory and execution time.

    Args:
        model (str): The name of model to use.
        batch_sizes (list of int): The sizes of batches to simulate in each iteration.
        num_workers (int): The number of workers (e.g., GPUs or nodes) used during training.
        fsdp_kwargs (dict): The configuration arguments for FSDP, defining how the model and 
                            training process should be distributed and optimized.

    Returns:
        dict: Metrics including model size, execution time, and memory usage.
    """
    logger.info("Starting training...")
    device = torch.device("cuda", torch.cuda.current_device())

    # Get model and optimizer
    logger.debug("Loading model specific variables...")
    dataset, dataloader, model, optimizer_cls, loss_fn = get_model_specific_vars(
        model,
        world_size=int(os.environ['WORLD_SIZE']),
        rank=int(os.environ['RANK']),
        batch_sizes=batch_sizes,
        num_workers=num_workers
    )
    model_size = get_model_size(model)
    logger.debug(f"Got model specific variables. Model size: {model_size:.2f} MB, Max possible iterations: {len(dataloader)}")

    # Calculate loss coefficient
    logger.debug("Calculating loss coefficient...")
    if use_loss_coef:
        participate_ratios = torch.Tensor(batch_sizes).to(device)
        participate_ratios /= participate_ratios.sum()
        loss_coef = int(os.environ['WORLD_SIZE']) * participate_ratios[int(os.environ['RANK'])]
    else:
        loss_coef = 1.0
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

    if logger.level <= logging.DEBUG:
        progress_bar = tqdm(total=len(dataloader), desc="Training", unit="batch")
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
            if logger.level <= logging.DEBUG:
                progress_bar.update(1)

            # Skip the initial warm-up batches for timing
            if iteration == num_workers:
                start_time = time.perf_counter()  # Start timing after warm-up

            # if iteration >= num_iterations:
            #     logger.debug("Target number of iterations reached.")
            #     completed = True
            #     break

        epoch += 1
        break
    if logger.level <= logging.DEBUG:
        progress_bar.close()

    end_time = time.perf_counter()  # End timing
    execution_time = (end_time - start_time) * 1000 if start_time else -1
    peak_memory = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    peak_reserved = torch.cuda.max_memory_reserved(device) / (1024 ** 2)

    logger.info(f"Training completed in {execution_time:.4f} ms.")
    logger.info(f"Peak GPU memory usage: Allocated {peak_memory:.2f} MB, Reserved {peak_reserved:.2f} MB")

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
    parser.add_argument("--model", type=str, choices=["VIT_L_32", "VIT_L_16"], help="The model to use", required=True)
    parser.add_argument("--mode", type=str, choices=["DDP", "DDP+FP", "FSDP", "FSDP+OFFLOAD", "LSHDP"], help="Training mode", required=True)
    parser.add_argument("--total_batch_size", type=int, help="Sum of all nodes batch sizes", required=False)
    parser.add_argument("--batch_sizes_path", choices=["fixed_total", "no_restriction"], type=str, help="Path to batch sizes per rank", required=False)
    parser.add_argument("--num_workers", default=4, type=int, help="Number of dataloader workers", required=True)
    parser.add_argument("--use_loss_coef", action='store_true', default=False, help="Whether using loss coef or not", required=False)
    parser.add_argument("--results_dir", type=str, help="Path to results dir", required=True)
    args = parser.parse_args()

    model = args.model
    mode = args.mode
    total_batch_size = args.total_batch_size
    batch_sizes_path = args.batch_sizes_path
    num_workers = args.num_workers
    use_loss_coef = args.use_loss_coef
    results_dir = args.results_dir

    if total_batch_size is not None and total_batch_size > 0:
        if total_batch_size < int(os.environ['WORLD_SIZE']):
            logger.error(f"Total batch size ({total_batch_size}) is less than world size ({int(os.environ['WORLD_SIZE'])}).")
            raise RuntimeError(f"Total batch size ({total_batch_size}) is less than world size ({int(os.environ['WORLD_SIZE'])}).")
        if total_batch_size % int(os.environ['WORLD_SIZE']) != 0:
            logger.error(f"Total batch size ({total_batch_size}) is not divisable by world size ({int(os.environ['WORLD_SIZE'])}).")
            raise RuntimeError(f"Total batch size ({total_batch_size}) is not divisable by world size ({int(os.environ['WORLD_SIZE'])}).")
        batch_sizes = [total_batch_size//int(os.environ['WORLD_SIZE']) for _ in range(int(os.environ['WORLD_SIZE']))]
        result_path = os.path.join(results_dir, f"trainer_equal.json")
    else:
        result_path = os.path.join(results_dir, f"trainer_{batch_sizes_path}.json")
        batch_sizes = torch.load(os.path.join(results_dir, f"best_batch_sizes_{batch_sizes_path}.pt"), weights_only=True, map_location="cpu").tolist()

    os.makedirs(results_dir, exist_ok=True)
    setup()
    fsdp_kwargs = get_fsdp_kwargs(mode)

    try:
        training_stats = train(model, batch_sizes, num_workers, use_loss_coef, fsdp_kwargs)

    except torch.cuda.OutOfMemoryError as gpu_oom:
        logger.warning("GPU Out of Memory Error occurred.")
        training_stats = {
            "model_size": -1,
            "execution_time": -1,
            "peak_memory": -1,
            "peak_reserved": -1,
            "losses": []
        }
    except MemoryError as cpu_oom:
        logger.warning("CPU Out of Memory Error occurred.")
        training_stats = {
            "model_size": -1,
            "execution_time": -1,
            "peak_memory": -1,
            "peak_reserved": -1,
            "losses": []
        }
    except RuntimeError as runtime_err:
        if "CUBLAS_STATUS_ALLOC_FAILED" in str(runtime_err) or "Failed to CUDA calloc" in str(runtime_err):
            logger.warning("RuntimeError related to GPU memory allocation.")
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
    training_stats["batch_sizes"] = batch_sizes

    if use_loss_coef:
        result_path = result_path.replace(".json", "_wlc.json")

    # Save the training stats
    with open(result_path, 'w') as result_file:
        json.dump(training_stats, result_file)
    logger.info(f"Training results saved to {result_path}.")

    # Ensure all processes finish before cleanup
    dist.barrier()
    cleanup()
    logger.debug("Training process completed and resources cleaned up.")
