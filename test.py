import os
import torch
import torch.distributed as dist

print(os.environ['MASTER_ADDR'])
print(os.environ['MASTER_PORT'])
print(os.environ['WORLD_SIZE'])
print(os.environ['RANK'])

dist.init_process_group("gloo", rank=os.environ['RANK'], world_size=os.environ['WORLD_SIZE'])

# Dummy tensor operation
tensor = torch.ones(1) * os.environ['RANK']
print(f"Rank {os.environ['RANK']} has tensor: {tensor.item()} before communication")

# All-reduce example (sum across all nodes)
dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
print(f"Rank {os.environ['RANK']} has tensor: {tensor.item()} after all_reduce")

dist.destroy_process_group()
