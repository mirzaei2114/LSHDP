import functools

from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.fsdp.api import (
    BackwardPrefetch,
    CPUOffload,
    ShardingStrategy,
)

from torchvision.models.vision_transformer import EncoderBlock


total_batch_size = 128
num_workers = 4
results_dir = "./temp_results/"
# fsdp_kwargs = None
fsdp_kwargs = {
    "sharding_strategy": ShardingStrategy.NO_SHARD,
    "cpu_offload": CPUOffload(offload_params=True),
    "auto_wrap_policy": functools.partial(transformer_auto_wrap_policy, transformer_layer_cls=(EncoderBlock,)),
    "backward_prefetch": BackwardPrefetch.BACKWARD_PRE,
    "forward_prefetch": True,
    # "backward_prefetch": None,
    # "forward_prefetch": False,
}
