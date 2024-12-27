import math
from typing import TypeVar, Optional, Iterator, List

import torch
from torch.utils.data import Sampler, Dataset
import torch.distributed as dist

__all__ = ["HeterogeneousDistributedSampler", ]

T_co = TypeVar('T_co', covariant=True)


# modified version of torch.utils.data.DistributedSampler
class HeterogeneousDistributedSampler(Sampler[T_co]):
    r"""Sampler that restricts data loading to subsets of the dataset with uneven lengths.

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such a case, each
    process can pass a :class:`~torch.utils.data.HeterogeneousDistributedSampler` instance as a
    :class:`~torch.utils.data.DataLoader` sampler, and load an uneven subset of the
    original dataset that is exclusive to it.

    .. note::
        Dataset is assumed to be of constant size and that any instance of it always
        returns the same elements in the same order.

    Args:
        dataset: Dataset used for sampling.
        num_replicas (int, optional): Number of processes participating in
            distributed training. By default, :attr:`world_size` is retrieved from the
            current distributed group.
        participate_ratios (List[int], optional): How much each of ranks 
            participate in training. *Does not need to sum to 1.* By default, 
            :attr:`world_size` is retrieved from the current distributed group 
            and is used to divide data evenly.
        rank (int, optional): Rank of the current process within :attr:`num_replicas`.
            By default, :attr:`rank` is retrieved from the current distributed
            group.
        shuffle (bool, optional): If ``True`` (default), sampler will shuffle the
            indices.
        seed (int, optional): random seed used to shuffle the sampler if
            :attr:`shuffle=True`. This number should be identical across all
            processes in the distributed group. Default: ``0``.
        drop_last (bool, optional): if ``True``, then the sampler will drop the
            tail of the data to make it evenly divisible across the number of
            replicas. If ``False``, the sampler will add extra indices to make
            the data evenly divisible across the replicas. Default: ``False``.

    .. warning::
        In distributed mode, calling the :meth:`set_epoch` method at
        the beginning of each epoch **before** creating the :class:`DataLoader` iterator
        is necessary to make shuffling work properly across multiple epochs. Otherwise,
        the same ordering will be always used.

    Example::

        >>> # xdoctest: +SKIP
        >>> sampler = HeterogeneousDistributedSampler(dataset) if is_distributed else None
        >>> loader = DataLoader(dataset, shuffle=(sampler is None),
        ...                     sampler=sampler)
        >>> for epoch in range(start_epoch, n_epochs):
        ...     if is_distributed:
        ...         sampler.set_epoch(epoch)
        ...     train(loader)
    """

    def __init__(self, dataset: Dataset, num_replicas: Optional[int] = None,
                 participate_ratios: Optional[List[int]] = None,
                 rank: Optional[int] = None, shuffle: bool = True,
                 seed: int = 0, drop_last: bool = False) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if participate_ratios is None:
            participate_ratios = [1] * num_replicas
        else:
            for participate_ratio in participate_ratios:
                if participate_ratio < 0:
                    raise ValueError(
                        f"Invalid participate ratios {participate_ratios}, participate ratios should be greater than zero")
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                f"Invalid rank {rank}, rank should be in the interval [0, {num_replicas - 1}]")
        if len(participate_ratios) != num_replicas:
            raise ValueError(
                f"Invalid participate ratios {participate_ratios}, participate ratios should have length equal to num replicas ({num_replicas})")
        self.dataset = dataset
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        # If the dataset length is evenly divisible by sum of participate ratios, then there
        # is no need to drop any data, since the dataset will be split equally.
        num_parts = sum(participate_ratios) # The data should be divided to this number of parts
        if self.drop_last and len(self.dataset) % num_parts != 0:  # type: ignore[arg-type]
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.block_size = math.ceil(
                (len(self.dataset) - num_parts) / num_parts  # type: ignore[arg-type]
            )
        else:
            self.block_size = math.ceil(len(self.dataset) / num_parts)  # type: ignore[arg-type]
        self.num_samples = [(self.block_size * participate_ratio) for participate_ratio in participate_ratios]
        self.total_size = self.block_size * num_parts
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self) -> Iterator[T_co]:
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
        else:
            indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[sum(self.num_samples[:self.rank]):sum(self.num_samples[:self.rank+1])]
        assert len(indices) == len(self)

        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples[self.rank]

    def set_epoch(self, epoch: int) -> None:
        r"""
        Set the epoch for this sampler.

        When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch
