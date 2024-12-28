import functools

from torch import nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.models.vision_transformer import vit_b_32, ViT_B_32_Weights
from torch.optim import AdamW

from heterogeneous_distributed_sampler import HeterogeneousDistributedSampler


def get_model_optim_loss():
    model = vit_b_32(weights=ViT_B_32_Weights.IMAGENET1K_V1)
    optimizer_cls = functools.partial(AdamW, lr=1e-4)
    loss = nn.CrossEntropyLoss()
    return model, optimizer_cls, loss

def get_dataset_and_dataloader(world_size, rank, batch_sizes, num_workers):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = datasets.CIFAR10(root='../data', train=True, transform=transform, download=True)
    sampler = HeterogeneousDistributedSampler(dataset, num_replicas=world_size, participate_ratios=batch_sizes, rank=rank, drop_last=True)
    dataloader = DataLoader(dataset, batch_size=batch_sizes[rank], sampler=sampler, pin_memory=True, num_workers=num_workers)
    return dataset, dataloader
