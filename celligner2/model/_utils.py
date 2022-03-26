import torch
import numpy as np
import logging

logger = logging.getLogger(__name__)


def one_hot_encoder(idx, n_cls):
    """
    One hot encoder for categorical features

    Args:
        idx: index of the categorical feature
        n_cls: number of classes

    Returns:
        one hot encoded tensor
    """
    if idx.dim() == 1:
        idx = idx.unsqueeze(1)
    onehot = torch.zeros(idx.size(0), n_cls)
    onehot = onehot.to(idx.device)
    onehot.scatter_(1, idx.long(), 1)
    return onehot


def partition(data, partitions, num_partitions, main_partition=None):
    res = []
    main = []
    partdim = partitions.shape[1]
    for i in torch.unique(partitions, dim=0):
        if main_partition is not None and i[main_partition]:
            main += [data[(partitions == i).sum(1) == partdim]]
        else:
            res += [data[(partitions == i).sum(1) == partdim]]
    return [torch.cat(main)] + res if main_partition is not None else res
