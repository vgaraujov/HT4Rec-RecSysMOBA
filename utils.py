import torch


def delete(arr: torch.Tensor, ind: int, dim: int) -> torch.Tensor:
    skip = [i for i in range(arr.size(dim)) if i not in ind]
    indices = [slice(None) if i != dim else skip for i in range(arr.ndim)]
    return arr.__getitem__(indices)