from typing import List

import torch


def cat_torch_with_nones(
        tensors: List[torch.Tensor]
):
    """
    Concatenate a list of tensors. If any are None, they are excluded
    """
    tensors = [t for t in tensors if t is not None]
    print(tensors)
    if len(tensors) == 1:
        return tensors[0]
    elif len(tensors) > 0:
        return torch.cat(*tensors)
    else:
        return None
