from typing import List

import torch


def cat_tensor_with_nones(
        tensors: List[torch.Tensor]
):
    """
    Concatenate a list of tensors. If any are None, they are excluded
    """
    tensors = [t for t in tensors if t is not None]
    if len(tensors) == 1:
        return tensors[0]
    elif len(tensors) == 2:
        return torch.cat((tensors[0], tensors[1]), dim=1)
    elif len(tensors) == 0:
        return None
    else:
        raise Exception("not implemented")
