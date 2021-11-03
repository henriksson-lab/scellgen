"""
    This code is copied straight from SCVI for future compatibility.
    BSD 3-Clause License
    Copyright (c) 2020 Romain Lopez, Adam Gayoso, Galen Xing, Yosef Lab
    All rights reserved.
"""

from typing import Callable, Dict, Iterable, Optional, Tuple, Union

import torch


class LossRecorder:
    """

    Loss signature for models.
    This class provides an organized way to record the model loss, as well as
    the components of the ELBO. This may also be used in MLE, MAP, EM methods.
    The loss is used for backpropagation during inference. The other parameters
    are used for logging/early stopping during inference.
    Parameters
    ----------
    loss
        Tensor with loss for minibatch. Should be one dimensional with one value.
        Note that loss should be a :class:`~torch.Tensor` and not the result of `.item()`.
    reconstruction_loss
        Reconstruction loss for each observation in the minibatch.
    kl_local
        KL divergence associated with each observation in the minibatch.
    kl_global
        Global kl divergence term. Should be one dimensional with one value.
    **kwargs
        Additional metrics can be passed as keyword arguments and will
        be available as attributes of the object.
    """

    def __init__(
            self,
            loss: Union[Dict[str, torch.Tensor], torch.Tensor],
            reconstruction_loss: Union[
                Dict[str, torch.Tensor], torch.Tensor
            ] = torch.tensor(0.0),
            kl_local: Union[Dict[str, torch.Tensor], torch.Tensor] = torch.tensor(0.0),
            kl_global: Union[Dict[str, torch.Tensor], torch.Tensor] = torch.tensor(0.0),
            **kwargs,
    ):
        self._loss = loss if isinstance(loss, dict) else dict(loss=loss)
        self._reconstruction_loss = (
            reconstruction_loss
            if isinstance(reconstruction_loss, dict)
            else dict(reconstruction_loss=reconstruction_loss)
        )
        self._kl_local = (
            kl_local if isinstance(kl_local, dict) else dict(kl_local=kl_local)
        )
        self._kl_global = (
            kl_global if isinstance(kl_global, dict) else dict(kl_global=kl_global)
        )
        for key, value in kwargs.items():
            setattr(self, key, value)

    @staticmethod
    def _get_dict_sum(dictionary):
        total = 0.0
        for value in dictionary.values():
            total += value
        return total

    @property
    def loss(self) -> torch.Tensor:
        return self._get_dict_sum(self._loss)

    @property
    def reconstruction_loss(self) -> torch.Tensor:
        return self._get_dict_sum(self._reconstruction_loss)

    @property
    def kl_local(self) -> torch.Tensor:
        return self._get_dict_sum(self._kl_local)

    @property
    def kl_global(self) -> torch.Tensor:
        return self._get_dict_sum(self._kl_global)
