import collections
from typing import Callable, Iterable, List, Optional
import abc

import torch
import torch.nn as nn

import loss


class DVAEencoder(metaclass=abc.ABCMeta):
    """
    Takes data and crunches it down to latent space input.

    This is an abstract class meant to be inherited
    """

    def __init__(self, n_input, n_output):
        self.n_input = n_input
        self.n_output = n_output

    @abc.abstractmethod
    def forward(
            self,
            x: torch.Tensor,
            loss_recorder: loss.DVAEloss
    ):
        pass


######################################################################################################
######################################################################################################
######################################################################################################


class DVAEencoderFC(DVAEencoder):
    def __init__(
            self,
            n_input: int,
            n_output: int,
            n_covariates: int,
            n_layers: int = 1,
            n_hidden: int = 128
    ):
        """
        Fully connected neural network encoder
        """
        super().__init__(n_input, n_output)
        self.layer = FullyConnectedLayers(
            n_in=n_input,
            n_out=n_output,
            n_covariates=n_covariates,
            n_hidden=n_hidden,
            n_layers=n_layers
        )

    def forward(
            self,
            x: torch.Tensor,
            loss_recorder: loss.DVAEloss
    ):
        return self.layer.forward(x)


# #####################################################################################################
# ################# helper class ######################################################################
# #####################################################################################################


class SequentialInject(nn.Module):

    def __init__(
            self,
            modules: [nn.Module],
            n_input: int,
            n_covariates: int,
            inject_covariates: bool = True,
    ):
        """
        Connect modules sequentially. The input is assumed to be of dimension n_input + n_cov.
        if inject_covariates is set then the covariate part of the tensor will be added as input to each module.
        """
        super().__init__()
        self.modules = modules
        self.n_input = n_input
        self.n_covariates = n_covariates
        self.inject_covariates = inject_covariates

    def forward(self, x):
        x_main, x_cov = torch.split(x, [self.n_input, self.n_covariates])
        for i, one_module in enumerate(self.modules):
            if i == 0 or self.inject_covariates:
                x_main = one_module(torch.cat(x_main, x_cov))
            else:
                x_main = one_module(torch.cat(x_main))
        return x_main


# #####################################################################################################
# ################# helper class ######################################################################
# #####################################################################################################

class FullyConnectedLayers(nn.Module):
    """
    Taken and modified from SCVI. Their license applies (move this code in the future).
    Only added support for weights, which could be pulled out of this class if needed

    I would write the covariate injection rather differently. It is possible to define the network
    such that the forward function is a simple layer(x). This just means that we should not use
    nn.sequential and instead explicitly construct inputs that feed the covariates in at that point.
    The code is frankly a horror show right now


    A helper class to build fully-connected layers for a neural network.
    Parameters
    ----------
    n_in
        The dimensionality of the input, NOT COUNTING COVARIATES
    n_out
        The dimensionality of the output
    n_cat_list
        A list containing, for each category of interest,
        the number of categories. Each category will be
        included using a one-hot encoding.
    n_layers
        The number of fully-connected hidden layers
    n_hidden
        The number of nodes per hidden layer
    dropout_rate
        Dropout rate to apply to each of the hidden layers
    use_batch_norm
        Whether to have `BatchNorm` layers or not
    use_layer_norm
        Whether to have `LayerNorm` layers or not
    use_activation
        Whether to have layer activation or not
    bias
        Whether to learn bias in linear layers or not
    inject_covariates
        Whether to inject covariates in each layer, or just the first (default).
    activation_fn
        Which activation function to use
    """

    def __init__(
            self,
            n_in: int,
            n_out: int,
            n_covariates: int,
            n_layers: int = 1,
            n_hidden: int = 128,
            dropout_rate: float = 0.1,
            use_batch_norm: bool = True,
            use_layer_norm: bool = False,
            use_activation: bool = True,
            bias: bool = True,
            inject_covariates: bool = True,
            activation_fn: nn.Module = nn.ReLU,
    ):
        super().__init__()
        layers_dim = [n_in] + (n_layers - 1) * [n_hidden] + [n_out]

        # Construct each layer
        total_layers = collections.OrderedDict()
        self.linear_layers = []
        for i, (one_n_in, one_n_out) in enumerate(zip(layers_dim[:-1], layers_dim[1:])):

            if i == 0 or inject_covariates:
                one_n_in += n_covariates

            # The layer that represents information flow. These are the weights we may wish to bias
            one_layer = nn.Linear(
                one_n_in,
                one_n_out,
                bias=bias
            )
            self.linear_layers.append(one_layer)

            # Normalize this layer such that convergence improves. Add activation function at the end
            if use_batch_norm:
                # read https://arxiv.org/pdf/1502.03167.pdf
                one_layer = nn.Sequential(one_layer, nn.BatchNorm1d(one_n_in, momentum=0.01, eps=0.001))
            if use_layer_norm:
                # read https://arxiv.org/pdf/1607.06450.pdf
                one_layer = nn.Sequential(one_layer, nn.LayerNorm(one_n_in, elementwise_affine=False))
            if use_activation:
                one_layer = nn.Sequential(one_layer, activation_fn())
            if dropout_rate > 0:
                one_layer = nn.Sequential(one_layer, nn.Dropout(p=dropout_rate))

            # Add this layer to the big list
            total_layers["Layer " + str(i)] = one_layer

        # Assemble all layers into one neural net
        self.fc_layers = SequentialInject(total_layers, n_in, n_covariates, inject_covariates)

    def get_weights(self) -> List[torch.Tensor]:
        """
        Get the weights of the layers

        Returns
        -------
        A list of weights for each layer
        """
        return [nn.weights for nn in self.linear_layers]

    def forward(self, x: torch.Tensor):
        return self.fc_layers(x)
