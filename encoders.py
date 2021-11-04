from typing import List, Optional

import abc
import torch

import torch
import torch.nn as nn
import torch.nn.functional as F

from hyperspherical_vae.distributions import VonMisesFisher
from hyperspherical_vae.distributions import HypersphericalUniform

def reparameterize_gaussian(mu, var):
    return Normal(mu, var.sqrt()).rsample()

def reparameterize_vonmises(self, z_mean, z_var):
        q_z = VonMisesFisher(z_mean, z_var)
        p_z = HypersphericalUniform(self.z_dim - 1, device = device)

        return q_z, p_z

def identity(x):
    return x


class DVAEencoder(metaclass=abc.ABCMeta):
    """
    Takes data and crunches it down to latent space input.

    This is an abstract class meant to be inherited
    """


    def __init__(self, n_input, n_hidden, n_output, activation=F.relu, distribution='normal'):    

        super().__init__()
    
        self.n_input = n_input
        self.n_hidden=n_hidden
        self.n_output = n_output
        self.activation = activation
        self.distribution = distribution

        if distribution == "vmf":
            self.activation = F.relu
            # 2 hidden layers encoder
            self.fc_e0 = nn.Linear(n_input, n_hidden * 2)
            self.fc_e1 = nn.Linear(n_hidden * 2, n_hidden)

            # compute mean and concentration of the von Mises-Fisher
            self.fc_mean = nn.Linear(n_hidden, n_output)
            self.fc_var = nn.Linear(n_hidden, 1)
        else:
            self.mean_encoder = nn.Linear(n_hidden, n_output)
            self.var_encoder = nn.Linear(n_hidden, n_output)

            if distribution == "ln":
                self.z_transformation = nn.Softmax(dim=-1)
            else:
                self.z_transformation = identity
            self.var_activation = torch.exp if var_activation is None else var_activation


    @abc.abstractmethod
    def forward(self, x: torch.Tensor):
        r"""
        The forward computation for a single sample.

         #. Encodes the data into latent space using the encoder network
         #. Generates a mean \\( q_m \\) and variance \\( q_v \\)
         #. Samples a new value from an i.i.d. multivariate normal \\( \\sim Ne(q_m, \\mathbf{I}q_v) \\)
         or von Mises Fisher distribution

        Parameters
        ----------
        x
            tensor with shape (n_input,)
        cat_list
            list of category membership(s) for this sample

        Returns
        -------
        3-tuple of :py:class:`torch.Tensor`
            tensors of shape ``(n_latent,)`` for mean and var, and sample

        """
        if distribution == "vmf":

            # 2 hidden layers encoder
            x = self.activation(self.fc_e0(x))
            x = self.activation(self.fc_e1(x))

            # compute mean and concentration of the von Mises-Fisher
            z_mean = self.fc_mean(x)
            z_mean = z_mean / z_mean.norm(dim=-1, keepdim=True)
            # the `+ 1` prevent collapsing behaviors
            z_var = F.softplus(self.fc_var(x)) + 1

            # Parameters for latent distribution

            q_m, q_v = reparameterize_vonmises(z_mean, z_var)
            latent = q_m.rsample()

            return q_m, q_v, latent

        else:
            # Parameters for latent distribution
            q = self.encoder(x, *cat_list)
            q_m = self.mean_encoder(q)
            q_v = self.var_activation(self.var_encoder(q)) + self.var_eps
            latent = self.z_transformation(reparameterize_gaussian(q_m, q_v))
            return q_m, q_v, latent




    @abc.abstractmethod
    def get_loss(self) -> torch.Tensor:
        """
        This loss is primarily from priors on the weights

        Returns
        -------
        py:class:`torch.Tensor`
            tensor of shape ``(1)``
        """
        pass


######################################################################################################
######################################################################################################
######################################################################################################


class DVAEencoderFC(DVAEencoder):
    """
    Fully connected neural network encoder
    """


    def __init__(
            self,
            n_input: int,
            n_output: int,
            n_hidden: int = 128
    ):
        super().__init__(n_input, n_output)
        # todo set up the Linear pytorch layers

        self.layer = FCLayersSCVI(n_in, n_output, n_hidden)

    def forward(
            self,
            x: torch.Tensor
    ) -> torch.Tensor:
        return self.layer.forward(x)

    def get_loss(self) -> torch.Tensor:
        666


#todo network based on some meaningful biology











######################################################################################################
######################################################################################################
######################################################################################################

class FCLayersSCVI(nn.Module):
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
        The dimensionality of the input
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
        n_cat_list: Iterable[int] = None,
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
        self.inject_covariates = inject_covariates
        layers_dim = [n_in] + (n_layers - 1) * [n_hidden] + [n_out]

        if n_cat_list is not None:
            # n_cat = 1 will be ignored
            self.n_cat_list = [n_cat if n_cat > 1 else 0 for n_cat in n_cat_list]
        else:
            self.n_cat_list = []

        # Construct each layer
        total_layers = collections.OrderedDict()
        self.linear_layers = []
        for i, (n_in, n_out) in enumerate(zip(layers_dim[:-1], layers_dim[1:])):

            # The layer that represents information flow. These are the weights we may wish to bias
            linear_layer = nn.Linear(
                n_in + cat_dim * self.inject_into_layer(i),
                n_out,
                bias=bias
            )
            self.linear_layers.append(linear_layer)

            # Normalize this layer such that convergence improves. Add activation function at the end
            norm_onelayer = nn.Sequential(
                linear_layer,
                nn.BatchNorm1d(n_out, momentum=0.01, eps=0.001)  # read https://arxiv.org/pdf/1502.03167.pdf
                if use_batch_norm
                else None,
                nn.LayerNorm(n_out, elementwise_affine=False)  # read https://arxiv.org/pdf/1607.06450.pdf
                if use_layer_norm
                else None,
                activation_fn() if use_activation else None,
                nn.Dropout(p=dropout_rate) if dropout_rate > 0 else None,
            )

            #todo store this layer for later forward().

            # Add this layer to the big list
            total_layers.update("Layer {}".format(i), norm_onelayer)

        cat_dim = sum(self.n_cat_list)
        self.fc_layers = nn.Sequential(total_layers)   #todo cannot use this function

    def get_weights(self) -> List[torch.Tensor]:
        """
        Get the weights of the layers

        Returns
        -------
        A list of weights for each layer
        """
        return [nn.weights for nn in self.linear_layers]



    #todo do we really need?
    def inject_into_layer(self, layer_num) -> bool:
        """Helper to determine if covariates should be injected."""
        user_cond = layer_num == 0 or (layer_num > 0 and self.inject_covariates)
        return user_cond



    def forward(self, x: torch.Tensor, *cat_list: int):
        """
        Forward computation on ``x``.
        Parameters
        ----------
        x
            tensor of values with shape ``(n_in,)``
        cat_list
            list of category membership(s) for this sample
        x: torch.Tensor
        Returns
        -------
        py:class:`torch.Tensor`
            tensor of shape ``(n_out,)``
        """
        one_hot_cat_list = []  # for generality in this list many indices useless.

        if len(self.n_cat_list) > len(cat_list):
            raise ValueError(
                "nb. categorical args provided doesn't match init. params."
            )
        for n_cat, cat in zip(self.n_cat_list, cat_list):
            if n_cat and cat is None:
                raise ValueError("cat not provided while n_cat != 0 in init. params.")
            if n_cat > 1:  # n_cat = 1 will be ignored - no additional information
                if cat.size(1) != n_cat:
                    one_hot_cat = one_hot(cat, n_cat)
                else:
                    one_hot_cat = cat  # cat has already been one_hot encoded
                one_hot_cat_list += [one_hot_cat]
        for i, layers in enumerate(self.fc_layers):
            for layer in layers:
                if layer is not None:
                    if isinstance(layer, nn.BatchNorm1d):
                        if x.dim() == 3:
                            x = torch.cat(
                                [(layer(slice_x)).unsqueeze(0) for slice_x in x], dim=0
                            )
                        else:
                            x = layer(x)
                    else:
                        if isinstance(layer, nn.Linear) and self.inject_into_layer(i):
                            if x.dim() == 3:
                                one_hot_cat_list_layer = [
                                    o.unsqueeze(0).expand(
                                        (x.size(0), o.size(0), o.size(1))
                                    )
                                    for o in one_hot_cat_list
                                ]
                            else:
                                one_hot_cat_list_layer = one_hot_cat_list
                            x = torch.cat((x, *one_hot_cat_list_layer), dim=-1)
                        x = layer(x)
        return x
