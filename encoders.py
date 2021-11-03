from typing import List, Optional

import abc

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



    """
    This loss is primarily from priors on the weights
    """
    @abc.abstractmethod
    def get_loss(self):
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
            n_hidden: List[int]
    ):
        super(DVAEencoder, n_input, n_output)
        # todo set up the Linear pytorch layers

    def forward(
            self
            #todo more stuff
    ):
        pass
        # todo

    def get_loss(self):
        pass


#todo network based on some meaningful biology
