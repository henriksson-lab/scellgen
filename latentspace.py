from abc import ABC
from typing import List, Optional

import abc
import torch
import functools
import loss


from torch.distributions import Normal, Poisson, Distribution
from torch.distributions import kl_divergence as kl

from hyperspherical_vae.distributions import VonMisesFisher
from hyperspherical_vae.distributions import HypersphericalUniform


######################################################################################################
######################################################################################################
######################################################################################################

class DVAElatentspace(metaclass=abc.ABCMeta):

    def __init__(
            self,
            n_dim_in: int,
            n_dim_out: int
    ):
        """
        A latent space takes a number of inputs and then outputs. These technically need not be of the same size.
        This is an abstract class meant to be inherited
        """
        self.n_dim_in = n_dim_in
        self.n_dim_out = n_dim_out

    @abc.abstractmethod
    def reparameterize(
            self,
            x: torch.Tensor,
            loss_recorder: loss.DVAEloss
    ):
        """
        Takes n-dim input and returns the distribution corresponding to the sample
        """
        pass



######################################################################################################
######################################################################################################
######################################################################################################


# https://github.com/nicola-decao/s-vae-pytorch/blob/master/examples/mnist.py
class DVAElatentspacePeriodic(DVAElatentspace):

    def __init__(
            self,
            n_dim: int = 1
    ):
        """
        A periodic latent space, S^n. Implemented using von Mises and uniform periodic distribution
        """
        super(DVAElatentspace, self).__init__(n_dim * 2, n_dim * 2)

    @abc.abstractmethod
    def reparameterize(
            self,
            x: torch.Tensor,
            loss_recorder: loss.DVAEloss
    ):
        """
        Takes n-dim input and returns a n-dim output which holds the random sampling
        """
        # Split input vector into latent space parameters
        z_dim = self.n_dim_in
        z_mean, z_var = torch.split([z_dim, z_dim])

        # compute mean and concentration of the von Mises-Fisher
        z_mean = z_mean / z_mean.norm(dim=-1, keepdim=True)
        # the `+ 1` prevent collapsing behaviors
        z_var = F.softplus(z_var) + 1

        # The distributions to compare
        q_z = VonMisesFisher(z_mean, z_var)
        p_z = HypersphericalUniform(z_dim - 1)

        loss_recorder.add_kl(torch.distributions.kl.kl_divergence(q_z, p_z).mean())
        return q_z, p_z




######################################################################################################
######################################################################################################
######################################################################################################


class DVAElatentspaceLinear(DVAElatentspace):

    def __init__(
            self,
            n_dim: int = 1
    ):
        """
        A linear latent space, N^n - the regular kind
        """
        super(DVAElatentspace, self).__init__(n_dim * 2, n_dim * 2)

    def reparameterize(
            self,
            x: torch.Tensor,
            loss_recorder: loss.DVAEloss
    ):
        """
        Takes n-dim input and returns a n-dim output which holds the random sampling
        """
        # Split input vector into latent space parameters
        z_dim = self.n_dim_in
        z_mean, z_var = torch.split([z_dim, z_dim])

        # ensure positive variance. use exp instead?
        z_var = torch.nn.functional.softplus(self.fc_var(x))

        # The distributions to compare
        q_z = torch.distributions.normal.Normal(z_mean, z_var)
        p_z = torch.distributions.normal.Normal(torch.zeros_like(z_mean), torch.ones_like(z_var))

        loss_recorder.add_kl(torch.distributions.kl.kl_divergence(q_z, p_z).sum(-1).mean())
        return q_z, p_z



######################################################################################################
######################################################################################################
######################################################################################################


class DVAElatentspaceConcat(DVAElatentspace):

    def __init__(
            self,
            spaces: List[DVAElatentspace]
    ):
        """
        This class glues multiple latent spaces together
        """
        self.spaces = spaces
        n_dim_in = functools.reduce(lambda a, b: a.n_dim_in + b.n_dim_in, spaces, 0)
        n_dim_out = functools.reduce(lambda a, b: a.n_dim_out + b.n_dim_out, spaces, 0)
        if len(spaces) < 2:
            print("Need at leas 2 latent spaces to concatenate")
            raise
        super(DVAElatentspace, self).__init__(n_dim_in, n_dim_out)

    def reparameterize(
            self,
            x: torch.Tensor,
            loss_recorder: loss.DVAEloss
    ):
        """
        Reparameterize each latent space and return the combined product
        """
        # split into subspaces and reparameterize each
        sizes = [x.n_dim for x in self.spaces]
        start_i = 0
        list_z = []
        for i, s in enumerate(sizes):
            z = self.spaces.reparameterize(
                x[range(start_i, start_i + s)],
                loss_recorder
            )
            list_z.append(z)
            start_i += s

        # return everything concatenated
        return list_z  # todo wrong. need to make a joint distribution
