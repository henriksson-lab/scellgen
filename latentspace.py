from typing import List, Optional

import functools
import model

import torch
import torch.nn.functional as F

from hyperspherical_vae.distributions import VonMisesFisher
from hyperspherical_vae.distributions import HypersphericalUniform


######################################################################################################
######################################################################################################
######################################################################################################


# https://github.com/nicola-decao/s-vae-pytorch/blob/master/examples/mnist.py
class DVAElatentspacePeriodic(model.DVAEstep):

    def __init__(
            self,
            mod: model.DVAEmodel,
            inputs,  # complex object!
            output: str
    ):
        """
        A periodic latent space, S^n. Implemented using von Mises and uniform periodic distribution
        """
        super().__init__(mod)

        self._inputs = inputs
        self._output = output

        # Check input size and ensure it is there
        n_input = mod.env.get_variable_dims(inputs)

        # For latent spaces, the input and output coordinate dimensions are generally the same
        self._z_dim = n_input
        mod.env.define_variable(output, self._z_dim) # todo should be half number outputs

        if not (n_input % 2 == 0 and n_input > 0):
            raise Exception(
                "Periodic latent spaces need an even number of inputs, representing mean and average. Got {}".
                format(n_input))

    def forward(
            self,
            env: model.Environment,
            loss_recorder: model.DVAEloss
    ):
        """
        Perform the reparameterization
        """
        # Split input vector into latent space parameters
        z_dim = self._z_dim
        z_mean, z_var = torch.split([z_dim, z_dim])

        # compute mean and concentration of the von Mises-Fisher
        z_mean = z_mean / z_mean.norm(dim=-1, keepdim=True)
        # the `+ 1` prevent collapsing behaviors
        z_var = F.softplus(z_var) + 1

        # The distributions to compare
        q_z = VonMisesFisher(z_mean, z_var)
        p_z = HypersphericalUniform(z_dim - 1)

        loss_recorder.add_kl(torch.distributions.kl.kl_divergence(q_z, p_z).mean())

        env.store_variable(self._output, q_z)


######################################################################################################
######################################################################################################
######################################################################################################


class DVAElatentspaceLinear(model.DVAEstep):

    def __init__(
            self,
            mod: model.DVAEmodel,
            inputs,  # complex object!
            output: str
    ):
        """
        A linear latent space, N^n - the regular kind
        """
        super().__init__(mod)

        self._inputs = inputs
        self._output = output

        # Check input size and ensure it is there
        n_input = mod.env.get_variable_dims(inputs)

        # For latent spaces, the input and output coordinate dimensions are generally the same
        self._z_dim = n_input
        mod.env.define_variable(output, self._z_dim)  # todo should be half number outputs

        if not (n_input % 2 == 0 and n_input > 0):
            raise Exception(
                "Linear latent spaces need an even number of inputs, representing mean and average. Got {}".
                format(n_input))

    def forward(
            self,
            env: model.Environment,
            loss_recorder: model.DVAEloss
    ):
        """
        Perform the reparameterization
        """
        # Split input vector into latent space parameters
        z_dim = self.n_dim_in
        z_mean, z_var = torch.split([z_dim, z_dim])

        # ensure positive variance. use exp instead?
        z_var = torch.nn.functional.softplus(z_var)

        # The distributions to compare
        q_z = torch.distributions.normal.Normal(z_mean, z_var)
        p_z = torch.distributions.normal.Normal(torch.zeros_like(z_mean), torch.ones_like(z_var))

        loss_recorder.add_kl(torch.distributions.kl.kl_divergence(q_z, p_z).sum(-1).mean())
        env.store_variable(self._output, q_z)


######################################################################################################
######################################################################################################
######################################################################################################


class DVAElatentspaceSizeFactor(model.DVAEstep):

    def __init__(
            self,
            mod: model.DVAEmodel,
            inputs,  # complex object!
            output: str = "sf_latent",
            sf_empirical: str = "sf_emp"  # todo make a loader that prepares this data
    ):
        """
        A size factor latent space, N^1
        """
        super().__init__(mod)

        self.sf_empirical = sf_empirical
        self._inputs = inputs
        self._output = output

        # Check input size and ensure it is there
        n_input = mod.env.get_variable_dims(inputs)

        # For latent spaces, the input and output coordinate dimensions are generally the same
        self._z_dim = n_input
        mod.env.define_variable(output, self._z_dim)

        if not (n_input != 2):
            raise Exception(
                "Size factor latent spaces should have 2 inputs, representing mean and average. Got {}".format(n_input))

    def forward(
            self,
            env: model.Environment,
            loss_recorder: model.DVAEloss
    ):
        """
        Perform the reparameterization
        """
        # Split input vector into latent space parameters
        z_dim = self.n_dim_in
        z_mean, z_var = torch.split([z_dim, z_dim])

        # ensure positive variance. use exp instead?
        z_var = torch.nn.functional.softplus(z_var)

        # Obtain empirical distributions of sizes
        sf_empirical = env.get_variable_as_tensor(self.sf_empirical)
        sf_empirical_mean, sf_empirical_var = torch.split(sf_empirical, [1,1])

        # The distributions to compare
        q_z = torch.distributions.normal.Normal(z_mean, z_var)
        p_z = torch.distributions.normal.Normal(sf_empirical_mean, sf_empirical_var)

        loss_recorder.add_kl(torch.distributions.kl.kl_divergence(q_z, p_z).sum(-1).mean())
        env.store_variable(self._output, q_z)