from typing import List, Optional

import functools
import core

import torch
import torch.nn.functional as F

from hyperspherical_vae.distributions import VonMisesFisher
from hyperspherical_vae.distributions import HypersphericalUniform


######################################################################################################
######################################################################################################
######################################################################################################


class DVAElatentspacePeriodic(core.DVAEstep):

    def __init__(
            self,
            mod: core.DVAEmodel,
            inputs,  # complex object!
            output: str
    ):
        """
        A periodic latent space, S^n. Implemented using von Mises and uniform periodic distribution

        See https://github.com/nicola-decao/s-vae-pytorch/blob/master/examples/mnist.py
        """
        super().__init__()

        self._inputs = inputs
        self._output = output

        # Check input size and ensure it is there
        n_input = mod.env.define_variable_inputs(self, inputs)

        # For latent spaces, the input and output coordinate dimensions are generally the same
        self._z_dim = n_input / 2
        mod.env.define_variable_output(output, n_input)

        if not (n_input % 2 == 0 and n_input > 0):
            raise Exception(
                "Periodic latent spaces need an even number of inputs, representing mean and average. Got {}".
                    format(n_input))

        # Add this computational step to the model
        mod.add_step(self)

    def forward(
            self,
            mod: core.DVAEmodel,
            env: core.Environment,
            loss_recorder: core.DVAEloss,
            do_sampling: bool
    ):
        """
        Perform the reparameterization
        """
        # Split input vector into latent space parameters
        z_dim = self._z_dim
        z_input = env.get_variable_as_tensor(self._inputs)
        z_mean, z_var = torch.split(z_input, [z_dim, z_dim], dim=1)

        if do_sampling:
            # compute mean and concentration of the von Mises-Fisher
            z_mean = z_mean / z_mean.norm(dim=-1, keepdim=True)
            # the `+ 1` prevent collapsing behaviors
            z_var = F.softplus(z_var) + 1

            # The distributions to compare
            q_z = VonMisesFisher(z_mean, z_var)
            p_z = HypersphericalUniform(z_dim - 1)

            loss_recorder.add_kl(torch.distributions.kl.kl_divergence(q_z, p_z).mean())

            env.store_variable(self._output, q_z)
        else:
            # todo not sure this is right. check
            env.store_variable(self._output, z_mean)


    def define_outputs(
            self,
            mod: core.DVAEmodel,
            env: core.Environment,
    ):
        """
        Register the outputs and information about them
        """
        # For latent spaces, the input and output coordinate dimensions are generally the same
        _z_dim = self.n_input
        env.define_variable_output(self, self._output, _z_dim)  # todo here I think the same num dims?


######################################################################################################
######################################################################################################
######################################################################################################


class DVAElatentspaceLinear(core.DVAEstep):

    def __init__(
            self,
            mod: core.DVAEmodel,
            inputs,  # complex object!
            output: str
    ):
        """
        A linear latent space, N^n - the regular kind
        """
        super().__init__()

        self._inputs = inputs
        self._output = output

        # Check input size and ensure it is there
        self.n_input = mod.env.define_variable_inputs(self, inputs)
        if not (self.n_input % 2 == 0 and self.n_input > 0):
            raise Exception(
                "Linear latent spaces need an even number of inputs, representing mean and average. Got {}".
                    format(self.n_input))

        # Add this computational step to the model
        mod.add_step(self)

    def forward(
            self,
            mod: core.DVAEmodel,
            env: core.Environment,
            loss_recorder: core.DVAEloss,
            do_sampling: bool
    ):
        """
        Perform the reparameterization
        """
        # Split input vector into latent space parameters
        z_dim = int(self.n_input / 2)  # not sure why int needed, even if not float
        z_input = env.get_variable_as_tensor(self._inputs)
        z_mean, z_var = torch.split(z_input, [z_dim, z_dim], dim=1)

        # ensure positive variance. use exp instead?
        z_var = torch.nn.functional.softplus(z_var)

        # The distributions to compare
        q_z = torch.distributions.normal.Normal(z_mean, z_var)
        p_z = torch.distributions.normal.Normal(torch.zeros_like(z_mean), torch.ones_like(z_var))

        loss_recorder.add_kl(torch.distributions.kl.kl_divergence(q_z, p_z).sum(-1).mean())
        env.store_variable(self._output, q_z)

    def define_outputs(
            self,
            mod: core.DVAEmodel,
            env: core.Environment,
    ):
        """
        Register the outputs and information about them
        """
        # For normal latent spaces, there is one output coordinate given input mu, var
        _z_dim = int(self.n_input / 2)
        env.define_variable_output(self, self._output, _z_dim)

    def get_latent_coordinates(self):
        env = self.model.env
        z_dim = int(self.n_input / 2)
        z_input = env.get_variable_as_tensor(self._inputs)
        z_mean, z_var = torch.split(z_input, [z_dim, z_dim], dim=1)
        return z_mean

######################################################################################################
######################################################################################################
######################################################################################################


class DVAElatentspaceSizeFactor(core.DVAEstep):

    def __init__(
            self,
            mod: core.DVAEmodel,
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
        self.n_input = mod.env.define_variable_inputs(self, inputs)
        self.n_input_sf = mod.env.define_variable_inputs(self, sf_empirical)

        if self.n_input != 2:
            raise Exception(
                "Size factor latent spaces should have 2 inputs, representing mean and average. Got {}".
                    format(self.n_input))

        if self.n_input_sf != 3:
            raise Exception(
                "Size factor latent spaces should have a dim=3 sf prior, representing observed, mean and average. Got {}".
                    format(self.n_input_sf))

        # Add this computational step to the model
        mod.add_step(self)

    def forward(
            self,
            env: core.Environment,
            loss_recorder: core.DVAEloss,
            do_sampling: bool
    ):
        """
        Perform the reparameterization
        """
        # Split input vector into latent space parameters
        z_input = env.get_variable_as_tensor(self._inputs)
        z_mean, z_var = torch.split(z_input, [1, 1], dim=1)
        # ensure positive variance
        z_var = torch.exp(z_var)

        if do_sampling:
            # Obtain empirical distributions of sizes
            sf_empirical = env.get_variable_as_tensor(self.sf_empirical)
            sf_observed, sf_empirical_mean, sf_empirical_var = torch.split(sf_empirical, [1, 1, 1], dim=1)

            # The distributions to compare
            q_z = torch.distributions.normal.Normal(z_mean, z_var)
            p_z = torch.distributions.normal.Normal(sf_empirical_mean, sf_empirical_var)

            loss_recorder.add_kl(torch.distributions.kl.kl_divergence(q_z, p_z).sum(-1).mean())
            env.store_variable(self._output, q_z)
        else:
            env.store_variable(self._output, z_mean)


    def define_outputs(
            self,
            env: core.Environment,
    ):
        """
        Register the outputs and information about them
        """
        # For latent spaces, the input and output coordinate dimensions are generally the same
        env.define_variable_output(self, self._output, 1)
