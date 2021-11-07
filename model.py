from typing import List, Optional, Union, Dict
import abc
import operator
import functools

import anndata

import torch
from torch.distributions import Distribution


######################################################################################################
######################################################################################################
######################################################################################################

class DVAEloss():

    def __init__(self):
        """
        This class keeps tracks of all types of losses
        """
        self._losses = dict()

    def add(self,
            category: str,
            loss: torch.Tensor
            ):
        """
        Add a loss with an arbitrary name
        """
        if category in self._losses:
            self._losses[category] = self._losses[category] + loss
        else:
            self._losses[category] = loss

    def add_kl(self, loss: torch.Tensor):
        """
        Add a KL-loss
        """
        self.add("kl", loss)

    def get_total_loss(self):
        """
        Get the total recorded loss
        """
        assert bool(self._losses), "No losses have been recorded"
        return functools.reduce(lambda x, y: x + y, list(self._losses.values()))

    def add_reconstruction_loss(self, loss):
        """
        Add a reconstruction loss
        """
        self.add("reconstruction", loss)


######################################################################################################
######################################################################################################
######################################################################################################


class DVAEstep(metaclass=abc.ABCMeta):

    def __init__(
            self,
            model: 'DVAEmodel'
    ):
        """
        A step in which computation is performed or data loaded
        """
        self.model = model

    @abc.abstractmethod
    def forward(
            self,
            env: 'Environment',
            loss_recorder: DVAEloss
    ):
        """
        Performs the computation
        """
        pass

    @abc.abstractmethod
    def define_outputs(self):
        """
        Register the outputs and information about them
        """
        pass


######################################################################################################
######################################################################################################
######################################################################################################

class DVAEloader(metaclass=abc.ABCMeta):

    def __init__(
            self,
            model: 'DVAEmodel'
    ):
        """
        Loads data from the adata, including preprocessing as needed
        """
        self.model = model

    @abc.abstractmethod
    def define_outputs(self):
        """
        Register the outputs and information about them
        """
        pass

    @abc.abstractmethod
    def inject_environment(
            self,
            env: 'Environment',
            data
    ) -> None:
        """
        Performs the loading into the environment
        """
        pass

######################################################################################################
######################################################################################################
######################################################################################################

class Environment:

    def __init__(
            self,
            model: 'DVAEmodel'
    ):
        """
        An environment is a place where all inputs and outputs are stored while the
        torch graph is being built

        _output_values   List of values in the environment. Tensor or Distribution
        _output_samples  Caches samples from _output_values, if it is a Distribution
        """
        self._model = model
        self._outputs = dict()
        self._output_values = dict()
        self._output_samples = dict()

    def define_variable(
            self,
            output_name: str,
            dim: int
    ):
        """
        Define an output from a step or loader
        """
        if output_name in self._outputs:
            raise "Tried to add output {} but it already existed from another step".format(output_name)
        else:
            self._outputs[output_name] = dim

    def _get_variable_dims_of_one(self, one_input):
        """
        Compute the dimensions of one input
        """
        if isinstance(one_input, str):
            if one_input in self._output_values:
                return len(self._output_values[one_input])
            else:
                raise Exception("not stored: {}".format(one_input))
        else:
            raise Exception("not implemented yet, subsets of inputs")

    def get_variable_dims(self, inputs):
        """
        Compute the dimensions of the given input
        """
        if inputs is None:
            raise Exception("No inputs")
        if not isinstance(inputs, list):
            inputs = [inputs]
        return functools.reduce(operator.add, [self._get_variable_dims_of_one(i) for i in inputs], 0)

    def get_variable_as_tensor(self, inputs):
        """
        Get a tensor for the values that have stored either by a loader or another computational step.
        If a Distribution was stored, then instead obtain a sample
        """

        # Ensure the input is a list of items
        if inputs is None:
            raise Exception("No inputs")
        if not isinstance(inputs, list):
            inputs = [inputs]

        # Gather all inputs
        all_values = []
        for one_input in inputs:
            value = self._output_values[one_input]
            if isinstance(value, Distribution):
                # Distributions must be sampled. Cache this value in case it is used multiple times
                if one_input not in self._output_samples:
                    one_sample = value.sample()
                    self._output_samples[one_input] = one_sample
                    return one_sample
                else:
                    return self._output_samples[one_input]
            elif isinstance(value, torch.Tensor):
                # Tensors can be added right away
                all_values.append(value)
            else:
                raise Exception("Unknown type of variable {}".format(one_input))
        return torch.cat(all_values)

    def store_variable(
            self,
            output: str,
            out: Union[torch.Tensor,Distribution]
    ):
        """
        Store one value in the environment. Can be a Tensor or Distribution
        """
        self._output_values[output] = out

    def clear_variables(self):
        self._output_values = []
        self._output_samples = []


######################################################################################################
######################################################################################################
######################################################################################################


class DVAEmodel:

    def __init__(self):
        """
        This class contains the definition of a VAE, along with the loaders and steps of computation
        """
        self._steps = []
        self._loaders = []
        self.env = Environment(self)

    def add_step(
            self,
            step: DVAEstep
    ):
        """
        Add a computational step to perform
        """
        self._steps.append(step)
        step.define_outputs()

    def add_loader(
            self,
            loader: DVAEloader
    ):
        """
        Add a data loader
        """
        self._loaders.append(loader)
        loader.define_outputs()

    def get_latent_representation(self):
        """
        Not sure this should be here at all, but this is the SCVI name of the method
        """
        pass

    def forward(
            self,
            input_data: List
    ) -> DVAEloss:
        """
        Perform all the steps
        """
        self.env.clear_variables()
        loss_recorder = DVAEloss()

        # Prepare the data
        for i, loader in enumerate(self._loaders):
            loader.inject_environment(
                self.env,
                input_data[i]
            )

        # Run all the steps
        for step in self._steps:
            step.forward(
                self.env,
                loss_recorder
            )
        return loss_recorder

    def parameters(self):
        """
        Return parameters for the model
        """
        # todo do properly
        return []

    def get_dataloader(self):
        """

        :return:
        """