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
import _dataloader


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


class DVAEstep(torch.nn.Module, metaclass=abc.ABCMeta):

    def __init__(
            self,
            model: 'DVAEmodel'
    ):
        """
        A step in which computation is performed or data loaded. It inherits nn.Module
        to ensure that parameters() function as it is supposed to. This means that all
        modules need be stored in self. See
        https://discuss.pytorch.org/t/how-does-parameter-work/11960
        """
        super().__init__()
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
    def define_outputs(
            self,
            env: 'Environment',
    ):
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
    def define_outputs(
            self,
            env: 'Environment',
    ):
        """
        Register the outputs and information about them
        """
        pass


    @abc.abstractmethod
    def get_dataset(self) -> Dict[str, torch.utils.data.Dataset]:
        """
        Return a dictionary of variable name -> Dataset
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
        self._output_dims = dict()
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
        if output_name in self._output_dims:
            raise "Tried to add output {} but it already existed from another step".format(output_name)
        else:
            self._output_dims[output_name] = dim

    def _get_variable_dims_of_one(self, one_variable):
        """
        Compute the dimensions of one input
        """
        if isinstance(one_variable, str):
            if one_variable in self._output_dims:
                return self._output_dims[one_variable]
            else:
                raise Exception("not stored: {}".format(one_variable))
        else:
            raise Exception("not implemented yet, subsets of inputs")

    def get_variable_dims(self, inputs):
        """
        Compute the dimensions of the given input
        """
        if inputs is None:
            return 0
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
            out: Union[torch.Tensor, Distribution]
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


class DVAEmodel(torch.nn.Module):

    def __init__(
            self, 
            adata: anndata.AnnData
    ):
        """
        This class contains the definition of a VAE, along with the loaders and steps of computation
        """
        super().__init__()
        self.adata = adata
        self._steps = torch.nn.ModuleList()  # this ensures that parameters() recurse this list
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
        step.define_outputs(self.env)

    def add_loader(
            self,
            loader: DVAEloader
    ):
        """
        Add a data loader
        """
        self._loaders.append(loader)
        loader.define_outputs(self.env)

    def get_latent_representation(self):
        """
        Not sure this should be here at all, but this is the SCVI name of the method
        """
        pass

    def forward(
            self,
            input_data: Dict[str, None]
    ) -> DVAEloss:
        """
        Perform all the steps
        """
        self.env.clear_variables()
        loss_recorder = DVAEloss()

        # Load the data into the environment
        for data_key, data_name in input_data.items():
            self.env.store_variable(data_key, data_name)

        # Run all the steps
        for step in self._steps:
            step.forward(
                self.env,
                loss_recorder
            )
        return loss_recorder

    def get_dataset(self) -> torch.utils.data.Dataset:
        """
        Obtain a dataloader given the definitions in the object
        """
        all_datasets = {}
        for one_loader in self._loaders:
            these_datasets = one_loader.get_dataset()
            for (k,v) in these_datasets.items():
                all_datasets[k] = v
                #todo do a sanity check here
        return _dataloader.ConcatDictDataset(all_datasets)