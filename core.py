from typing import List, Optional, Union, Dict
import abc
import operator
import functools

import anndata

import torch
from torch.distributions import Distribution

from matplotlib import pyplot as plt

import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout

######################################################################################################
######################################################################################################
######################################################################################################
import _dataloader
import _util


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

    def __str__(self):
        """
        This method is used when the class is cast to a string
        """
        return str(self._losses)


    def add_losses(
            self,
            loss_dict: Dict[str,float]
    ) -> Dict[str,float]:
        for (loss_name, one_loss) in self._losses.items():
            one_loss = float(one_loss.cpu())
            if loss_name in loss_dict:
                loss_dict[loss_name] = loss_dict[loss_name] + one_loss
            else:
                loss_dict[loss_name] = one_loss
        return loss_dict


######################################################################################################
######################################################################################################
######################################################################################################


class DVAEstep(torch.nn.Module, metaclass=abc.ABCMeta):

    def __init__(
            self
    ):
        """
        A step in which computation is performed or data loaded. It inherits nn.Module
        to ensure that parameters() function as it is supposed to. This means that all
        modules need be stored in self. See
        https://discuss.pytorch.org/t/how-does-parameter-work/11960
        """
        super().__init__()

    @abc.abstractmethod
    def forward(
            self,
            model: 'DVAEmodel',
            env: 'Environment',
            loss_recorder: DVAEloss,
            do_sampling: bool
    ):
        """
        Performs the computation
        """
        pass

    @abc.abstractmethod
    def define_outputs(
            self,
            model: 'DVAEmodel',
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
            self
    ):
        """
        Loads data from the adata, including preprocessing as needed
        """

    @abc.abstractmethod
    def define_outputs(
            self,
            model: 'DVAEmodel',
            env: 'Environment',
    ):
        """
        Register the outputs and information about them
        """
        pass

    @abc.abstractmethod
    def get_dataset(
            self,
            model: 'DVAEmodel'
    ) -> Dict[str, torch.utils.data.Dataset]:
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
        self._variable_dims = dict()
        self._variable_value = dict()
        self._variable_sample = dict()
        self._variable_source = dict()
        self._variable_destination = dict()
        self.debug = False

    def define_variable_output(
            self,
            from_module,  # todo 666
            output_name: str,
            dim: int
    ):
        """
        Define an output from a step or loader
        """
        if output_name in self._variable_dims:
            raise "Tried to add output {} but it already existed from another step".format(output_name)
        else:
            self._variable_dims[output_name] = dim
            self._variable_source[output_name] = from_module

    def _define_variable_inputs_one(
            self,
            to_module,
            one_variable
    ):
        """
        Define where one input/variable will go
        Returns dimension of input
        """
        if isinstance(one_variable, str):
            if one_variable in self._variable_dims:
                if one_variable not in self._variable_destination:
                    one_variable_input = []
                    self._variable_destination[one_variable] = one_variable_input
                else:
                    one_variable_input = self._variable_destination[one_variable]
                one_variable_input.append(to_module)
                return self._variable_dims[one_variable]
            else:
                raise Exception("not stored: {}".format(one_variable))
        else:
            raise Exception("not implemented yet, subsets of inputs")

    def define_variable_inputs(
            self,
            to_module,
            inputs
    ):
        """
        Define where several inputs/variables will go.
        Return total dimension of inputs
        """
        total_dim = 0
        if inputs is None:
            return 0
        if not isinstance(inputs, list):
            inputs = [inputs]
        for i in inputs:
            total_dim += self._define_variable_inputs_one(to_module, i)
        return total_dim

    def get_variable_as_tensor(self, inputs):
        """
        Get a tensor for the values that have stored either by a loader or another computational step.
        If a Distribution was stored, then instead obtain a sample
        """

        # Ensure the input is a list of items
        if inputs is None:
            return None  # todo or return empty Tensor?
            # raise Exception("No inputs")
        if not isinstance(inputs, list):
            inputs = [inputs]

        # Gather all inputs
        all_values = []
        for one_input in inputs:
            value = self._variable_value[one_input]
            if isinstance(value, Distribution):
                # Distributions must be sampled. Cache this value in case it is used multiple times
                if one_input not in self._variable_sample:
                    one_sample = value.sample()
                    self._variable_sample[one_input] = one_sample
                    return one_sample
                else:
                    return self._variable_sample[one_input]
            elif isinstance(value, torch.Tensor):
                # Tensors can be added right away
                all_values.append(value)
            else:
                raise Exception("Unknown type of variable {}, being {}".format(one_input, type(value)))
        return _util.cat_tensor_with_nones(all_values).type(torch.FloatTensor)  # todo nasty type cast. better way?

    def store_variable(
            self,
            output: str,
            out: Union[torch.Tensor, Distribution]
    ):
        """
        Store one value in the environment. Can be a Tensor or Distribution
        """
        self._variable_value[output] = out
        if self.debug:
            if hasattr(out, "shape"):
                print("Storing variable {}, {}".format(output, out.shape))
            else:
                print("Storing variable {}, {}".format(output, type(out)))

    def clear_variables(self):
        self._variable_value = {}
        self._variable_sample = {}

    def print_variable_defs(self):
        print("------------------------ sizes of variables ---------------------")
        print(self._variable_dims)
        print("---------------------- where variables are used as inputs -------")

        def kn2(x):
            return [(key, [c.__class__.__name__ for c in n]) for (key, n) in x.items()]

        print(kn2(self._variable_destination))
        print("---------------------- where variables are outputs --------------")

        def kn(x):
            return [(key, n.__class__.__name__) for (key, n) in x.items()]

        print(kn(self._variable_source))
        print("-----------------------------------------------------------------")

    def call_graph(self, show = True, saveas = None):
        self.show = show
        self.saveas = saveas

        def plot(generic_nodes, generic_edges):

            generic_graph = nx.DiGraph()

            i = 0
            edge_labels = dict()
            for node, weight in zip(generic_nodes, generic_edges):

                if i + 1 == len(generic_nodes):
                    break
                node1 = node
                node2 = generic_nodes[i + 1]
                generic_graph.add_edge(node1, node2, labels=str(weight))
                edge_labels[(node1, node2)] = weight
                i += 1

            # Draw the graph

            pos = graphviz_layout(generic_graph, prog='dot')
            nx.draw_networkx(generic_graph, pos=pos, font_size=6, node_shape="s", arrows = False)
            # draw everything but the edge labels
            nx.draw_networkx_edge_labels(generic_graph, pos=pos, edge_labels=edge_labels, font_size=6)



        def plot_call_graph(x_variables,y_variables):

            in_nodes = []
            in_weights = []
            for key, value in x_variables.items():
                for module_name in value:
                    in_nodes.append(key)
                    in_weights.append(module_name.__class__.__name__)

            out_nodes = []
            out_weights = []
            for key, module_name in y_variables.items():
                out_nodes.append(key)
                out_weights.append(module_name.__class__.__name__)

            print("=======input nodes=======")
            print(in_nodes)
            print(in_weights)

            print("=======output nodes=======")
            print(out_nodes)
            print(out_weights)


            if self.show:
                plot(in_nodes, in_weights)
                plot(out_nodes, out_weights)
                plt.show()
            else:
                f, _ = plt.subplots(figsize=(8,8))

                plot(in_nodes, in_weights)
                plot(out_nodes, out_weights)
                f.savefig(self.saveas, format="PDF")

        plot_call_graph(self._variable_destination, self._variable_source)



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
        step.define_outputs(self, self.env)

    def add_loader(
            self,
            loader: DVAEloader
    ):
        """
        Add a data loader
        """
        self._loaders.append(loader)
        loader.define_outputs(self, self.env)

    def get_latent_representation(self):
        """
        Not sure this should be here at all, but this is the SCVI name of the method
        """
        pass

    def forward(
            self,
            input_data: Dict[str, None],
            do_sampling: False
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
                self,
                self.env,
                loss_recorder,
                do_sampling
            )
        return loss_recorder

    def get_dataset(self) -> torch.utils.data.Dataset:
        """
        Obtain a dataloader given the definitions in the object
        """
        all_datasets = {}
        for one_loader in self._loaders:
            these_datasets = one_loader.get_dataset(self)
            for (k, v) in these_datasets.items():
                all_datasets[k] = v
                # todo do a sanity check here
        return _dataloader.ConcatDictDataset(all_datasets)

