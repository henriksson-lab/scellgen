from typing import List, Optional
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import torch
import torch.nn as nn
from torch.distributions import Normal, Poisson, Distribution

import _util
import encoders
import predictionerror
from fromscvi import ZeroInflatedNegativeBinomial, NegativeBinomial

import core


######################################################################################################
######################################################################################################
######################################################################################################

# todo a decoder into binary categories. binary cross entropy


######################################################################################################
######################################################################################################
######################################################################################################

# todo a decoder into linear values. L2 loss

class DVAEdecoderFC(core.DVAEstep):
    def __init__(
            self,
            mod: core.DVAEmodel,
            inputs,  # complex object!
            gene_list: List[str] = None,
            output: str = "rnaseq_count",
            covariates=None,  # complex object!
            n_hidden: int = 128
    ):
        """
        Generative function for various continuous data
        """
        super().__init__()

        # Generate all genes if not specified
        if gene_list is None:
            # todo need a way of storing gene names
            gene_list = mod.adata.var.index.tolist()

        self._inputs = inputs
        self._covariates = covariates
        self._gene_list = gene_list
        self._output = output
        self._n_output = len(gene_list)
        self._n_hidden = n_hidden

        # Check input size and ensure it is there. Then define the output
        self._n_input = mod.env.define_variable_inputs(self, inputs)
        self._n_covariates = mod.env.define_variable_inputs(self, covariates)

        self.layer = encoders.FullyConnectedLayers(
            n_in=self._n_input,
            n_hidden=self._n_hidden,
            n_out=self._n_output,
            n_covariates=self._n_covariates
        )

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
        Perform the decoding into distributions representing various continuous data
        """
        z = env.get_variable_as_tensor(self._inputs)
        cov = env.get_variable_as_tensor(self._covariates)

        px = self.layer.forward(_util.cat_tensor_with_nones([z, cov]))

        # Store the fitted distribution of gene counts
        count_distribution = Normal(px, torch.ones_like(px)*100)
        env.store_variable(self._output, count_distribution)

        # Add reconstruction error. Should it really be done here? todo
        gene_counts = env.get_variable_as_tensor("X")
        # predictionerror.DVAEpredictionErrorLogp().store_loss(
        #     gene_counts,
        #     count_distribution,
        #     loss_recorder
        # )

        predictionerror.DVAEpredictionErrorL2().store_loss(
            gene_counts,
            px,
            loss_recorder
        )


    def define_outputs(
            self,
            mod: core.DVAEmodel,
            env: core.Environment
    ):
        env.define_variable_output(self, self._output, self._n_output)


######################################################################################################
######################################################################################################
######################################################################################################

class DVAEdecoderRnaseq(core.DVAEstep):
    def __init__(
            self,
            mod: core.DVAEmodel,
            inputs,  # complex object!
            gene_list: List[str] = None,
            input_sf: str = "X_sf",
            output: str = "rnaseq_count",
            covariates=None,  # complex object!
            n_hidden: int = 128,

            # dispersion: Literal["gene", "gene-batch", "gene-label", "gene-cell"] = "gene",
            gene_likelihood: Literal["zinb", "nb", "poisson"] = "zinb",

    ):
        """
        Generative function for RNAseq data according to the SCVI model
        """
        super().__init__()

        # Generate all genes if not specified
        if gene_list is None:
            # todo need a way of storing gene names
            gene_list = mod.adata.var.index.tolist()

        self._inputs = inputs
        self._covariates = covariates
        self._gene_list = gene_list
        self._output = output
        self._n_output = len(gene_list)
        self._n_hidden = n_hidden
        self._input_sf = input_sf

        # Check input size and ensure it is there. Then define the output
        self._n_input = mod.env.define_variable_inputs(self, inputs)
        self._n_covariates = mod.env.define_variable_inputs(self, covariates)

        # self.dispersion = dispersion
        self._gene_likelihood = gene_likelihood

        self.px_decoder = encoders.FullyConnectedLayers(
            n_in=self._n_input,
            n_out=self._n_hidden,
            n_covariates=self._n_covariates
        )

        # mean gamma - rho in the SCVI paper. softmax makes the output clamp in [0,1]
        self.rho_decoder = nn.Sequential(
            nn.Linear(self._n_hidden, self._n_output),
            nn.Softmax(dim=-1),
        )

        # dispersion. no covariates handled yet
        self.px_r_decoder = nn.Linear(self._n_hidden, self._n_output)

        # dropout
        self.px_dropout_decoder = nn.Linear(self._n_hidden, self._n_output)

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
        Perform the decoding into distributions representing RNAseq counts
        """
        z = env.get_variable_as_tensor(self._inputs)
        cov = env.get_variable_as_tensor(self._covariates)

        # Take the library scale back to normal non-log scale
        library = torch.exp(env.get_variable_as_tensor(self._input_sf))

        px = self.px_decoder.forward(_util.cat_tensor_with_nones([z, cov]))

        ############################ https://github.com/YosefLab/scvi-tools/blob/9855238ae13543aefd212716f4731446bb2922bb/scvi/nn/_base_components.py

        # px is rho in https://www.nature.com/articles/s41592-018-0229-2
        rho = self.rho_decoder(px)
        px_rate = library * rho
        px_dropout = self.px_dropout_decoder(px)  # f_h in the SCVI paper

        ############################
        # self.dispersion is a messy parameter. we could instead have a separate
        # cov variable that is fed into this neural network. then we would support
        # all cases in one mechanism
        px_r = torch.exp(self.px_r_decoder(px))   # or px_rate?

        # Store the fitted distribution of gene counts
        if self._gene_likelihood == "zinb":
            count_distribution = ZeroInflatedNegativeBinomial(mu=px_rate, theta=px_r, zi_logits=px_dropout)  # todo corrected wrong?
        elif self._gene_likelihood == "nb":
            count_distribution = NegativeBinomial(mu=px_rate, theta=px_r)
        elif self._gene_likelihood == "poisson":
            count_distribution = Poisson(px_r)
        else:
            raise "Unsupported gene likelihood {}".format(self._gene_likelihood)

        if do_sampling:
            env.store_variable(self._output, count_distribution)

            # Add reconstruction error. Should it really be done here? todo
            gene_counts = env.get_variable_as_tensor("X")
            predictionerror.DVAEpredictionErrorLogp().store_loss(
                gene_counts,
                count_distribution,
                loss_recorder
            )
        else:
            env.store_variable(self._output, count_distribution.mean)  #todo later: rho
#            env.store_variable(self._output, px_rate)  #todo later: rho

    def define_outputs(
            self,
            mod: core.DVAEmodel,
            env: core.Environment
    ):
        env.define_variable_output(self, self._output, self._n_output)


    def get_normalized_expression(self):
        # this is rho
        #https://github.com/YosefLab/scvi-tools/blob/85fb04a30ca1a9162cd25e57673b9af81a03ca70/scvi/model/base/_rnamixin.py#L41
        pass

    def differential_expression(self):
        # https://github.com/YosefLab/scvi-tools/blob/85fb04a30ca1a9162cd25e57673b9af81a03ca70/scvi/model/base/_rnamixin.py#L168
        pass
