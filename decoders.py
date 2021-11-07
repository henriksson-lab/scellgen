from typing import List, Optional, Literal

import torch
import torch.nn as nn
from torch.distributions import Normal, Poisson, Distribution

from fromscvi import ZeroInflatedNegativeBinomial, NegativeBinomial

import core


######################################################################################################
######################################################################################################
######################################################################################################

# todo a decoder into binary categories. binary cross entropy


######################################################################################################
######################################################################################################
######################################################################################################

class DVAEdecoderRnaseq(core.DVAEstep):
    def __init__(
            self,
            mod: core.DVAEmodel,
            inputs,  # complex object!
            gene_list: List[str],
            output: str = "rnaseq_count",
            covariates=None,  # complex object!
            n_hidden: int = 128,

            dispersion: Literal["gene", "gene-batch", "gene-label", "gene-cell"] = "gene",
            gene_likelihood: Literal["zinb", "nb", "poisson"] = "zinb",

    ):
        """
        Generative function for RNAseq data according to the SCVI model
        """
        super().__init__(mod)
        self._inputs = inputs
        self._covariates = covariates
        self._gene_list = gene_list
        self._output = output
        n_output = len(gene_list)

        # Check input size and ensure it is there. Then define the output
        n_input = mod.env.get_variable_dims(inputs)
        n_covariates = mod.env.get_variable_dims(covariates)
        mod.env.define_variable(output, n_output)

        self.dispersion = dispersion
        self.gene_likelihood = gene_likelihood

        # mean gamma
        self.px_scale_decoder = nn.Sequential(
            nn.Linear(n_hidden, n_output),
            nn.Softmax(dim=-1),
        )

        # dispersion. no covariates handled yet
        self.px_r_decoder = nn.Linear(n_hidden, n_output)

        # dropout
        self.px_dropout_decoder = nn.Linear(n_hidden, n_output)

    def forward(
            self,
            env: core.Environment,
            loss_recorder: core.DVAEloss
    ):
        """
        Perform the decoding into distributions representing RNAseq counts
        """
        library = env.get_variable_as_tensor(self._input_sf)
        z = env.get_variable_as_tensor(self._inputs)
        cov = env.get_variable_as_tensor(self._covariates)

        ############################ https://github.com/YosefLab/scvi-tools/blob/9855238ae13543aefd212716f4731446bb2922bb/scvi/nn/_base_components.py

        # px is rho in https://www.nature.com/articles/s41592-018-0229-2
        rho = self.rho_decoder(torch.cat(z, cov))
        px_scale = self.px_scale_decoder(rho)
        px_rate = torch.exp(library) * px_scale
        px_dropout = self.px_dropout_decoder(rho)

        ############################
        # self.dispersion is a messy parameter. we could instead have a separate
        # cov variable that is fed into this neural network. then we would support
        # all cases in one mechanism
        px_r = torch.exp(self.px_r_decoder(rho))

        # Store the fitted distribution of gene counts
        if self.gene_likelihood == "zinb":
            count_distribution = ZeroInflatedNegativeBinomial(mu=px_rate, theta=px_r, zi_logits=px_dropout)
        elif self.gene_likelihood == "nb":
            count_distribution = NegativeBinomial(mu=px_rate, theta=px_r)
        elif self.gene_likelihood == "poisson":
            count_distribution = Poisson(px_rate)
        else:
            raise "Unsupported gene likelihood {}".format(self.gene_likelihood)

        env.store_variable(self._output, count_distribution)
