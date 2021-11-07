from typing import List, Optional, Literal

import torch
import torch.nn as nn
from torch.distributions import Normal, Poisson, Distribution

from fromscvi import ZeroInflatedNegativeBinomial, NegativeBinomial

import model


######################################################################################################
######################################################################################################
######################################################################################################
#
# class DVAEdecoder(metaclass=abc.ABCMeta):
#     """
#     Takes data and crunches it down to latent space input.
#
#     This is an abstract class meant to be inherited
#     """
#
#     def __init__(
#             self,
#             n_input,
#             n_output
#     ):
#         self.n_input = n_input
#         self.n_output = n_output
#
#     @abc.abstractmethod
#     def forward_from_point(
#             self,
#             loss_recorder: loss.DVAEloss,
#             z: torch.Tensor,
#             cov: torch.Tensor
#     ):
#         """
#         Generate samples given the latent space coordinates
#         """
#         pass
#
#     def forward_from_distribution(
#             self,
#             loss_recorder: loss.DVAEloss,
#             list_z: List[Distribution],
#             cov: torch.Tensor
#     ):
#         """
#         Generate samples given the latent space distribution
#         """
#         z_samples = [z.sample() for z in list_z]
#         # todo what about multiple samples per point?
#
#         return self.forward_from_point(loss_recorder, z_samples, cov)


######################################################################################################
######################################################################################################
######################################################################################################

# todo a decoder into binary categories. binary cross entropy


######################################################################################################
######################################################################################################
######################################################################################################

class DVAEdecoderRnaseq(model.DVAEstep):
    def __init__(
            self,
            mod: model.DVAEmodel,
            inputs,  # complex object!
            gene_list: [str],
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
        n_output = len(gene_list)

        # Check input size and ensure it is there. Then define the output
        n_input = mod.env.get_input_dims(inputs)
        n_covariates = mod.env.get_input_dims(covariates)
        mod.env.define_output(output, n_output)

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
            env: model.Environment,
            loss_recorder: model.DVAEloss
    ):
        """
        Perform the decoding
        """
        library = env.get_input_tensor(self._input_sf)
        z = env.get_input_tensor(self._inputs)
        cov = env.get_input_tensor(self._covariates)

        ############################ https://github.com/YosefLab/scvi-tools/blob/9855238ae13543aefd212716f4731446bb2922bb/scvi/nn/_base_components.py

        rho = self.rho_decoder(torch.cat(z, cov))  # px is rho in https://www.nature.com/articles/s41592-018-0229-2
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

        env.store_output(self._output, count_distribution)
