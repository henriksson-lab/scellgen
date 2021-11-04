from typing import List, Optional


import decoders
import abc

import torch.nn as nn
from torch.distributions import VonMises


from anndata import AnnData
from scvi.distributions import NegativeBinomial, ZeroInflatedNegativeBinomial

class DVAEpredictionError(metaclass=abc.ABCMeta):
    """
    Compares input with output and returns the loss

    This is an abstract class meant to be inherited
    """

    @abc.abstractmethod
    def get_loss(self):
        pass

######################################################################################################
######################################################################################################
######################################################################################################


class DVAEpredictionErrorCE(DVAEpredictionError):

    def __init__(self, input_x, output_x, input_dim):
        self.input_x = input_x
        self.output_x = output_x
        self.input_dim = input_dim

    def get_loss(self):
        reconstruction_loss = nn.CrossEntropyLoss(reduction='none')(self.output_x, self.input_x.reshape(-1, self.input_dim)).sum(-1).mean()
        return reconstructiion_loss

######################################################################################################
######################################################################################################
######################################################################################################


class DVAEpredictionErrorZINB(DVAEpredictionError):
    """
    Returns loss based on a fitted ZINB distribution
    """

    def __init__(
            self,
            decoder: decoders.DVAEdecoderRnaseq
    ):
        pass  # hmmm

    """
    Compare https://github.com/YosefLab/scvi-tools/blob/ac0c3e04fcc2772fdcf7de4de819db3af9465b6b/scvi/module/_vae.py#L458
    """
    def get_loss(
            self
            # todo more stuff
            # todo how do we forward the fitted distribution to this funcion?
            # the error can get the decoder as an argument to pull this out
    ):

        #todo this class should likely have access to these values in an interpreted manner
        px_rate = generative_outputs["px_rate"]
        px_r = generative_outputs["px_r"]
        px_dropout = generative_outputs["px_dropout"]
    
        loc = generative_outputs["loc"]
        concentration = generative_outputs["concentration"]

        if self.gene_likelihood == "zinb":
            loss = (
                -ZeroInflatedNegativeBinomial(
                    mu=px_rate, theta=px_r, zi_logits=px_dropout
                )
                    .log_prob(x)
                    .sum(dim=-1)
            )
        elif self.gene_likelihood == "nb":
            loss = (
                -NegativeBinomial(mu=px_rate, theta=px_r).log_prob(x).sum(dim=-1)
            )
        elif self.gene_likelihood == "poisson":
            reconst_loss = -Poisson(px_rate).log_prob(x).sum(dim=-1)
        elif self.gene_likelihood == "vmf":
            reconst_loss = -VonMises(loc, concentration).log_prob(x).sum(dim=-1)
        return loss




# todo might need a logistic regression class, if we want to be able to predict labels
