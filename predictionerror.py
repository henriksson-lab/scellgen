from typing import List, Optional

import abc

import torch.nn as nn

######################################################################################################
######################################################################################################
######################################################################################################

class DVAEpredictionError(metaclass=abc.ABCMeta):
    def __init__(
            self
    ):
        """
        Compares input with output and returns the loss

        This is an abstract class meant to be inherited
        """
        pass

    @abc.abstractmethod
    def get_loss(
            self,
            encoder_output,
            input_x):
        pass


######################################################################################################
######################################################################################################
######################################################################################################


class DVAEpredictionErrorCE(DVAEpredictionError):

    def __init__(self):
        """
        Returns loss based on -log p, where p is based on a distribution to be provided
        """
        super().__init__()

    def get_loss(self, encoder_output, input_x):
        # todo pull out the dim
        return nn.CrossEntropyLoss(reduction='none')(self.output_x, input_x.reshape(-1, self.input_dim)).sum(-1).mean()


######################################################################################################
######################################################################################################
######################################################################################################


class DVAEpredictionErrorLogp(DVAEpredictionError):

    def __init__(self):
        """
        Returns loss based on -log p, where p is based on a distribution to be provided
        """
        super().__init__()

    """
    Compare https://github.com/YosefLab/scvi-tools/blob/ac0c3e04fcc2772fdcf7de4de819db3af9465b6b/scvi/module/_vae.py#L458
    """

    def get_loss(
            self,
            encoder_output,
            input_x
    ):
        return -encoder_output.log_prob(input_x).sum(dim=-1)

# todo might need a logistic regression class, if we want to be able to predict labels
