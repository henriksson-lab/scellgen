from typing import List, Optional

import abc

import model
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
    def store_loss(
            self,
            input_x,
            encoder_output,
            loss_recorder: model.DVAEloss
            ):
        pass


######################################################################################################
######################################################################################################
######################################################################################################


class DVAEpredictionErrorCE(DVAEpredictionError):

    def __init__(self):
        """
        Returns loss based on cross entropy. The input is expected to be a tensor.
        This loss can be used to predict classes, with values [0,1].
        """
        super().__init__()

    def store_loss(
            self,
            input_x,
            encoder_output,
            loss_recorder: model.DVAEloss
            ):
        """
        Compute the loss
        """
        input_dim = 666 # todo somehow get from encoder_output shape?
        loss = nn.CrossEntropyLoss(reduction='none')(encoder_output, input_x.reshape(-1, input_dim)).sum(-1).mean()
        loss_recorder.add_reconstruction_loss(loss)


######################################################################################################
######################################################################################################
######################################################################################################

class DVAEpredictionErrorLogp(DVAEpredictionError):

    def __init__(self):
        """
        Returns loss based on -log p, where p is based on a Distribution to be provided.
        This loss is generally suitable whenever the distribution of values is explicitly modelled
        """
        super().__init__()

    def store_loss(
            self,
            input_x,
            encoder_output,
            loss_recorder: model.DVAEloss
            ):
        """
        Compute the loss
        """
        loss = -encoder_output.log_prob(input_x).sum(dim=-1)
        loss_recorder.add_reconstruction_loss(loss)
