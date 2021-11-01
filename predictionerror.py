from typing import List, Optional

from anndata import AnnData

import decoders


class DVAEpredictionError():
    """
    Compares input with output and returns the loss

    This is an abstract class meant to be inherited
    """

    def get_loss(
            self
            # todo more stuff
    ):
        pass

######################################################################################################
######################################################################################################
######################################################################################################


class DVAEpredictionZINB(DVAEpredictionError):
    """
    Returns loss based on a fitted ZINB distribution
    """

    def __init__(
            self,
            decoder: decoders.DVAEdecoderRnaseq
    ):
        pass  # hmmm

    def get_loss(
            self
            # todo more stuff
            # todo how do we forward the fitted distribution to this funcion?
            # the error can get the decoder as an argument to pull this out
    ):
        pass



# todo might need a logistic regression class, if we want to be able to predict labels