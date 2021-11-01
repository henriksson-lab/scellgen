from typing import List, Optional

from anndata import AnnData


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


class DVAEpredictionZINB(DVAEpredictionError):
    """
    Returns loss based on a fitted ZINB distribution
    """

    def __init__(self
                 ):
        pass  # hmmm

    def get_loss(
            self
            # todo more stuff
            # todo how do we forward the fitted distribution to this funcion?
    ):
        pass
