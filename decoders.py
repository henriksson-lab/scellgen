from typing import List, Optional

import abc


class DVAEdecoder(metaclass=abc.ABCMeta):
    """
    Takes data and crunches it down to latent space input.

    This is an abstract class meant to be inherited
    """

    def __init__(
            self,
            n_input,
            n_output
    ):
        self.n_input = n_input
        self.n_output = n_output

    """
    Generate samples given the latent space coordinates
    """
    @abc.abstractmethod
    def forward(
            self
            # todo more stuff
    ):
        pass

    """
    This loss is primarily from priors on the weights
    """

    @abc.abstractmethod
    def get_loss(self):
        pass


######################################################################################################
######################################################################################################
######################################################################################################


class DVAEdecoderFC(DVAEdecoder):
    """
    Fully connected neural network decoder.
    Will likely never be used except for testing stuff
    """

    def __init__(
            self,
            n_input: int,
            n_output: int,
            n_hidden: List[int]
    ):
        super(DVAEdecoder, n_input, n_output)
        # todo set up the Linear pytorch layers.

        # todo consider making a class like FClayer, but simpler. or reuse internally.

    def forward(
            self
            # todo more stuff
    ):

    # todo

    """
    This loss is primarily from priors on the weights
    """

    def get_loss(self):
        pass


# todo network based on some meaningful biology


######################################################################################################
######################################################################################################
######################################################################################################

class DVAEdecoderRnaseq(DVAEdecoder):
    """
    The SCVI decoder
    """

    def __init__(
            self,
            n_input: int,
            n_output: int
    ):
        666
        super(DVAEdecoder, n_input, n_output)
        # todo set up the Linear pytorch layers.

    def forward(
            self
            # todo more stuff
    ):

    # todo

    """
    This loss is primarily from priors on the weights
    """

    def get_loss(self):
        pass
