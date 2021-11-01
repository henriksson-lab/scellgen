
from typing import List, Optional



class DVAEdecoder():
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

    def forward(
            self
            # todo more stuff
    ):
        pass


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
        666
        super(DVAEdecoder, n_input, n_output)
        # todo set up the Linear pytorch layers.

        # todo consider making a class like FClayer, but simpler. or reuse internally.

    def forward(
            self
            # todo more stuff
    ):
# todo


# todo network based on some meaningful biology