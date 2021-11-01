from typing import List, Optional




class DVAEencoder():
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
            #todo more stuff
    ):
        pass


class DVAEencoderFC(DVAEencoder):
    """
    Fully connected neural network encoder
    """


    def __init__(
            self,
            n_input: int,
            n_output: int,
            n_hidden: List[int]
    ):
        666
        super(DVAEencoder, n_input, n_output)
        # todo set up the Linear pytorch layers

    def forward(
            self
            #todo more stuff
    ):
        # todo




#todo network based on some meaningful biology