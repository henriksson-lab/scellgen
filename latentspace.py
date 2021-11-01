

from typing import List, Optional



class DVAElatentspace():
    """
    A latent space takes a number of inputs and then outputs. These technically need not be of the same size.

    This is an abstract class meant to be inherited
    """

    def __init__(
        self,
        n_dim_in: int,
        n_dim_out: int
    ):
        self.n_dim_in = n_dim_in
        self.n_dim_out = n_dim_out

    def foo():
        pass




class DVAElatentspacePeriodic(DVAElatentspace):
    """
    A periodic latent space, S^n. Implemented using von Mises and uniform periodic distribution
    """

    def __init__(
        self,
        n_dim: int = 1
    ):
        super(DVAElatentspace, self).__init__(n_dim*2, n_dim*2)



class DVAElatentspaceSizefactor(DVAElatentspace):
    """
    This corresponds to the l-space in the SCVI model

    """
    def __init__(
        self
    ):
        # TODO this class likely needs a bit special treatment



class DVAElatentspaceConcat(DVAElatentspace):
    """
    This glues multiple latent spaces together
    """

    def __init__(
        self,
        spaces: List[DVAElatentspace]
    ):

        self.spaces = spaces

        # TODO sum up the dimensions of the spaces
        super(DVAElatentspace, self).__init__(n_dim*2, n_dim*2)
