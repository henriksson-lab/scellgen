from typing import List, Optional

import abc

######################################################################################################
######################################################################################################
######################################################################################################

class DVAElatentspace(metaclass=abc.ABCMeta):
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


    @abc.abstractmethod
    def reparameterize(self
                       # todo more stuff
                       ):
        """
        Takes n-dim input and returns a n-dim output which holds the random sampling
        """
        pass

    """
    Calculate the loss - KL distance
    """

    @abc.abstractmethod
    def get_loss(self):
        pass


######################################################################################################
######################################################################################################
######################################################################################################


class DVAElatentspacePeriodic(DVAElatentspace):
    """
    A periodic latent space, S^n. Implemented using von Mises and uniform periodic distribution
    """

    def __init__(
            self,
            n_dim: int = 1
    ):
        super(DVAElatentspace, self).__init__(n_dim * 2, n_dim * 2)

    """
    Calculate the loss - KL distance
    """

    def get_loss(self):
        pass


######################################################################################################
######################################################################################################
######################################################################################################


class DVAElatentspaceLinear(DVAElatentspace):
    """
    A linear latent space, N^n - the regular kind
    """

    def __init__(
            self,
            n_dim: int = 1
    ):
        super(DVAElatentspace, self).__init__(n_dim * 2, n_dim * 2)

    """
    Calculate the loss - KL distance
    compare https://github.com/YosefLab/scvi-tools/blob/master/scvi/module/_vae.py#L343 
    """

    def get_loss(self):
        pass

        qz_m = inference_outputs["qz_m"]
        qz_v = inference_outputs["qz_v"]

        mean = torch.zeros_like(qz_m)
        scale = torch.ones_like(qz_v)

        kl_divergence_z = kl(Normal(qz_m, qz_v.sqrt()), Normal(mean, scale)).sum(dim=1)


######################################################################################################
######################################################################################################
######################################################################################################

class DVAElatentspaceSizefactor(DVAElatentspace):
    """
    This corresponds to the l-space in the SCVI model

    """

    def __init__(self, n_dim_in: int, n_dim_out: int):
        super().__init__(n_dim_in, n_dim_out)


# TODO this class likely needs a bit special treatment
# but scATAC+RNA might have two of these!


######################################################################################################
######################################################################################################
######################################################################################################


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
        super(DVAElatentspace, self).__init__(n_dim * 2, n_dim * 2)

    """
    Calculate the loss - KL distance
    """

    def get_loss(self):
# todo sum up losses
