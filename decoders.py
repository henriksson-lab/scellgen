from typing import List, Optional, Literal

import abc
import torch
import torch.nn as nn
import covariate
import loss

from torch.distributions import Normal, Poisson, Distribution
from scvi.distributions import ZeroInflatedNegativeBinomial


######################################################################################################
######################################################################################################
######################################################################################################

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

    @abc.abstractmethod
    def forward_from_point(
            self,
            loss_recorder: loss.DVAEloss,
            z: torch.Tensor,
            cov: torch.Tensor
    ):
        """
        Generate samples given the latent space coordinates
        """
        pass

    def forward_from_distribution(
            self,
            loss_recorder: loss.DVAEloss,
            list_z: List[Distribution],
            cov: torch.Tensor
    ):
        """
        Generate samples given the latent space distribution
        """
        z_samples = [z.sample() for z in list_z]
        # todo what about multiple samples per point?

        return self.forward_from_point(loss_recorder, z_samples, cov)


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
        super().__init__(n_input, n_output)
        # todo set up the Linear pytorch layers.

        # todo consider making a class like FClayer, but simpler. or reuse internally.


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
            n_output: int,
            dispersion: Literal["gene", "gene-batch", "gene-label", "gene-cell"] = "gene",
            gene_likelihood: Literal["zinb", "nb", "poisson"] = "zinb",
    ):
        """
        Generative function for RNAseq data according to the SCVI model
        """
        super().__init__(n_input, n_output)

        self.dispersion = dispersion
        self.gene_likelihood = gene_likelihood

        if dispersion not in ["gene", "gene-batch", "gene-label", "gene-cell"]:
            raise ValueError(
                "dispersion must be one of ['gene', 'gene-batch', 'gene-label', 'gene-cell'], but input was {}".
                format(self.dispersion)
            )

        # mean gamma
        self.px_scale_decoder = nn.Sequential(
            nn.Linear(n_hidden, n_output),
            nn.Softmax(dim=-1),
        )

        # dispersion. no covariates handled yet
        self.px_r_decoder = nn.Linear(n_hidden, n_output)

        # dropout
        self.px_dropout_decoder = nn.Linear(n_hidden, n_output)

    def forward_from_point(
            self,
            loss_recorder: loss.DVAEloss,
            z: torch.Tensor,
            cov: torch.Tensor,
    ):
        """
        The decoder returns values for the parameters of the RNAseq count distribution
        """

        # for now, assume library scaling just come from somewhere...
        z_l = z[0]
        library = z_l.sample()
        # add support for using observed library size




        # todo Need the library scaling here!


        ############################ https://github.com/YosefLab/scvi-tools/blob/9855238ae13543aefd212716f4731446bb2922bb/scvi/nn/_base_components.py

        rho = self.rho_decoder(torch.cat(z, cov))  #px is rho in https://www.nature.com/articles/s41592-018-0229-2
        px_scale = self.px_scale_decoder(rho)
        px_rate = torch.exp(library) * px_scale
        px_dropout = self.px_dropout_decoder(rho)

        ############################
        # self.dispersion is a messy parameter. we could instead have a separate
        # cov variable that is fed into this neural network. then we would support
        # all cases in one mechanism
        px_r = torch.exp(self.px_r_decoder(rho))


        # Return the fitted distribution of gene counts
        if self.gene_likelihood == "zinb":
            return ZeroInflatedNegativeBinomial(mu=px_rate, theta=px_r, zi_logits=px_dropout)
        elif self.gene_likelihood == "nb":
            return NegativeBinomial(mu=px_rate, theta=px_r)
        elif self.gene_likelihood == "poisson":
            return Poisson(px_rate)
        else:
            raise "Unsupported gene likelihood {}".format(self.gene_likelihood)





# Decoder
class DecoderSCVI(nn.Module):
    """
    Decodes data from latent space of ``n_input`` dimensions into ``n_output``dimensions.
    Uses a fully-connected neural network of ``n_hidden`` layers.
    Parameters
    ----------
    n_input
        The dimensionality of the input (latent space)
    n_output
        The dimensionality of the output (data space)
    n_cat_list
        A list containing the number of categories
        for each category of interest. Each category will be
        included using a one-hot encoding
    n_layers
        The number of fully-connected hidden layers
    n_hidden
        The number of nodes per hidden layer
    dropout_rate
        Dropout rate to apply to each of the hidden layers
    inject_covariates
        Whether to inject covariates in each layer, or just the first (default).
    use_batch_norm
        Whether to use batch norm in layers
    use_layer_norm
        Whether to use layer norm in layers
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
        inject_covariates: bool = True,
        use_batch_norm: bool = False,
        use_layer_norm: bool = False,
    ):
        super().__init__()
        self.px_decoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=0,
            inject_covariates=inject_covariates,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
        )

        # mean gamma
        self.px_scale_decoder = nn.Sequential(
            nn.Linear(n_hidden, n_output),
            nn.Softmax(dim=-1),
        )

        # dispersion: here we only deal with gene-cell dispersion case
        self.px_r_decoder = nn.Linear(n_hidden, n_output)

        # dropout
        self.px_dropout_decoder = nn.Linear(n_hidden, n_output)

    def forward(
        self, dispersion: str, z: torch.Tensor, library: torch.Tensor, *cat_list: int
    ):
        """
        The forward computation for a single sample.
         #. Decodes the data from the latent space using the decoder network
         #. Returns parameters for the ZINB distribution of expression
         #. If ``dispersion != 'gene-cell'`` then value for that param will be ``None``
        Parameters
        ----------
        dispersion
            One of the following
            * ``'gene'`` - dispersion parameter of NB is constant per gene across cells
            * ``'gene-batch'`` - dispersion can differ between different batches
            * ``'gene-label'`` - dispersion can differ between different labels
            * ``'gene-cell'`` - dispersion can differ for every gene in every cell
        z :
            tensor with shape ``(n_input,)``
        library
            library size
        cat_list
            list of category membership(s) for this sample
        Returns
        -------
        4-tuple of :py:class:`torch.Tensor`
            parameters for the ZINB distribution of expression
        """
        # The decoder returns values for the parameters of the ZINB distribution
        px = self.px_decoder(z, *cat_list)
        px_scale = self.px_scale_decoder(px)
        px_dropout = self.px_dropout_decoder(px)
        px_rate = torch.exp(library) * px_scale
        px_r = self.px_r_decoder(px) if dispersion == "gene-cell" else None
        return px_scale, px_r, px_rate, px_dropout