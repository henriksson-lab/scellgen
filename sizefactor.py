import warnings
from typing import Tuple

import abc
import torch
import numpy as np
import anndata

import loss
import encoders


######################################################################################################
######################################################################################################
######################################################################################################


def calculate_library_size_priors(
        adata: anndata.AnnData,
        batch=None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return a list of library log mean/var for each cell. If batch is set the variances are calculated
    for each batch. Otherwise the mean/variance is for all the cells

    Returns observed log counts, mean log counts, variance log counts
    """
    data = adata.X

    sum_counts = data.sum(axis=1)
    library_log_obs = np.ma.log(sum_counts)

    if batch is None:
        library_log_mean = library_log_obs * 0 + np.mean(library_log_obs).astype(np.float32)
        library_log_var = library_log_obs * 0 + np.var(library_log_obs).astype(np.float32)
    else:
        # Calculate the variance for each batch and set the same variance for all the cells
        library_log_var = library_log_obs * 0
        library_log_mean = library_log_obs * 0
        for batch_name in np.unique(adata.obs[batch].tolist()):
            the_ind = adata.obs[batch] == batch_name
            batch_log_means = library_log_obs[the_ind]
            library_log_mean[the_ind] = np.mean(batch_log_means).astype(np.float32)
            library_log_var[the_ind] = np.var(batch_log_means).astype(np.float32)

    return library_log_obs, library_log_mean, library_log_var


######################################################################################################
######################################################################################################
######################################################################################################

class DVAEsizefactor(metaclass=abc.ABCMeta):
    """
    Abstract class: Implementation of size factors
    """

    @abc.abstractmethod
    def sample(
            self,
            x: torch.Tensor,
            sf_empirical_mean: torch.Tensor,
            sf_empirical_var: torch.Tensor,
            loss_recorder: loss.DVAEloss
    ):
        pass


######################################################################################################
######################################################################################################
######################################################################################################

class DVAEsizefactorFixed(DVAEsizefactor):
    """
    Size factors as simply the constant factors calculated on the cells
    """

    def encode(self,
               x: torch.Tensor,
               loss_recorder: loss.DVAEloss
               ):
        pass  # nothing to do

    def reparameterize(
            self,
            x: torch.Tensor,
            sf_empirical_mean: torch.Tensor,
            sf_empirical_var: torch.Tensor,
            loss_recorder: loss.DVAEloss
    ):
        pass

    def sample(
            self,
            x: torch.Tensor,
            sf_empirical_mean: torch.Tensor,
            sf_empirical_var: torch.Tensor,
            loss_recorder: loss.DVAEloss
    ):
        return sf_empirical_mean


######################################################################################################
######################################################################################################
######################################################################################################

class DVAEsizefactorLatentspace(DVAEsizefactor):
    """
    Size factors fitted using a latent space with a prior centered over the observed mean and variance.
    This corresponds to the l-space in the SCVI model
    """

    def __init__(
            self,
            n_input,
            n_covariates
    ):
        self.encoder = encoders.DVAEencoderFC(
            n_input=n_input,
            n_output=1,
            n_covariates=n_covariates
        )

    def encode(self,
               x: torch.Tensor,
               loss_recorder: loss.DVAEloss
               ):
        return self.layer.forward(x)

    def reparameterize(
            self,
            x: torch.Tensor,
            sf_empirical_mean: torch.Tensor,
            sf_empirical_var: torch.Tensor,
            loss_recorder: loss.DVAEloss
    ):
        # Split input vector into latent space parameters
        z_dim = self.n_dim_in
        z_mean, z_var = torch.split([z_dim, z_dim])

        # ensure positive variance. use exp instead?
        z_var = torch.nn.functional.softplus(self.fc_var(x))

        # The distributions to compare
        q_z = torch.distributions.normal.Normal(z_mean, z_var)
        p_z = torch.distributions.normal.Normal(sf_empirical_mean, sf_empirical_var)

        loss_recorder.add_kl(torch.distributions.kl.kl_divergence(q_z, p_z).sum(-1).mean())

        return q_z

    def sample(
            self,
            x: torch.Tensor,
            sf_empirical_mean: torch.Tensor,
            sf_empirical_var: torch.Tensor,
            loss_recorder: loss.DVAEloss
    ):
        q_z = self.reparameterize(
            x,
            sf_empirical_mean,
            sf_empirical_var,
            loss_recorder
        )
        return q_z.sample()

# check out _compute_local_library_params() in SCVI


# this is really the same as a normal dist space

# we could add support for using observed sizes
# SCVI uses  use_observed_lib_size   =True as default!!
