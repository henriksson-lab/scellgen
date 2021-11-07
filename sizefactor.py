from typing import Tuple

import numpy as np
import anndata


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



