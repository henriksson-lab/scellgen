import warnings
from typing import Tuple

def stats_library_size(
    adata: anndata.AnnData, batch = "batch"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes and returns library size.
    Parameters
    ----------
    adata
        AnnData object.
    n_batch
        Number of batches.
    Returns
    -------
    type
        Tuple of two 1 x n_batch ``np.ndarray`` containing the means and variances
        of library size in each batch in adata.
        If a certain batch is not present in the adata, the mean defaults to 0,
        and the variance defaults to 1. These defaults are arbitrary placeholders which
        should not be used in any downstream computation.
    """
    data = adata.X
    
    batch_indices = [v for i, v in enumerate(adata.obs[batch].tolist())]
    batch_indices = pd.DataFrame({'batch_indices': batch_indices})
    
    n_batch = len(np.unique(batch_indices))

    library_log_means = np.zeros(n_batch)
    library_log_vars = np.ones(n_batch)

    for i, i_batch in enumerate(np.unique(batch_indices)):
        idx_batch = np.squeeze(batch_indices[batch_indices['batch_indices'] == i_batch].copy().index.tolist())
        
        batch_data = data[
            idx_batch.nonzero()[0]
        ]  # h5ad requires integer indexing arrays.
        sum_counts = batch_data.sum(axis=1)
        masked_log_sum = np.ma.log(sum_counts)
        if np.ma.is_masked(masked_log_sum):
            warnings.warn(
                "This dataset has some empty cells, this might fail inference."
                "Data should be filtered with `scanpy.pp.filter_cells()`"
            )

        log_counts = masked_log_sum.filled(0)
        library_log_means[i] = np.mean(log_counts).astype(np.float32)
        library_log_vars[i] = np.var(log_counts).astype(np.float32)

    return library_log_means.reshape(1, -1), library_log_vars.reshape(1, -1)
