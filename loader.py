from typing import List, Optional

import torch
from sklearn import preprocessing
import pandas as pd
import numpy as np

import model

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


class DVAEloaderCounts(model.DVAEloader):

    def __init__(
            self,
            mod: model.DVAEmodel,
            output: str = "X",
            adata_varname: str = "X",

            # Set if size factors should be computed
            sf_output: str = "X_sf",
            sf_batch_variable: str = None
    ):
        """
        Loads data from adata["X"]

        Empirical size factors will be computed if an output variable is specified, with batch-specific
        factors if an obs-column denoting batch is provided
        """
        super().__init__(mod)
        self._sf_batch_variable = sf_batch_variable
        self._sf_output = sf_output
        self._varname = adata_varname
        self._output = output

        mod.env.define_variable(output, mod.adata[adata_varname].shape[0])  # todo can be totally wrong

        if sf_output is not None:
            mod.env.define_variable(sf_output, 3)


    def get_dataset(self) -> torch.Dataset:
        """
        Get the count matrix as a Torch Dataset
        """

        if self._sf_output is not None:
            library_log_obs, library_log_mean, library_log_var = calculate_library_size_priors(
                self.model,
                self._sf_batch_variable)

            df = pd.DataFrame({
                "obs":library_log_obs,
                "mean":library_log_mean,
                "var":library_log_var
            })


        pass


######################################################################################################
######################################################################################################
######################################################################################################

class DVAEloaderObs(model.DVAEloader):

    def __init__(
            self,
            mod: model.DVAEmodel,
            list_cat: List[str] = [],
            list_cont: List[str] = [],
            output: str = "obs",
            adata_varname: str = "obs"
    ):
        """
        Loads data from the adata["obs"], including preprocessing as needed.
        This includes encoding of the categorial data especially
        """
        super().__init__(mod)
        self.list_cat = list_cat
        self.list_cont = list_cont
        self._varname = adata_varname
        self._output = output

        # For each continuous category, produce a mapping to mean 0, variance 1
        self._num_dim_out = len(self.list_cont)
        for one_cat in list_cont:
            lb = preprocessing.StandardScaler()
            lb.fit(obs_df[one_cat])
            self._value_mapping[one_cat] = lb

        # For each discrete category, produce a mapping to [0,1].
        # Currently this is done using one-hot encoding
        for one_cat in list_cat:
            lb = preprocessing.LabelBinarizer()
            lb.fit(obs_df[one_cat])
            self._value_mapping[one_cat] = lb
            self._num_dim_out += len(lb.transform(lb.classes_[0])[0])

        mod.env.define_variable(output, self._num_dim_out)




    def get_dataset(
            self,
            device: torch.device,
            dtype: torch.dtype = torch.float32
    ) -> torch.Dataset:
        """
        Get the obs dataframe as a Torch Dataset
        """
        obs_df = self.model.adata[self._varname]

        list_tx = []
        for key, mapping in enumerate(self._value_mapping):
            x = np.array(mapping.transform(obs_df[key]))
            tx = torch.from_numpy(x, device=device, dtype=dtype)
            list_tx.append(tx)

        a_tensor = torch.cat(list_tx, dim=1)


        # todo turn into a Dataset

