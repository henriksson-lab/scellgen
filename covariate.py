from sklearn import preprocessing
from typing import List

import pandas as pd
import numpy as np
import torch

class DVAEobsmapper():

    def __init__(
            self,
            obs_df: pd.DataFrame,
            list_cat: List[str] = [],
            list_cont: List[str] = []
    ):
        """
        Mapper from table data to encodings suited for a neural network
        """
        self.list_cat = list_cat
        self.list_cont = list_cont
        self._value_mapping = dict()

        # For each discrete category, produce a mapping to [0,1].
        # Currently this is done using one-hot encoding
        self._num_dim_out = len(self.list_cont)
        for one_cat in list_cat:
            lb = preprocessing.LabelBinarizer()
            lb.fit(obs_df[one_cat])
            self._value_mapping[one_cat] = lb
            self._num_dim_out += len(lb.transform(lb.classes_[0])[0])

        # For each continuous category, produce a mapping to mean 0, variance 1
        for one_cat in list_cont:
            lb = preprocessing.StandardScaler()
            lb.fit(obs_df[one_cat])
            self._value_mapping[one_cat] = lb

    def get_dim_encodings(self):
        """
        Calculate the number of outputs after encoding
        """
        return self._num_dim_out

    def encode_obs(
            self,
            obs_df,
            device: torch.device,
            dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        """
        Encode values
        """
        list_tx = []
        for key, mapping in enumerate(self._value_mapping):
            x = np.array(mapping.transform(obs_df[key]))
            tx = torch.from_numpy(x, device=device, dtype=dtype)
            list_tx.append(tx)

        return torch.cat(list_tx, dim=1)


class DVAEcovariate(DVAEobsmapper):

    def __init__(
            self,
            adata: anndata.AnnData,
            list_cat: List[str] = [],
            list_cont: List[str] = []
    ):
        super().__init__(
            adata[obs_variable],
            list_cat,
            list_cont
        )
        """
        This class keeps track of where the table came from, unlike the superclass which is not limited to Adata
        """
        self._obs_variable = obs_variable


def empty_covariate():
    """
    Return an empty covariate
    """
    return DVAEcovariate([], [])
