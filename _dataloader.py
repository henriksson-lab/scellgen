from typing import Optional, Dict, List, Union

import anndata
from anndata._core.sparse_dataset import SparseDataset
import torch
from torch.utils.data import DataLoader, Dataset
import h5py
import numpy as np
import pandas as pd
from math import ceil
import copy
import scipy


# aim to replace https://docs.scvi-tools.org/en/0.8.0/api/reference/scvi.data.setup_anndata.html


class AnnTorchDataset(Dataset):

    def __init__(
            self,
            num_items,
            np_dtype,  # : Union[np.float32, np.int64],
            data: Union[anndata.AnnData, pd.DataFrame, h5py.Dataset, SparseDataset, scipy.sparse.csr.csr_matrix]
    ):
        """
        Dataset capable of using AnnData objects as input; but also Pandas dataframes and Numpy matrices
        https://pytorch.org/docs/stable/data.html#map-style-datasets

        Largely taken from SCVI; see their license file
        """
        self.np_dtype = np_dtype
        self.data = data
        self._num_items = num_items

        # or scipy.sparse.csr.csr_matrix
        #if isinstance(data, h5py.Dataset) or \
        #        isinstance(data, SparseDataset) or \
        #        isinstance(data, pd.DataFrame) or \
        #        isinstance(data, np.ndarray):  # todo may need to check dims of ndarray
        #    # ok!
        #    pass
        #else:
        #    raise Exception("not implemented for type {}".format(type(data)))

    def __getitem__(
            self,
            idx: List[int]
    ) -> np.ndarray:
        """
        Get observations at idx.
        """
        data = self.data
        if isinstance(data, h5py.Dataset) or isinstance(data, SparseDataset):
            # for backed anndata
            # need to sort idxs for h5py datasets
            if hasattr(idx, "shape"):
                argsort = np.argsort(idx)
            else:
                argsort = idx
            data = data[idx[argsort]]
            # now unsort
            i = np.empty_like(argsort)
            i[argsort] = np.arange(argsort.size)
            # this unsorts it
            idx = i

        # Consider the various types. The code above turns h5py and SparseDataset into below
        if isinstance(data, np.ndarray):
            data = data[idx].astype(self.np_dtype)
        elif isinstance(data, pd.DataFrame):
            data = data.iloc[idx, :].to_numpy().astype(self.np_dtype)
        else:
            data = data[idx].toarray().astype(self.np_dtype)
        return data

    def __len__(self):
        """
        Return the number of observations
        """
        return self._num_items


class BatchSampler(torch.utils.data.sampler.Sampler):

    def __init__(
            self,
            indices: np.ndarray,
            batch_size: int,
            shuffle: bool,
            drop_last: Union[bool, int] = False,
    ):
        """
        Custom torch Sampler that returns a list of indices of size batch_size.

        Taken from SCVI; see their license file

        Parameters
        ----------
        indices
            list of indices to sample from
        batch_size
            batch size of each iteration
        shuffle
            if ``True``, shuffles indices before sampling
        drop_last
            if int, drops the last batch if its length is less than drop_last.
            if drop_last == True, drops last non-full batch.
            if drop_last == False, iterate over all batches.
        """
        self.indices = indices
        self.n_obs = len(indices)
        self.batch_size = batch_size
        self.shuffle = shuffle

        if drop_last > batch_size:
            raise ValueError(
                "drop_last can't be greater than batch_size. "
                + "drop_last is {} but batch_size is {}.".format(drop_last, batch_size)
            )

        last_batch_len = self.n_obs % self.batch_size
        if (drop_last is True) or (last_batch_len < drop_last):
            drop_last_n = last_batch_len
        elif (drop_last is False) or (last_batch_len >= drop_last):
            drop_last_n = 0
        else:
            raise ValueError("Invalid input for drop_last param. Must be bool or int.")

        self.drop_last_n = drop_last_n

    def __iter__(self):
        if self.shuffle is True:
            idx = torch.randperm(self.n_obs).tolist()
        else:
            idx = torch.arange(self.n_obs).tolist()

        if self.drop_last_n != 0:
            idx = idx[: -self.drop_last_n]

        data_iter = iter(
            [
                self.indices[idx[i: i + self.batch_size]]
                for i in range(0, len(idx), self.batch_size)
            ]
        )
        return data_iter

    def __len__(self):
        if self.drop_last_n != 0:
            length = self.n_obs // self.batch_size
        else:
            length = ceil(self.n_obs / self.batch_size)
        return length


class BatchSamplerLoader(DataLoader):

    def __init__(
            self,
            dataset: Dataset,
            shuffle=False,
            indices=None,
            batch_size=128,
            drop_last: Union[bool, int] = False,
            **data_loader_kwargs,
    ):
        """
        DataLoader for loading tensors from AnnData objects.

        Modified from SCVI; see their license file

        Parameters
        ----------
        adata
            An anndata objects
        shuffle
            Whether the data should be shuffled
        indices
            The indices of the observations in the adata to load
        batch_size
            minibatch size to load each iteration
        data_and_attributes
            Dictionary with keys representing keys in data registry (`adata.uns["_scvi"]`)
            and value equal to desired numpy loading type (later made into torch tensor).
            If `None`, defaults to all registered data.
        data_loader_kwargs
            Keyword arguments for :class:`~torch.utils.data.DataLoader`
        """

        sampler_kwargs = {
            "batch_size": batch_size,
            "shuffle": shuffle,
            "drop_last": drop_last,
        }

        if indices is None:
            # Use all indices if not provided
            indices = np.arange(len(dataset))
        else:
            # If boolean list given, turn to absolute indices
            if hasattr(indices, "dtype") and indices.dtype is np.dtype("bool"):
                indices = np.where(indices)[0].ravel()
            indices = np.asarray(indices)
        sampler_kwargs["indices"] = indices

        self.indices = indices
        self.sampler_kwargs = sampler_kwargs
        sampler = BatchSampler(**self.sampler_kwargs)
        self.data_loader_kwargs = copy.copy(data_loader_kwargs)
        # do not touch batch size here, sampler gives batched indices
        self.data_loader_kwargs.update({"sampler": sampler, "batch_size": None})

        super().__init__(dataset, **self.data_loader_kwargs)


######################################################################################################
######################################################################################################
######################################################################################################

class ConcatListDataset(torch.utils.data.Dataset):
    def __init__(self, datasets: List[torch.utils.data.Dataset]):
        self.datasets = datasets

    def __getitem__(self, i):
        return [d[i] for d in self.datasets]

    def __len__(self):
        return min(len(d) for d in self.datasets)


class ConcatDictDataset(torch.utils.data.Dataset):
    def __init__(self, datasets: Dict[str, torch.utils.data.Dataset]):
        self.datasets = datasets

    def __getitem__(self, i):
        return dict([(key, d[i]) for (key, d) in self.datasets.items()])

    def __len__(self):
        return min(len(d) for d in self.datasets.values())

# Tensors are Dataset instances already. They are created this way
# torch.from_numpy(datamat_withz_zmean.detach().cpu().numpy()).to(device)
