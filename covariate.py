import anndata
import torch
from torch.nn.functional import one_hot
from typing import List

def categories_tensor(values):
    '''

    changes categorical values (strings) to integer valuesdict
    required for torch.nn.functional.one_hot
    which in turn converts it to Tensor form

    '''
    unique_classes = list(set(values))
    valuesdict = {key: value for key, value in enumerate(unique_classes)}
    classvalues = [valuesdict[v] for v in values]
    classtensor = one_hot(torch.arange(0,len(classvalues)), num_classes=len(unique_classes))
    return classtensor

class DVAEcovariate():

    def __init__(
        adata,
        list_cat: List[str],
        list_cont: List[str]
    ):

        self.adata = adata
        self.list_cat = list_cat # list of obs columns to use as discrete covariates
        self.list_cont = list_cont # list of obs columns to use as continuous covariates

    def forward(self):

        for i, category in enumerate(self.list_cat):

            current_tensor = torch.Tensor(self.adata.obs[category].values)

            if i == 0:
                total_categorical_tensor = current_tensor
            else:
                total_categorical_tensor = torch.cat(total_categorical_tensor, current_tensor)

        for i, category in enumerate(self.list_cont):

            current_tensor = torch.Tensor(np.array(adata.obs[category].values))

            if i == 0:
                total_cont_tensor = current_tensor
            else:
                total_cont_tensor = torch.cat(total_cont_tensor, current_tensor)

        return torch.cat(total_categorical_tensor, total_cont_tensor)



    def get_num_covariate(self):
        pass


    def encode_covariate(self, obs_for_the_sample) -> torch.Tensor:
        """
        used once, when creating data loader. NOT in encoder/decoder
        :return:
        """
        pass #return encoded, for the entire adata





import torch

## total rewrite
def one_hot(index: torch.Tensor, n_cat: int) -> torch.Tensor:
    """One hot a tensor of categories."""
    onehot = torch.zeros(index.size(0), n_cat, device=index.device)
    onehot.scatter_(1, index.type(torch.long), 1)
    return onehot.type(torch.float32)
