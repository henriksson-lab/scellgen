

class DVAEcovariate():

    __init__(
        adata,
        list_cat: List[str],
        list_cont: List[str]
    )



    adata.obs # make one hot

    self.onehot= ...  #pandas?



    def get_num_covariate(self):
        ...


    def encode_covariate(self, obs_for_the_sample) -> torch.Tensor:
        """
        used once, when creating data loader. NOT in encoder/decoder
        :return:
        """
        ... #return encoded, for the entire adata





import torch

## total rewrite
def one_hot(index: torch.Tensor, n_cat: int) -> torch.Tensor:
    """One hot a tensor of categories."""
    onehot = torch.zeros(index.size(0), n_cat, device=index.device)
    onehot.scatter_(1, index.type(torch.long), 1)
    return onehot.type(torch.float32)