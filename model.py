from anndata import AnnData

from typing import List, Optional
import abc

import covariate
import latentspace
import anndata

######################################################################################################
######################################################################################################
######################################################################################################



class DVAEmodel(metaclass=abc.ABCMeta):
    """
    This class contains everything else.
    Technically, this class might not even need to operate over adatas... :o but maybe too general
    """

    def get_latent_representation(self):
        pass



######################################################################################################
######################################################################################################
######################################################################################################


class DVAEmodelAnndata(DVAEmodel):
    """
    Class that sets up everything needed to model data the usual way.
    equivalent to the SCVI model but later we can extend it
    """

    def __init__(
            self,
            adata: anndata.AnnData,
            latentspace: latentspace.DVAElatentspace,
            covariates: covariate.DVAEcovariate = None
    ):
        self.adata = adata
        self.latentspace = latentspace

        # todo how do we compose the data loaders the best way that fits more complex scenarios?



    def add_genes(
            self,
            input_genes: List[str] = [],
            output_genes: List[str] = None,
            latent_space_mapping: List[int]   #todo figure this out

            # todo optional error model

    ):
        """
        Add genes to predict, with suitable error function
        """
        if output_genes is None:
            output_genes = input_genes

        # todo add size factor latent space
        lspace = latentspace.DVAElatentspaceSizefactor()



    def add_covariates(
            self,
            list_cat: List[str] = [],
            list_cont: List[str] = [],
            obs_variable: str = "obs"
    ):
        """
        Add covariates to use for prediction. These do not contribute to the error function
        """
        self.cov = covariate.DVAEobsmapper(
            self.adata[obs_variable],
            list_cat,
            list_cont
        )



    def add_isoforms(
            self,
            map_gene_isoform,  # todo format? dict? table?
            input_genes: List[str] = [],
            output_genes: List[str] = None,
    ):
        """
        Add isoforms to predict, with suitable error function
        """
        if output_genes is None:
            output_genes = input_genes

        # todo isoform-gene mapping

    def get_normalized_expression(
            self,
            library_size: float = 10e4):
        pass
