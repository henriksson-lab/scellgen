from anndata import AnnData

from typing import List, Optional
import abc

import latentspace


class DVAEdatadeclaration(metaclass=abc.ABCMeta):
    """
    This class is responsible for organizing what we want to do with the adata. What comes in and out?


    Container for all other classes?

    Can have specialized classes that sets up the standard stuff

    Technically, this class might not even need to operate over adatas... :o but maybe too general

    """


######################################################################################################
######################################################################################################
######################################################################################################


class DVAEdatadeclarationAnndata(DVAEdatadeclaration):
    """

    Class that sets up everything needed to model data the usual way.
    equivalent to the SCVI model but later we can extend it

    """

    def __init__(
            self,
            adata: AnnData,
            latentspace: latentspace.DVAElatentspace
    ):
        self.adata = adata

        # todo how do we compose the data loaders the best way that fits more complex scenarios?

    def add_genes(
            self,
            input_genes: List[str] = [],
            output_genes: List[str] = None
    ):
        if output_genes is None:
            output_genes = input_genes

        # todo add size factor latent space
        lspace = latentspace.DVAElatentspaceSizefactor()



    def add_batch_variables(
            self,
            batch_variables: List[str]
    ):

        666
        # todo

    # todo isoform-gene mapping

    """
    Add isoforms to predict, with suitable error function
    """

    def add_isoforms(
            self,
            map_gene_isoform,  # todo format? dict? table?
            input_genes: List[str] = [],
            output_genes: List[str] = None,
    ):
        if output_genes is None:
            output_genes = input_genes

        # todo
