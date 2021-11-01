from anndata import AnnData

from typing import List, Optional


class DVAEdatadeclaration():

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


    def __init__(self,
         adata: AnnData,
         genes: List[str] = [],
         peaks: List[str] = [],
                 # todo isoform-gene mapping
         batch_variables: List[str] = []
     ):
        666

        # todo how do we compose the data loaders the best way that fits more complex scenarios?


        # todo calculate priors for the l-space? mean and variance. store where for later?
        if len(genes)!=0:
            666
            # todo create gene l-space
        if len(peaks)!=0:
            666







