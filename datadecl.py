from anndata import AnnData



class DVAEdatadeclaration():

    """
    This class is responsible for organizing what we want to do with the adata. What comes in and out?


    Container for all other classes?

    Can have specialized classes that sets up the standard stuff

    Technically, this class might not even need to operate over adatas... :o but maybe too general

    """




class DVAEdatadeclarationRNAseq(DVAEdatadeclaration):
    """

    Class that sets up everything needed to model RNAseq data the usual way.
    equivalent to the SCVI model

    """


    def __init__(self,
                 adata: AnnData
                 ):
        666
