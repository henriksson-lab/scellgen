import unittest
import scanpy as sc

import encoders
import decoders
import core
import latentspace
import training
import loader

adata = sc.read("data/small_rna.h5ad")
device = training.get_torch_device()

class TestStringMethods(unittest.TestCase):

    # ##################################################################################################################
    # ######## Test: simplest use case, RNAseq #########################################################################
    # ##################################################################################################################

    def test1(self):
        m = core.DVAEmodel(adata)

        # rename to input
        loader.DVAEloaderCounts(m)

        # encoder layer
        encoders.DVAEencoderFC(m, inputs="X", output="enc_rna", n_output=10)

        # latent space
        latentspace.DVAElatentspaceSizeFactor(m, inputs="X_sf", output="sf_rna")
        latentspace.DVAElatentspaceLinear(m, inputs="enc_rna", output="z")

        # decoder layer
        decoders.DVAEdecoderRnaseq(m, inputs="z", sf="sf_rna", dispersion="zinb")

        trainer = training.DVAEtrainingBasic()
        trainer.train(m)


if __name__ == '__main__':
    unittest.main()
