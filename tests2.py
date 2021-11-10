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
        encoders.DVAEencoderFC(m, inputs="X", output="enc_rna", n_output=4)

        # latent space
        latentspace.DVAElatentspaceLinear(m, inputs="enc_rna", output="z")

        # decoder layer
        decoders.DVAEdecoderLinear(m, inputs="z")

        m.env.print_variable_defs()

        trainer = training.DVAEtrainingBasic()
        trainer.train(m)

        m.env.call_graph()

if __name__ == '__main__':
    unittest.main()
