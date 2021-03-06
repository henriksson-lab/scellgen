import unittest
import scanpy as sc

import encoders
import decoders
import core
import latentspace
import training
import loader

# later test joshua data at: /corgi/joshuac/FAP_CAF_6S_10x/HomSa.FAP.CAFs.Files.To.Play.With

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
        encoders.DVAEencoderFC(m, inputs="X", output="enc_sf", n_output=2)

        # latent space
        latentspace.DVAElatentspaceSizeFactor(m, inputs="enc_sf", sf_empirical="X_sf", output="sf_rna")
        latentspace.DVAElatentspaceLinear(m, inputs="enc_rna", output="z")

        # decoder layer
        decoders.DVAEdecoderRnaseq(
            m,
            inputs="z",
            input_sf="sf_rna",
            gene_likelihood="poisson"  # "zinb"
        )

        m.env.print_variable_defs()

        trainer = training.DVAEtrainingBasic(lr=1e-3)
        trainer.train(m)

        m.env.call_graph()


if __name__ == '__main__':
    unittest.main()
