import scanpy as sc

import encoders
import decoders
import core
import latentspace
import training
import loader

adata = sc.read("data/small_rna.h5ad")

device = training.get_torch_device()


# ##################################################################################################################
# ######## Test: simplest use case, RNAseq #########################################################################
# ##################################################################################################################


m = core.DVAEmodel(adata)

# rename to input
loader.DVAEloaderCounts(m)

# encoder layer
encoders.DVAEencoderFC(m, inputs="X", output="enc_rna", n_output=10)

# latent space
latentspace.DVAElatentspaceSizeFactor(m, output="sf_rna")
latentspace.DVAElatentspaceLinear(m, inputs="enc_rna", output="z")

# decoder layer
output_genes = adata.var.index[adata.var.is_highly_variable]
decoders.DVAErnaseq(m, inputs="z", sf="sf_rna", dispersion="zinb")

trainer = training.DVAEtrainingBasic()
trainer.train(m)
