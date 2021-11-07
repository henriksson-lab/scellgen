import scanpy as sc

import encoders
import decoders
import model
import latentspace
import training
import loader

adata = sc.read("foo.h5")


device = training.get_torch_device()


# ##################################################################################################################
# ######## Test: simplest use case, RNAseq #########################################################################
# ##################################################################################################################


model = model.DVAEmodelAnndata(adata)

# rename to input
loader.DVAEloaderCounts(model)

# encoder layer
encoders.DVAEencoderFC(model, inputs="X", output="enc_rna", n_output=10)

# latent space
latentspace.DVAElatentspaceSizeFactor(model, output="sf_rna")
latentspace.DVAElatentspaceLinear(model, inputs="enc_rna", output="z")

# decoder layer
output_genes = adata.var.index[adata.var.is_highly_variable]
decoders.DVAErnaseq(model, inputs="z", sf="sf_rna", dispersion="zinb")

trainer = training.DVAEtrainingBasic()
trainer.train(model)
