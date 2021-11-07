import scanpy as sc

import encoders
import decoders
import model
import latentspace
import training
import loader

adata = sc.read("foo.h5")


# ##################################################################################################################
# ######## Example: predict cell cycle impact ################################################################
# ##################################################################################################################


model = model.DVAEmodelAnndata(adata)

# rename to input
loader.DVAEloaderCounts(model)
loader.DVAEloaderObs(model, list_cat=["donor"])

# encoder layer
input_genes = adata.var.index[adata.var.is_cc]
encoders.DVAEencoderFC(model, inputs=[("X", input_genes),"batch"], output="enc_rna")

# latent space
latentspace.DVAElatentspaceSizeFactor(model, output="sf_rna")
latentspace.DVAElatentspaceLinear(model, inputs="enc_rna", output="z") # num dims detected by number of inputs

# decoder layer
output_genes = adata.var.index[adata.var.is_highly_variable]
decoders.DVAErnaseq(model, inputs="z", sf="sf_rna", dispersion="zinb")

### idea! can support non-V AEs as well. just detect if input is tensor or distribution

trainer = training.DVAEtrainingNormal()
trainer.train(model)
