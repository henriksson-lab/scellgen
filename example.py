import scanpy as sc
from anndata import AnnData


import datadecl
import latentspace
import training

adata = sc.read("foo.h5")
#assume highly variables genes set



# ######## Example: fit all RNAseq genes ################################################################


#




# ######## Example: predict cell cycle impact ################################################################

# Define latent space
zspace = latentspace.DVAElatentspaceLinear(n_dim=1)

# Define input and output genes
input_genes = adata.obs.index[adata.obs.is_cc]
output_genes = adata.obs.index[adata.obs.is_highly_variable]

decl = datadecl.DVAEdatadeclarationAnndata(
    adata,
    zspace)

decl.add_genes(input_genes, output_genes)
#this automatically adds the right loss function unless other given


trainer = training.DVAEtrainingNormal()
trainer.train(decl)


# ######## Example: plot coordinates ################################################################

# theta from x,y - how? or just get x,y, leave user the problem?
# this is a pandas df with named coords
# actually, this function can automatically calculate theta as well as give x,y
adata.obsm["zcoord"] = decl.get_latent_coordinates()

#calculate
adata.obsm["_PCA"] = adata.obsm["zcoord"]



# ######## Example: predict isoforms ################################################################



# ######## Example: predict ATAC from RNA ################################################################




# ######## Example: train on RNA and ATAC together ################################################################





