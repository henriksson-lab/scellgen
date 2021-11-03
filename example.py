import scanpy as sc
from anndata import AnnData


import model
import latentspace
import training

adata = sc.read("foo.h5")
#assume highly variables genes set



# ##################################################################################################################
# ######## Example: fit all RNAseq genes ################################################################
# ##################################################################################################################





# ##################################################################################################################
# ######## Example: predict cell cycle impact ################################################################
# ##################################################################################################################

# Define latent space
zspace = latentspace.DVAElatentspaceLinear(n_dim=1)

# Define input and output genes
input_genes = adata.obs.index[adata.obs.is_cc]
output_genes = adata.obs.index[adata.obs.is_highly_variable]

model = model.DVAEmodelAnndata(
    adata,
    zspace)

model.add_genes(input_genes, output_genes)
#this should automatically add the right loss function unless other given


trainer = training.DVAEtrainingNormal()
trainer.train(model)

# ##################################################################################################################
# ######## Example: plot coordinates. case of theta ################################################################
# ##################################################################################################################

# theta from x,y - how? or just get x,y, leave user the problem?
# this is a pandas df with named coords
# actually, this function can automatically calculate theta as well as give x,y
adata.obsm["X_pca"] = model.get_latent_representation()


sc.pl.pca(adata, components = ['theta1,theta2'], ncols=2)  # Some tweaks needed
# optionally, write the umap coord. we may need an extended plotting function. scanpy may have a hidden
# function for umap and pca together


# ##################################################################################################################
# ######## Example: plot coordinates. case of N^m ################################################################
# ##################################################################################################################

adata.obsm["X_VAE"] = model.get_latent_representation()
sc.pp.neighbors(adata, use_rep="X_VAE")
sc.tl.umap(adata, min_dist=0.3)
sc.pl.umap(adata)




# ##################################################################################################################
# ######## Example: impute data ################################################################
# ##################################################################################################################

adata.layers["scvi_normalized"] = model.get_normalized_expression(library_size=10e4)



# ##################################################################################################################
# ######## Example: predict isoforms ################################################################
# ##################################################################################################################



# ##################################################################################################################
# ######## Example: predict ATAC from RNA ################################################################
# ##################################################################################################################




# ##################################################################################################################
# ######## Example: train on RNA and ATAC together ################################################################
# ##################################################################################################################





# ##################################################################################################################
# ######## Example: differential expression ################################################################
# ##################################################################################################################



#caveat: what if we have atac-seq data? maybe specialize this function a bit for RNAseq data
de_df = model.differential_expression(
    groupby="cell_type",
    group1="Endothelial",
    group2="Fibroblast"
)

de_df = model.differential_atac(
    groupby="cell_type",
    group1="Endothelial",
    group2="Fibroblast"
)

de_df = model.differential_isoforms(
    groupby="cell_type",
    group1="Endothelial",
    group2="Fibroblast"
)
