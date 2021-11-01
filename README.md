# dvae
Directional variational auto-encoder


# TODO
Generative model that uses the same pipeline (but with our in-house code) to produce expression values from latent space as in (https://www.nature.com/articles/s41592-018-0229-2) replacing Gaussian with hyperspherical/ von Mises Fisher distribution) https://github.com/nicola-decao/s-vae-pytorch

The Inference (Encoder) implementation stays the same but we write our own utilities since we need to avoid possible 
license issues with scvi
