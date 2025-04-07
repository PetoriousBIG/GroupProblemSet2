# Tasks

## Decisions to make
- Which dataset?
- Are we doing Task 4?

## Preprocessing Requirements
### The following tasks can be completed using previously completed work from the final project. __Due 4/7__
- [] Upload dataset to repo
- [] Represent abstracts as a vector (one or more of Bag of Words, TF-IDF, Word2Vec, CloVe, BERT, etc)

## Tasks
### Task 1 __Due 4/10__
- [] Reduce dimensionality of abstract vectors using either Autoencoder or Restricted Boltzmann Machine
- [] Visualize latent space using PCA, T-SNE, or UMAP

### Task 2 __Due 4/10__
- [] Apply Gaussian Mixture Model or Restricted Boltzmann Machine to vector representations
    - [] Cluster abstract into k groups
    - [] Same the abstracts from each cluster
    - [] Interpret clusters: what groups each cluster together

### Task 3 (If Using Autoencoder) __Due 4/13__
- [] Reconstruct the abstract vectors and measure information loss
- [] Identify common traits of reconstructions
- [] Identify poor reconstructions

### Task 4 (Optional, If using VAE) __DUE ???__
- [] Sample new latent vectors and decode them
- [] Interpolate between two abstracts and analyze how content evolves

### Task 5 (Deliverables) __DUE 4/16__

- [] Write report, (5 pages max)
    - Dataset and embedding methods
    - Description of models used
    - Latent Space Visuals
    - Cluster interpretations
    - Option reconstruction/generation insights
- [] Google colab or Jupyter notebook
- [] 10-15 minute presentation on workflow and results

#### Focus on the following in the deliverables
- [] How did the model help uncover the latent structure?
- [] What suprised us about the clusters or reconstructions?
- [] How could this be extended to new research areas or languages?


