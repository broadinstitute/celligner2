# Notes

## Objectives

### run scARCHES on CCLE + TCGA (use the batch annotation)

### run scARCHES on CCLE + TCGA + CCLF + additional


### purify tcga rnaseq

did someone already do it? NO
make cibersortX's copy work
make our own cibersort version 

### semi supervision like mfMAP (need good annotations)

- sample specific information (will need ashir's new annotations, also annotations for each dataset used)
- label smoothing on semi-supervision (weak supervion)

### do data augmentation thing: (see list in other document)

- Use replicates when available.
- Up to some random noise, based on how high is the expression (randomly add)
- mix cell lines and normals to create fake impurity to correct
  - do: ccle_line + impurity * gtex_tissue
  - put semi supervision annotation as impurity: impurity, origin: gtex_tissue

- add more datasets (list in asana)
- add pseudo bulk from scRNAseq  (list in asana)

- add mice data (once in graph)

### add Ashir's annotation to the data and make it cross modal (add more QC)

### make it graph like and deeper, and add edge prediction

- use dgl's vgae
- skip connections
- add graph embedding learning and edge prediction
  - drop links from model and train it to classify missing links (positive, negative)

### what mfmap is not doing

- no purity estimation?
- not robust to outliers / not able to detect undif cluster
- Maybe it should use gene network from cmap data.

## what we would want to do

min(A - (X_a*Y_a + I_a))
min(B - (X_b*Y_b + I_b)) s.t. min(MIN(dist(X_a, X_b))) ; max(Y_a\*Y_b)

## How to test

- visually
- confounding matrix (cell type distance (see allie's paper))
- distance between known good similar lines.
  - list of close matching lines using other paper's data and Allie's data (tumorComparer)
  - fake tumor data (cell line + purity*normal)
- ability to find out misslabeled lines:
  - use list of putative mislabelled (outliers in the bioarxiv paper)
  - create fake misslabeling
- using HCMI's line
- does known gene dependency of typical pecancer lines match with clustering?
- ask sanger for their RNAseq
- make latent space arithmetic
  - plot X_bar of fake samples compared to others
  - show diff expr. on fake samples
- create correlation matrix between X - X_bar, X_cl - X_tu, 

## ideas

### random

- celligner that uses gene loadings found by Josh's tool (Webster)
- celligner is already working on subsets:
  - set of genes are droped when cPCA, when mNN realignment

### important

- celligner that finds [a mapping / a cell line for a tumor] given a specific gene grouping (dependency/geneset..)
- celligner that finds the best set of lines for all groupings for a cancer specific expression signature

## paper architecture

### introduction

### the model

### feature engineering

#### results

### comparison with Celligner v1, MFmap, other?

### data

## reproducibility (+badge)