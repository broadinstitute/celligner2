# Notes

## Objectives

### run scARCHES on CCLE + TCGA (use the batch annotation)

DONE

### run scARCHES on CCLE + TCGA + CCLF + additional

DONE

### semi supervision like mfMAP (need good annotations)

- make MMD work
- there might be a batch issue with the MMD regularization.. on only 1400 CCLE samples..
- sample specific information (will need ashir's new annotations, also annotations for each dataset used)
- label smoothing on semi-supervision (weak supervion)
- add Lr scheduler to TRVAE (ReduceLROnPlateau)

### do data augmentation thing: (see list in other document)

- Use replicates when available.
- Up to some random noise, based on how high is the expression (randomly add)
- mix cell lines and normals to create fake impurity to correct
  - do: ccle_line + impurity * gtex_tissue
  - put semi supervision annotation as impurity: impurity, origin: gtex_tissue
- add CCLE / TCGA hg19 expression
- add pseudo bulk from scRNAseq  (list in asana)

- add mice data (once in graph)

### add Ashir's annotation to the data and make it cross modal (add more QC)

### make it graph like and deeper, and add edge prediction

https://github.com/kipoi/models/blob/master/Xpresso/kipoi_example.ipynb
https://github.com/dmlc/dgl/tree/0.7.x/examples/pytorch/vgae
https://github.com/shionhonda/gae-dgl/blob/master/gae_dgl/prepare_data.py
https://optuna.org/

- use dgl's vgae
- skip connections
- add graph embedding learning and edge prediction
  - drop links from model and train it to classify missing links (positive, negative)

### purify rnaseq

did someone already do it? NO
make cibersortX's copy work
make our own cibersort version?
remake of cibersortX https://github.com/ysuzukilab/Cibersortx
--> use : https://github.com/icbi-lab/immunedeconv (EPIC or cibersort)

add correction of bias between single cell and bulk (use william's method)

### what mfmap is not doing

- no purity estimation?
- not robust to outliers / not able to detect undif cluster
- Maybe it should use gene network from cmap data.

## what we would want to do

min(A - (X_a*Y_a + I_a))
min(B - (X_b*Y_b + I_b)) s.t. min(MIN(dist(X_a, X_b))) ; max(Y_a\*Y_b)

## How to test

- confounding matrix (cell type distance (see allie's paper))
  - plot
- distance between known good similar lines.
  - list of close matching lines using other paper's data and Allie's data (tumorComparer)
  - fake tumor data (cell line + purity*normal)
  - 2D pllot with heat color for distances to good qual data
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