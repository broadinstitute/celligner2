# celligner2

[![codecov](https://codecov.io/gh/broadinstitute/celligner2/branch/main/graph/badge.svg?token=celligner2_token_here)](https://codecov.io/gh/broadinstitute/celligner2)
[![CI](https://github.com/broadinstitute/celligner2/actions/workflows/main.yml/badge.svg)](https://github.com/broadinstitute/celligner2/actions/workflows/main.yml)

Created by Jérémie Kalfon @jkobject (BroadInsitute, Celligner2 is a new version
of the
[celligner](https://github.com/broadinstitute/celligner/tree/master/celligner)
tool to align cancer transcriptomics data through tumors and models. Find out
more about celligner1 here:
[Global computational alignment of tumor and cell line transcriptional profiles](https://www.nature.com/articles/s41467-020-20294-x)

This method is based on the trVAE/scArches method from the Theis Lab and adds
multiple features to improve its performance for our needs. Amongst those:

- Semi-supervision to classify cell type and any other feature provided. This
  improves the latent space and makes the model focus on what the researcher is
  interested about.
- Improved surgery by allowing to increase model size and freezing trained
  weight.
- Multi dataset MMD on latent space together with better batch mixing. These are
  improvements to method already there and allows the user to :
  - have multiple dataset at once.
  - perform better correction when large bath effects exist. (e.g. between
    Cancer cell lines and frozen tumor tissues)
- Explainable AI tools like LRP with GSEA to look at pathway enrichment to
  understand the features the model is looking at to make a prediction.
- QC methods: getting at quality (using scIB). making interactive umap plots.
  looking at reconstruction, classifications and more..

A next phase of development regards the addition of the **expimap_mode**. In
this mode we have copied the code coming from
[expimap](https://www.biorxiv.org/content/10.1101/2022.02.05.479217v1) so that
the model can use a different latent space, based on gene sets and a decoder
that is replaced by a linear model masked by the genes in each gene set.
references to the expimap mode can be seen in places with the \#expimap comment.
only a partial implementation of that was made. This means some arguments and
functions have been copied from the
[expimap ode](https://github.com/theislab/scarches/tree/master/scarches/models/expimap)
and started to be used and adapted to the Celligner2 codebase. Running it
currently would yields bugs as this is not finished. Some references to the
graph NN model or improvements to the architectural surgery might be seen in the
code and don't have functional implications yet.

More about the model on this presentation:
[Celligner2.0 Update](https://docs.google.com/presentation/d/1KVS9dXTlZs2ekrd5XjcYE6a4xzY-TUHOOhm09_r-uds/edit#slide=id.g34613100a1_0_407)

## Install it

```bash
git clone https://github.com/broadinstitute/celligner2.git
cd ..
pip install -e .
```

#### pypi

**/!\ not functional yet**

```bash
pip install celligner2
```

## Usage

For information on usage please see the different notebooks in __runs/__. Unfortunately a general demo notebook is not yet present. The latest version of the run is in -v4.ipynb.

For information about data generation please see the __data/__ folder.


```py
from celligner2 import BaseClass
from celligner2 import base_function

BaseClass().base_method()
base_function()
```

**/!\ not functional yet**

```bash
$ python -m celligner2
#or
$ celligner2
```

#### About the Code

The code model is the one used by pytorch and the Theis lab. More can be
understood by looking at the code and the usage in the notebook Some base model
functions are implemented as different class (othermodels/base/\_base.py) to be
extended by the model/celligner2*model.py. This file contains the full
definition of the model (with the training, data management and some usage). The
model architecture however is listed in the model/celligner2.py file. additional
key model functions are model/modules and model/losses. The training definition
is in trainers/celligner2/trainer.py which is extended by
trainers/celligner2/semisupervised.py. Dataset management (encoding /
preprocessing etc..) is defined in dataset/* and dataset/celligner2/\_ .
Finally, plotting/ contains plotting/celligner2_eval.py which is the evaluator
of the model. it expects a trained celligner model and can produce many plots
and evaluation of the model, including things related to its use post training,
that would be better placed in the model/celligner2_model.py file.

The definition of things as /base and /celligner2 is made because initially
scArches is a reimplementation of many models where each is reusing and
reimplementing base modules/tools. We decided to keep it this way for ease of
use / collaboration with the Theis lab.

## Development

Read the [CONTRIBUTING.md](CONTRIBUTING.md) file.

Current ongoing tasks are in the Asana project:
[Celligner](https://app.asana.com/read-only/Celligner/9513920295503/79cbcf9c4ec63fc109669a7d8708d4ee/list)
in the Celligner2 _section_.
