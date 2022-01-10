import numpy as np
import scanpy as sc
import torch
import anndata
import matplotlib.pyplot as plt
from typing import Union

from celligner2.dataset.celligner2._utils import label_encoder
from celligner2.metrics.metrics import entropy_batch_mixing, knn_purity, asw, nmi
from celligner2.model import Celligner2, CELLIGNER2
from celligner2.trainers import Celligner2Trainer
from .sankey import sankey_diagram

sc.settings.set_figure_params(dpi=500, frameon=False)
sc.set_figure_params(dpi=500)
sc.set_figure_params(figsize=(10, 10))
torch.set_printoptions(precision=3, sci_mode=False, edgeitems=7)
np.set_printoptions(precision=2, edgeitems=7)


class CELLIGNER2_EVAL:
    def __init__(
            self,
            model: Union[Celligner2, CELLIGNER2],
            trainer: Celligner2Trainer = None,
    ):
        if type(model) is CELLIGNER2:
            trainer = model.trainer
            self.adata_latent = model.get_latent(add_classpred=True)
        else:
            self.adata_latent = model.get_latent(add_classpred=True)

        self.model = model
        self.trainer = trainer
        self.cell_type_names = None
        self.batch_names = None
        self.adata_latent = self.model.get_latent(add_classpred=True)

    def get_model_arch(self):
        for name, p in self.model.named_parameters():
            print(name, " - ", p.size(0), p.size(-1))

    def plot_latent(self,
                    show=True,
                    save=False,
                    n_neighbors=8,
                    dir_path=None,
                    umap_kwargs={},
                    **kwargs):
        if save:
            show=False
            if dir_path is None:
                save = False

        sc.pp.neighbors(self.adata_latent, n_neighbors=n_neighbors)
        sc.tl.leiden(self.adata_latent)
        sc.tl.umap(self.adata_latent, **umap_kwargs)
        sc.pl.umap(self.adata_latent, show=show, **kwargs)
        if save:
            plt.savefig(f'{dir_path}_batch.png', bbox_inches='tight')

    def plot_classification(self, classes=['tissue_type', 'disease_type', 'sex', 'age']):
        for val in classes:
            sankey_diagram(
                np.vstack([
                    self.adata_latent.obs[val].values, 
                    self.adata_latent.obs[val+"_pred"].values
                ]).T,
                show=True,
                title='sankey of ' + val,
            )


    def get_class_quality(self, classes=['tissue_type', 'disease_type', 'sex', 'age']):
        for val in classes:
            print(val)
            worked = len(self.adata_latent.obs[np.array(
                self.adata_latent.obs[val])==np.array(self.adata_latent.obs[val+'_pred']
            )])
            total = len(self.adata_latent)
            cat = set(self.adata_latent.obs[val+'_pred'])
            print('all predicted categories: ', cat)
            print('percent of correct predictions: ', worked/total)
            print('\n')
        

    def plot_history(self, show=True, save=False, dir_path=None):
        if save:
            show = False
            if dir_path is None:
                save = False

        if self.trainer is None:
            print("Not possible if no trainer is provided")
            return
        fig = plt.figure()
        elbo_train = self.trainer.logs["epoch_loss"]
        elbo_test = self.trainer.logs["val_loss"]
        x = np.linspace(0, len(elbo_train), num=len(elbo_train))
        plt.plot(x, elbo_train, label="Train")
        plt.plot(x, elbo_test, label="Validate")
        plt.ylim(min(elbo_test) - 50, max(elbo_test) + 50)
        plt.legend()
        if save:
            plt.savefig(f'{dir_path}.png', bbox_inches='tight')
        if show:
            plt.show()
        plt.clf()

    def get_ebm(self, n_neighbors=50, label_key='cell_type', n_pools=50, n_samples_per_pool=100, verbose=True):
        ebm_score = entropy_batch_mixing(
            adata=self.adata_latent,
            label_key=label_key,
            n_neighbors=n_neighbors,
            n_pools=n_pools,
            n_samples_per_pool=n_samples_per_pool
        )
        if verbose:
            print("Entropy of Batchmixing-Score: %0.2f" % ebm_score)
        return ebm_score

    def get_knn_purity(self, label_key='tissue_type', n_neighbors=10, verbose=True):
        knn_score = knn_purity(
            adata=self.adata_latent,
            label_key=label_key,
            n_neighbors=n_neighbors
        )
        if verbose:
            print("KNN Purity-Score:  %0.2f" % knn_score)
        return knn_score

    def get_asw(self, label_key='tissue_type', batch_key='cell_type'):
        asw_score_batch, asw_score_cell_types = asw(adata=self.adata_latent, label_key=label_key, batch_key=batch_key)
        print("ASW on batch:", asw_score_batch)
        print("ASW on celltypes:", asw_score_cell_types)
        return asw_score_batch, asw_score_cell_types

    def get_nmi(self, label_key='tissue_type'):
        nmi_score = nmi(adata=self.adata_latent, label_key=label_key)
        print("NMI score:", nmi_score)
        return nmi_score

    def get_latent_score(self, label_key='tissue_type'):
        ebm = self.get_ebm(verbose=False, label_key=label_key)
        knn = self.get_knn_purity(verbose=False, label_key=label_key)
        score = ebm + knn
        print("Latent-Space Score EBM+KNN, EBM, KNN: %0.2f, %0.2f, %0.2f" % (score, ebm, knn))
        return score
