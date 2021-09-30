import numpy as np
import scanpy as sc
import torch
import anndata
import matplotlib.pyplot as plt
from typing import Union
from itertools import product
from .dataset.trvae._utils import label_encoder
from .metrics.metrics import entropy_batch_mixing, knn_purity, asw, nmi
from .models import trVAE, TRVAE
from .trainers import trVAETrainer

sc.settings.set_figure_params(dpi=200, frameon=False)
sc.set_figure_params(dpi=200)
torch.set_printoptions(precision=3, sci_mode=False, edgeitems=7)
np.set_printoptions(precision=2, edgeitems=7)

import matplotlib
from . import _alluvial


def sankey_diagram(data, save_path=None, show=False, **kwargs):
    """Draws Sankey diagram for the given ``data``.
        Parameters
        ----------
        data: :class:`~numpy.ndarray`
            array with 2 columns. One for predictions and another for true values.
        save_path: str
            Path to save the drawn Sankey diagram. if ``None``, the diagram will not be saved.
        show: bool
            if ``True`` will show the diagram.
        kwargs:
            additional arguments for diagram configuration. See ``_alluvial.plot`` function.
    """
    font = {'family': 'Arial',
            'weight': 'bold',
            'size': 14}

    matplotlib.rc('font', **font)
    matplotlib.rc('ytick', labelsize=14)
    matplotlib.rc('xtick', labelsize=14)
    plt.close('all')
    ax = _alluvial.plot(data.tolist(),
                        color_side=kwargs.get("color_side", 1),
                        alpha=kwargs.get("alpha", 0.5),
                        x_range=kwargs.get("x_range", (0, 1)),
                        res=kwargs.get("res", 20),
                        figsize=kwargs.get("figsize", (21, 15)),
                        disp_width=kwargs.get("disp_width", True),
                        width_in=kwargs.get("width_in", True),
                        wdisp_sep=kwargs.get("wdisp_sep", ' ' * 2),
                        cmap=kwargs.get("cmap", matplotlib.cm.get_cmap('jet')),
                        v_gap_frac=kwargs.get("v_gap_frac", 0.03),
                        h_gap_frac=kwargs.get("h_gap_frac", 0.03),
                        labels=kwargs.get("labels", None),
                        fontname=kwargs.get("fontname", "Arial"),
                        )
    if save_path is not None:
        plt.savefig(save_path, dpi=kwargs.get("dpi", 200), bbox_inches='tight')
    if show:
        plt.show()
    plt.close()


def plot_abs_bfs_key(scores, terms, key, n_points=30, lim_val=2.3, fontsize=8, scale_y=2, yt_step=0.3,
                     title=None, ax=None):

    txt_args = dict(
        rotation='vertical',
        verticalalignment='bottom',
        horizontalalignment='center',
        fontsize=fontsize,
    )

    ax = ax if ax is not None else plt.axes()
    ax.grid(False)

    bfs = np.abs(scores[key]['bf'])
    srt = np.argsort(bfs)[::-1][:n_points]
    top = bfs.max()

    ax.set_ylim(top=top * scale_y)
    yt = np.arange(0, top * 1.1, yt_step)
    ax.set_yticks(yt)

    ax.set_xlim(0.1, n_points + 0.9)
    xt = np.arange(0, n_points + 1, 5)
    xt[0] = 1
    ax.set_xticks(xt)

    for i, (bf, term) in enumerate(zip(bfs[srt], terms[srt])):
        ax.text(i+1, bf, term, **txt_args)

    ax.axhline(y=lim_val, color='red', linestyle='--', label='')

    ax.set_xlabel("Rank")
    ax.set_ylabel("Absolute log bayes factors")
    ax.set_title(key if title is None else title)

    return ax.figure

def plot_abs_bfs(adata, scores_key="bf_scores", terms: Union[str, list]="terms",
                 keys=None, n_cols=3, **kwargs):

    scores = adata.uns[scores_key]

    if isinstance(terms, str):
        terms = adata.uns[terms]

    if len(terms) != len(next(iter(scores.values()))["bf"]):
        raise ValueError('Incorrect length of terms.')

    if keys is None:
        keys = list(scores.keys())

    if len(keys) == 1:
        keys = keys[0]

    if isinstance(keys, str):
        return plot_abs_bfs_key(scores, terms, keys, **kwargs)

    n_keys = len(keys)

    if n_keys <= n_cols:
        n_cols = n_keys
        n_rows = 1
    else:
        n_rows = int(np.ceil(n_keys / n_cols))

    fig, axs = plt.subplots(n_rows, n_cols)
    for key, ix in zip(keys, product(range(n_rows), range(n_cols))):
        if n_rows == 1:
            ix = ix[1]
        elif n_cols == 1:
            ix = ix[0]
        plot_abs_bfs_key(scores, terms, key, ax=axs[ix], **kwargs)

    n_inactive = n_rows * n_cols - n_keys
    if n_inactive > 0:
        for i in range(n_inactive):
            axs[n_rows-1, -(i+1)].axis('off')

    return fig

class TRVAE_EVAL:
    def __init__(
            self,
            model: Union[trVAE, TRVAE],
            adata: anndata.AnnData,
            trainer: trVAETrainer = None,
            condition_key: str = None,
            cell_type_key: str = None
    ):
        if type(model) is TRVAE:
            trainer = model.trainer
            model = model.model

        self.model = model
        self.trainer = trainer
        self.adata = adata
        self.device = model.device
        self.conditions, _ = label_encoder(
            self.adata,
            encoder=model.condition_encoder,
            condition_key=condition_key,
        )
        self.cell_type_names = None
        self.batch_names = None
        if cell_type_key is not None:
            self.cell_type_names = adata.obs[cell_type_key].tolist()
        if condition_key is not None:
            self.batch_names = adata.obs[condition_key].tolist()

        self.adata_latent = self.latent_as_anndata()

    def latent_as_anndata(self):
        if self.model.calculate_mmd == 'z' or self.model.use_mmd == False:
            latent = self.model.get_latent(
                self.adata.X,
                c=self.conditions,
            )
        else:
            latent = self.model.get_y(
                self.adata.X,
                c=self.conditions
            )
        adata_latent = sc.AnnData(latent)
        if self.cell_type_names is not None:
            adata_latent.obs['cell_type'] = self.cell_type_names
        if self.batch_names is not None:
            adata_latent.obs['batch'] = self.batch_names
        return adata_latent

    def get_model_arch(self):
        for name, p in self.model.named_parameters():
            print(name, " - ", p.size(0), p.size(-1))

    def plot_latent(self,
                    show=True,
                    save=False,
                    dir_path=None,
                    n_neighbors=8,
                    ):
        if save:
            show=False
            if dir_path is None:
                save = False

        sc.pp.neighbors(self.adata_latent, n_neighbors=n_neighbors)
        sc.tl.umap(self.adata_latent)
        color = [
            'cell_type' if self.cell_type_names is not None else None,
            'batch' if self.batch_names is not None else None,
        ]
        sc.pl.umap(self.adata_latent,
                   color=color,
                   frameon=False,
                   wspace=0.6,
                   show=show)
        if save:
            plt.savefig(f'{dir_path}_batch.png', bbox_inches='tight')

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

    def get_ebm(self, n_neighbors=50, n_pools=50, n_samples_per_pool=100, verbose=True):
        ebm_score = entropy_batch_mixing(
            adata=self.adata_latent,
            label_key='batch',
            n_neighbors=n_neighbors,
            n_pools=n_pools,
            n_samples_per_pool=n_samples_per_pool
        )
        if verbose:
            print("Entropy of Batchmixing-Score: %0.2f" % ebm_score)
        return ebm_score

    def get_knn_purity(self, n_neighbors=50, verbose=True):
        knn_score = knn_purity(
            adata=self.adata_latent,
            label_key='cell_type',
            n_neighbors=n_neighbors
        )
        if verbose:
            print("KNN Purity-Score:  %0.2f" % knn_score)
        return knn_score

    def get_asw(self):
        asw_score_batch, asw_score_cell_types = asw(adata=self.adata_latent, label_key='cell_type', batch_key='batch')
        print("ASW on batch:", asw_score_batch)
        print("ASW on celltypes:", asw_score_cell_types)
        return asw_score_batch, asw_score_cell_types

    def get_nmi(self):
        nmi_score = nmi(adata=self.adata_latent, label_key='cell_type')
        print("NMI score:", nmi_score)
        return nmi_score

    def get_latent_score(self):
        ebm = self.get_ebm(verbose=False)
        knn = self.get_knn_purity(verbose=False)
        score = ebm + knn
        print("Latent-Space Score EBM+KNN, EBM, KNN: %0.2f, %0.2f, %0.2f" % (score, ebm, knn))
        return score