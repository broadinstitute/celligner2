import numpy as np
import scanpy as sc
import torch
import os
import matplotlib.pyplot as plt
from typing import Union
import seaborn as sns
import pandas as pd
from anndata import AnnData
from sklearn.metrics import confusion_matrix, f1_score

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
        # if type(model) is CELLIGNER2:
        trainer = model.trainer
        self.adata_latent = model.get_latent(add_classpred=True)
        # else:
        #    self.adata_latent = model.model.get_latent(add_classpred=True)

        for val in set(self.adata_latent.obs) - set(
            ["celligner2_size_factors", "celligner2_labeled"]
        ):
            self.adata_latent.obs = self.adata_latent.obs.replace(
                {
                    val: {
                        n: n.split(val + "_")[-1] if n.startswith(val) else n
                        for n in set(self.adata_latent.obs[val])
                    },
                    val
                    + "_pred": {
                        n: n.split(val + "_")[-1] if n.startswith(val) else n
                        for n in set(self.adata_latent.obs[val])
                    },
                }
            )

        self.model = model
        self.trainer = trainer
        self.cell_type_names = None
        self.batch_names = None

    def get_model_arch(self):
        for name, p in self.model.named_parameters():
            print(name, " - ", p.size(0), p.size(-1))

    def plot_latent(
        self,
        show=True,
        save=False,
        n_neighbors=8,
        dir_path=None,
        umap_kwargs={},
        **kwargs,
    ):
        if save:
            show = False
            if dir_path is None:
                save = False
        self.adata_latent = AnnData(
            self.adata_latent.X, obs=self.adata_latent.obs, var=self.adata_latent.var
        )
        sc.pp.neighbors(self.adata_latent, n_neighbors=n_neighbors)
        sc.tl.leiden(self.adata_latent)
        sc.tl.umap(self.adata_latent, **umap_kwargs)
        sc.pl.umap(self.adata_latent, show=show, **kwargs)
        if save:
            # create folder if not exists
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            plt.savefig(f"{dir_path}_batch.png", bbox_inches="tight")

    def plot_classification(
        self, classes=["tissue_type", "disease_type", "sex", "age"]
    ):
        for val in classes:
            sankey_diagram(
                np.vstack(
                    [
                        self.adata_latent.obs[val].values,
                        self.adata_latent.obs[val + "_pred"].values,
                    ]
                ).T,
                show=True,
                title="sankey of " + val,
            )

    def get_class_quality(self, classes=["tissue_type", "disease_type", "sex", "age"]):
        for val in classes:
            print(val)
            goodloc = self.adata_latent.obs[val] != self.model.miss_
            worked = len(
                self.adata_latent.obs[
                    (
                        np.array(self.adata_latent.obs[val])
                        == np.array(self.adata_latent.obs[val + "_pred"])
                    )
                    & goodloc
                ]
            )
            total = len(self.adata_latent.obs[goodloc])
            cat = set(self.adata_latent.obs[val + "_pred"])
            score = f1_score(
                self.adata_latent.obs.loc[goodloc, val + "_pred"],
                self.adata_latent.obs.loc[goodloc, val],
                average="macro",
            )
            print("all predicted categories: ", cat)
            print("accuracy: ", worked / total)
            print("F1 Score: %0.2f" % score)
            print("\n")

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
            # create folder if not exists
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            plt.savefig(f"{dir_path}.png", bbox_inches="tight")
        if show:
            plt.show()
        plt.clf()

    def get_ebm(
        self,
        n_neighbors=50,
        label_key="cell_type",
        n_pools=50,
        n_samples_per_pool=100,
        verbose=True,
    ):
        ebm_score = entropy_batch_mixing(
            adata=self.adata_latent,
            label_key=label_key,
            n_neighbors=n_neighbors,
            n_pools=n_pools,
            n_samples_per_pool=n_samples_per_pool,
        )
        if verbose:
            print("Entropy of Batchmixing-Score: %0.2f" % ebm_score)
        return ebm_score

    def get_knn_purity(self, label_key="tissue_type", n_neighbors=10, verbose=True):
        knn_score = knn_purity(
            adata=self.adata_latent, label_key=label_key, n_neighbors=n_neighbors
        )
        if verbose:
            print("KNN Purity-Score:  %0.2f" % knn_score)
        return knn_score

    def get_asw(self, label_key="tissue_type", batch_key="cell_type"):
        asw_score_batch, asw_score_cell_types = asw(
            adata=self.adata_latent, label_key=label_key, batch_key=batch_key
        )
        print("ASW on batch:", asw_score_batch)
        print("ASW on celltypes:", asw_score_cell_types)
        return asw_score_batch, asw_score_cell_types

    def get_nmi(self, label_key="tissue_type"):
        nmi_score = nmi(adata=self.adata_latent, label_key=label_key)
        print("NMI score:", nmi_score)
        return nmi_score

    def get_latent_score(self, label_key="tissue_type"):
        ebm = self.get_ebm(verbose=False, label_key=label_key)
        knn = self.get_knn_purity(verbose=False, label_key=label_key)
        score = ebm + knn
        print(
            "Latent-Space Score EBM+KNN, EBM, KNN: %0.2f, %0.2f, %0.2f"
            % (score, ebm, knn)
        )
        return score

    def getconfusionMatrix(
        self,
        on="tissue_type",
        only=None,
        batch_key="cell_type",
        doplot=True,
        save=False,
        dir_path="temp/",
        figsize=(13, 13),
        font_scale=10,
    ):
        """getconfusionMatrix returns a confusion matrix for the given label_key.

        Args:
            on (str, optional): The label_key to use for the confusion matrix. Defaults to "tissue_type".
            only ([type], optional): Only show the given classes. Defaults to None.
            batch_key (str, optional): The label_key to use for the batch. Defaults to "cell_type".
            doplot (bool, optional): If True, plot the confusion matrix. Defaults to True.
            save (bool, optional): If True, save the confusion matrix. Defaults to False.
            dir_path (str, optional): The path to save the confusion matrix. Defaults to "temp/".

        Returns:
            pd.DataFrame: The confusion matrix.
        """
        if only is not None:
            loc = self.adata_latent.obs[batch_key] == only
            adata = self.adata_latent[loc]
        else:
            adata = self.adata_latent
        lab = list(set(adata.obs[on]) - set("U"))
        confusion = confusion_matrix(
            adata.obs[on], adata.obs[on + "_pred"], labels=lab, normalize="true"
        )
        confusion = pd.DataFrame(confusion, index=lab, columns=lab)
        if doplot:
            # if
            sns.set(font_scale=0.4)
            plt.figure(figsize=figsize, dpi=300)
            sns.heatmap(
                confusion,
                cmap="Blues",
                annot_kws={"size": font_scale},
            )
            plt.title("Confusion Matrix")
            plt.ylabel("True")
            plt.xlabel("Predicted")
            if save:
                # create dir if not exists
                if not os.path.exists(dir_path):
                    os.makedirs(dir_path)
                plt.savefig(f"{dir_path}_confusion.png", bbox_inches="tight")
            plt.show()

        return confusion

    def reconstruct(self, c):
        """reconstruct returns the reconstructed data.

        Args:
            c pd.dataframe: The batch.

        Returns:
            AnnData: The reconstructed data.
        """
        return self.model.reconstruct(self.adata_latent, c)

    def reconstructionQC(self, samples=None, group=None, on="", reco_as=None):
        if samples is not None:
            if group is not None:
                raise ValueError("Only one of samples and group can be given")
            size = len(samples)
            if reco_as is not None:
                reco_a = self.reconstruct(self.adata_latent[samples].X, reco_as)
            reco_b = self.reconstruct(
                self.adata_latent[samples].X,
                pd.DataFrame(data=["historical_cl"] * size, columns=["cell_type"]),
            )
            true = self.model.adata[samples].X
            reco_b.columns = self.adata.var.index
            reco_b = AnnData(reco_b, self.adata.obs.iloc[:size], self.adata.var)
            reco_a.columns = self.adata.var.index
            reco_a = AnnData(reco_a, self.adata.obs.iloc[:size], self.adata.var)
            true.columns = self.adata.var.index
            true = AnnData(true, self.adata.obs.iloc[:size], self.adata.var)
            name = true.obs.index.tolist() + [i + "_reco" for i in reco_b.obs.index]
            coeff = pd.DataFrame(
                data=np.corrcoef(true.X, reco_b.X), columns=name, index=name
            )
        # if group is not None:
        #
        f, ax = plt.subplots(figsize=(20, 20))
        sns.heatmap(coeff, ax=ax)
