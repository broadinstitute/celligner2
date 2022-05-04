import numpy as np
import scanpy as sc
import torch
import os
from collections import Counter
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
from .agg_model import Agg_class

from captum.attr import LRP, Saliency
import gseapy as gp
from genepy import rna

sc.settings.set_figure_params(dpi=500, frameon=False)
sc.set_figure_params(dpi=500)
sc.set_figure_params(figsize=(10, 10))
torch.set_printoptions(precision=3, sci_mode=False, edgeitems=7)
np.set_printoptions(precision=2, edgeitems=7)


def cleanup_annot(annot):
    for val in set(annot) - set(["celligner2_size_factors", "celligner2_labeled"]):
        annot = annot.replace(
            {
                val: {
                    n: n.split(val + "_")[-1] if n.startswith(val) else n
                    for n in set(annot[val])
                },
                val
                + "_pred": {
                    n: n.split(val + "_")[-1] if n.startswith(val) else n
                    for n in set(annot[val])
                },
            }
        )
    return annot


class CELLIGNER2_EVAL:
    def __init__(
        self,
        model: Union[Celligner2, CELLIGNER2],
        trainer: Celligner2Trainer = None,
        additional_adata: AnnData = None,
        only_additional: bool = False,
    ):
        # if type(model) is CELLIGNER2:
        trainer = model.trainer
        # if no predictors:
        adata_latent, fullpred = model.get_latent(
            add_classpred=len(model.predictors_) > 0,
            get_fullpred=True,
            adata=additional_adata,
        )
        if not only_additional and additional_adata is not None:
            adata_latent_more, fullpred_more = model.get_latent(
                add_classpred=len(model.predictors_) > 0, get_fullpred=True
            )
            self.adata_latent = AnnData(
                X=np.vstack([adata_latent.X, adata_latent_more.X]),
                obs=pd.concat([adata_latent.obs, adata_latent_more.obs]),
                var=adata_latent.var,
            )
            self.fullpred = np.vstack([fullpred, fullpred_more])
        else:
            self.adata_latent = adata_latent
            self.fullpred = fullpred
        # else:
        #    self.adata_latent = model.model.get_latent(add_classpred=True)
        self.adata_latent.obs = cleanup_annot(self.adata_latent.obs)
        self.model = model
        self.trainer = trainer
        self.cell_type_names = None
        self.batch_names = None

    def plot_latent(
        self,
        show=True,
        save=False,
        n_neighbors=8,
        dir_path=None,
        umap_kwargs={},
        rerun=True,
        use_genepy=False,
        **kwargs,
    ):
        if save:
            show = False
            if dir_path is None:
                save = False

        if rerun:
            sc.pp.neighbors(self.adata_latent, n_neighbors=n_neighbors)
            sc.tl.leiden(self.adata_latent)
            sc.tl.umap(self.adata_latent, **umap_kwargs)

        if use_genepy:
            plot.scatter()
        else:
            sc.pl.umap(
                self.adata_latent,
                show=show,
                **kwargs,
            )
            if save:
                # create folder if not exists
                if not os.path.exists(dir_path):
                    os.makedirs(dir_path)
                plt.savefig(f"{dir_path}_batch.png", bbox_inches="tight")

    def update_true_class(self, data, label_key="tissue_type"):
        self.adata_latent.obs[label_key] = data

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

    def get_class_quality(
        self,
        only=None,
        on="cell_type",
        classes=["tissue_type", "disease_type", "sex", "age"],
    ):
        if only is not None:
            only = [only] if type(only) is str else only
            obs = self.adata_latent.obs[self.adata_latent.obs[on].isin(only)]
        else:
            obs = self.adata_latent.obs
        for val in classes:
            print(val)
            goodloc = obs[val] != self.model.miss_
            worked = len(
                obs[(np.array(obs[val]) == np.array(obs[val + "_pred"])) & goodloc]
            )
            total = len(obs[goodloc])
            cat = set(obs[val + "_pred"])
            macro = f1_score(
                obs.loc[goodloc, val + "_pred"],
                obs.loc[goodloc, val],
                average="macro",
            )
            weighted = f1_score(
                obs.loc[goodloc, val + "_pred"],
                obs.loc[goodloc, val],
                average="weighted",
            )

            print("all predicted categories: ", cat)
            print("accuracy: ", worked / total)
            print("F1 Score (weigthed): %0.2f" % weighted)
            print("F1 Score (macro): %0.2f" % macro)
            print("\n")
        print("use confusion matrix to get more details")

    def get_confusion_matrix(
        self,
        of="tissue_type",
        only=None,
        on="cell_type",
        doplot=True,
        save=False,
        dir_path="temp/",
        figsize=(10, 10),
        font_scale=1,
    ):
        """getconfusionMatrix returns a confusion matrix for the given label_key.

        Args:
            on (str, optional): The label_key to use for the confusion matrix. Defaults to "tissue_type".
            only ([type], optional): Only show the given classes. Defaults to None.
            on (str, optional): The label_key to use for the batch. Defaults to "cell_type".
            doplot (bool, optional): If True, plot the confusion matrix. Defaults to True.
            save (bool, optional): If True, save the confusion matrix. Defaults to False.
            dir_path (str, optional): The path to save the confusion matrix. Defaults to "temp/".

        Returns:
            pd.DataFrame: The confusion matrix.
        """
        if only is not None:
            only = [only] if type(only) is str else only
            loc = self.adata_latent.obs[on].isin(only)
            adata = self.adata_latent[loc]
        else:
            adata = self.adata_latent
        lab = list(set(adata.obs[of]) - set("U"))

        confusion = confusion_matrix(
            adata.obs[of], adata.obs[of + "_pred"], labels=lab, normalize="true"
        )
        confusion = pd.DataFrame(confusion, index=lab, columns=lab)
        if doplot:
            # if
            # sns.set(font_scale=font_scale)
            plt.figure(figsize=figsize, dpi=300)
            sns.heatmap(
                confusion,
                cmap="Blues",
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

    def get_class_correlation(
        self, of=[], only=None, on="lineage", printabove=0.9, doplot=True
    ):
        """getClassCorrelation returns a correlation matrix for the given label_key.

        Args:
            on (str, optional): The label_key to use for the correlation matrix. Defaults to "tissue_type".
            only ([type], optional): Only show the given classes. Defaults to None.
            on (str, optional): The label_key to use for the batch. Defaults to "cell_type".

        Returns:
            pd.DataFrame: The correlation matrix.
        """
        if only is not None:
            only = [only] if type(only) is str else only
            loc = self.adata_latent.obs[on].isin(only)
            ind = self.adata_latent[loc].obs.index
            fullpred = pd.DataFrame(
                data=self.fullpred[loc], index=ind, columns=self.model.predictors_
            )
        else:
            fullpred = pd.DataFrame(
                data=self.fullpred,
                index=self.adata_latent.obs.index,
                columns=self.model.predictors_,
            )
        if len(of) == 0:
            corr = fullpred.corr()
        else:
            corr = fullpred[of].corr()
        found = []
        print("these are considered to be the same:\n")
        for n, m in zip(*np.where(corr > printabove)):
            if n == m:
                continue
            n = self.model.predictors_[n]
            m = self.model.predictors_[m]
            if (n, m) in found or (m, n) in found:
                continue
            else:
                found.append((n, m))
        print("\n".join([str(i)[1:-1] for i in found]))
        if doplot:
            sns.clustermap(corr, cmap="RdBu_r")
            plt.show()
        return corr, found

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

    def get_model_arch(self):
        for name, p in self.model.named_parameters():
            print(name, " - ", p.size(0), p.size(-1))

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

    def get_asw(self, label_key="tissue_type", on="cell_type"):
        asw_score_batch, asw_score_cell_types = asw(
            adata=self.adata_latent, label_key=label_key, batch_key=on
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

    def impute_missing(self, dataset, classonly=False, only=None):
        """
        Impute missing values in the data by doing reconstruction and using classification.

        Args:
            classonly (bool, optional): If True, only use the classifier to impute missing values. Defaults to False.

        Returns:
            newadata.AnnData: The initial adata dataset with nans replaced.
        """
        badloc = ~self.model.goodloc
        if not classonly:
            dataset.X[badloc] = self.reconstruct(samples=dataset.obs.index).X[badloc]
        samecol = set(dataset.obs.columns) & set(self.adata_latent.obs.columns)
        if only is not None:
            samecol = samecol & set(only)
        loc = dataset.obs[samecol].values == self.model.miss_
        dataset.X[badloc] = np.nan
        n = dataset.obs[samecol].values
        a = self.adata_latent.obs[samecol].values
        n[loc] = a[loc]
        dataset.obs[list(samecol)] = n
        return dataset

    def reconstruct(self, samples, **make_as):
        """reconstruct will reconstruct the given samples and produce a new annData dataset with it

        plot the reconstruction quality.

        Args:
            samples (list[index], optional): index in the latent space to reconstruct. Defaults to None.
                set to None if use group.
            make_as (condition_name:pd.dataframe(condition_values, obs_col)), optional):
                the set of conditions on which to reproduce. Defaults to None.
                If None, use the condition initially given for the set of samples passed.

        Returns:
            AnnData: The reconstructed data.
        """
        conditions = self.model.condition_keys_
        if make_as is None:
            make_as = {"self": self.model.adata.obs.loc[samples][conditions]}
        else:
            make_as.update({"self": self.model.adata.obs.loc[samples][conditions]})
        full = pd.DataFrame()
        fullobs = pd.DataFrame()
        for k, v in make_as.items():
            v = v.copy()
            val = self.model.reconstruct(self.adata_latent[samples].X, v)
            full = full.append(val)
            v.index = [i + "_" + k for i in samples]
            v["group"] = k
            fullobs = fullobs.append(v)
        full.index = fullobs.index
        reco = AnnData(full, obs=fullobs)
        return reco

    def compare_to(self, reco, samples=None):
        """compare_to_reconstruction will compare the given AnnData to the reconstructed data.

        Args:
            reco (AnnData): The reconstructed data.
            samples (list[index], optional): index in the latent space to reconstruct. Defaults to None.

        Returns:
            coeff (pd.DataFrame): The correlation coefficients.
        """
        if samples is None:
            samples = []
        samples = list(samples) + [
            val[:-5] for val in reco.obs.index if val.endswith("_self")
        ]
        if len(set(samples) - set(self.adata_latent.obs.index)) > 0:
            raise ValueError(
                "The samples in the given AnnData do not match the samples in the reconstructed AnnData"
            )
        true = self.model.adata[samples]
        name = true.obs.index.tolist() + reco.obs.index.tolist()
        coeff = pd.DataFrame(data=np.corrcoef(true.X, reco.X), columns=name, index=name)

        _, ax = plt.subplots(figsize=(13, 13))
        sns.heatmap(coeff, ax=ax)
        return coeff

    def explain_predictions(
        self,
        on,
        of,
        using="LRP",
        do_gsea=True,
        sets=[
            "temp/genesets/h.all.v7.5.1.entrez.gmt",
            "temp/genesets/c6.all.v7.5.1.entrez.gmt",
            "temp/genesets/c2.cp.reactome.v7.5.1.entrez.gmt",
            "temp/genesets/c8.all.v7.5.1.entrez.gmt",
        ],
        **kwargs,
    ):
        """
        Explain the predictions of a model.

        Args:
            on (str): The key of the group to use
            of (str): The key of the group to predict
            using (str, optional): The method to use. Defaults to 'LRP'.
            do_gsea (bool, optional): If True, perform GSEA on the predictions. Defaults to True.
        """
        inp = torch.tensor(
            self.model.adata[
                (self.model.adata.obs.tissue_type == on)
                & (self.model.adata.obs.cell_type == of)
            ].X
        )
        if using == "LRP":
            explainor = LRP(
                Agg_class(self.model.model.encoder, self.model.model.classifier)
            )

        elif using == "Saliency":
            explainor = Saliency(
                Agg_class(self.model.model.encoder, self.model.model.classifier)
            )
        elif using == "LIME_LRP":
            explainor = LIME_LRP(
                Agg_class(self.model.model.encoder, self.model.model.classifier)
            )
        else:
            raise ValueError("The given explainor is not valid.")
        attr = explainor.attribute(
            inputs=inp,
            additional_forward_args=torch.tensor([[0, 1]] * inp.shape[0]),
            target=self.model.model.predictor_encoder[on],
        )
        res = pd.DataFrame(
            data=attr.detach().numpy(),
            columns=self.model.adata.var.index,
            index=self.model.adata[
                (self.model.adata.obs.tissue_type == on)
                & (self.model.adata.obs.cell_type == of)
            ].obs.index,
        ).mean()
        if using == "LRP":
            res = res.abs() * 60
        found, _ = rna.convertGenes(
            res.index.tolist(), from_idtype="ensembl_gene_id", to_idtype="entrezgene_id"
        )
        res.index = found
        res = res[~res.index.isna()]
        res.index = res.index.astype(str)
        res = res[~res.index.str.startswith("ENS")]
        res.index = res.index.astype(float).astype(int).astype(str)
        res = res.reset_index().sort_values(by=0, ascending=False)
        if do_gsea:
            return self.do_gsea(res, sets=sets, **kwargs), res
        else:
            return None, res

    def do_gsea(
        self,
        data,
        do_filter=False,
        sets=[
            "temp/genesets/h.all.v7.5.1.entrez.gmt",
            "temp/genesets/c6.all.v7.5.1.entrez.gmt",
            "temp/genesets/c2.cp.reactome.v7.5.1.entrez.gmt",
            "temp/genesets/c8.all.v7.5.1.entrez.gmt",
        ],
        using="prerank",
        **kwargs,
    ):
        """
        Perform GSEA on the predictions.



        """
        if using == "prerank":
            gsea = gp.prerank
        elif using == "gsea":
            gsea = gp.gsea
        else:
            raise ValueError("The given method is not valid.")
        res = pd.concat(
            [gsea(data, val, **kwargs).res2d for val in sets], axis=1
        ).reset_index()
        if do_filter:
            res = res[res.fdr < 0.05]
        return res.sort_values(by="es", ascending=False)

    def define_clusters(self, lim=0.2, col="leiden"):
        """define_clusters will define clusters based on the leiden clustering.

        Args:
            lim (float, optional): The limit of the cluster. Defaults to 0.2.
            col (str, optional): The column to use for the clustering. Defaults to 'leiden'.
        """
        cat = {}
        counts = {}
        for i in set(self.adata_latent.obs[col]):
            group = ""
            lin = self.adata_latent.obs[self.adata_latent.obs.leiden == i].lineage
            count = Counter(lin)
            counts[i] = count
            for n, c in count.items():
                if c > lim * len(lin):
                    group = group + "+" + n
            name = group[1:]
            if name in cat.values():
                c = 2
                name += "_" + str(c)
                while name in cat.values():
                    c += 1
                    name = name[:-1] + str(c)
            cat.update({i: name})
        return cat, counts
