import inspect
from lib2to3.pgen2.token import OP
import os

import anndata

from numpy.lib.function_base import _percentile_dispatcher
import torch
import pickle
import numpy as np
import pandas as pd

from anndata import AnnData
from typing import Optional, Union

from .celligner2 import Celligner2
from celligner2.trainers.celligner2.semisupervised import Celligner2Trainer
from celligner2.othermodels.base._base import BaseMixin, SurgeryMixin, CVAELatentsMixin
from celligner2.dataset.celligner2._utils import label_encoder_2D


class CELLIGNER2(BaseMixin, SurgeryMixin, CVAELatentsMixin):
    """Model for scArches class. This class contains the implementation of Conditional Variational Auto-encoder.

    Parameters
    ----------
    adata: : `~anndata.AnnData`
         Annotated data matrix. Has to be count data for 'nb' and 'zinb' loss and normalized log transformed data
         for 'mse' loss.
    condition_keys: String
         column name of conditions in `adata.obs` data frame.
    conditions: List
         List of Condition names that the used data will contain to get the right encoding when used after reloading.
    hidden_layer_sizes: List
         A list of hidden layer sizes for encoder network. Decoder network will be the reversed order.
    latent_dim: Integer
         Bottleneck layer (z)  size.
    dr_rate: Float
         Dropput rate applied to all layers, if `dr_rate`==0 no dropout will be applied.
    use_mmd: Boolean
         If 'True' an additional MMD loss will be calculated on the latent dim. 'z' or the first decoder layer 'y'.
    mmd_on: String
         Choose on which layer MMD loss will be calculated on if 'use_mmd=True': 'z' for latent dim or 'y' for first
         decoder layer.
    mmd_boundary: Integer or None
         Choose on how many conditions the MMD loss should be calculated on. If 'None' MMD will be calculated on all
         conditions.
    recon_loss: String
         Definition of Reconstruction-Loss-Method, 'mse', 'nb' or 'zinb'.
    beta: Float
         Scaling Factor for MMD loss
    use_bn: Boolean
         If `True` batch normalization will be applied to layers.
    use_ln: Boolean
        If `True` layer normalization will be applied to layers.
    mask: Array or List
        if not None, an array of 0s and 1s from utils.add_annotations to create VAE with a masked linear decoder.
    mask_key: String
        A key in `adata.varm` for the mask if the mask is not provided.
    soft_mask: Boolean
        Use soft mask option. If True, the model will enforce mask with L1 regularization
        instead of multipling weight of the linear decoder by the binary mask.
    n_unconstrained: Integer
        Number of unconstrained terms in the latent layer.
    use_hsic: Boolean
        If True, add HSIC regularization for unconstarined extension terms.
        Used for query mapping.
    hsic_one_vs_all: Boolean
        If True, calculates the sum of HSIC losses for each unconstarined term vs the other terms.
        If False, calculates HSIC for all unconstarined terms vs the other terms.
        Used for query mapping.
    use_own_kl: Boolean
        If `True` the KL-Divergence will be calculated by the network itself. Otherwise torch
    miss: str
        the str value representing the missing data in the obs.
    apply_log: bool
        whether or not to log transform the expression data
    batch_knowledge: bool
        whether or not to give the model batch information (at the encoder/decoder level (TRVAE))
    main_dataset: str
        if one is provided, will do MMD only on the distance of other datasets to that one.
    classifier_hidden_layer_sizes: list
        a list with the number of hidden layers for the classifier.
    betaclass: int
        the weight of the classification loss.
    predictors: list
        a list of the different values in the obs dataframe that we need to predict on (can be across multiple columns). defaults to everything available.
    predictor_keys: list
        a list of only a subset of columns of the obs on which to get all available values as things to classify on.
    res_mult: int
        UNUSED. number of resnet blocks to use
    graph_layers: list
        UNUSED. a definition of the graph layers' sizes
    use_l_encoder: bool
        WIP. the l_encoder encodes into the genesets.. (see expimap code)
    """

    def __init__(
        self,
        adata: AnnData,
        hidden_layer_sizes: list = [256, 64],
        latent_dim: int = 10,
        dr_rate: float = 0.05,
        use_bn: bool = False,
        use_ln: bool = True,
        use_own_kl: bool = False,
        miss: str = "U",
        apply_log: bool = True,
        # condition pat
        condition_keys: Optional[list] = None,
        conditions: Optional[list] = None,
        use_mmd: bool = True,
        mmd_on: str = "z",
        mmd_boundary: Optional[int] = None,
        recon_loss: Optional[str] = "nb",
        beta: float = 1,
        batch_knowledge: bool = True,
        main_dataset=None,
        # classification part
        classifier_hidden_layer_sizes: list = [64, 32],
        betaclass: float = 0.8,
        predictors: Optional[list] = None,
        predictor_keys: Optional[list] = [],
        # GNN part
        res_mult: int = 0,
        graph_layers: int = 0,
        # expimap part
        mask: Optional[Union[np.ndarray, list]] = None,
        mask_key: str = "",
        n_unconstrained: int = 0,
        use_hsic: bool = False,
        hsic_one_vs_all: bool = False,
        use_l_encoder: bool = False,
        # only on load not to use directly
        n_expand: int = 0,
        ext_mask: Optional[Union[np.ndarray, list]] = None,
        ext_n_unconstrained: int = 0,
        predictor_set={},
        condition_set={},
    ):
        self.adata = adata
        self.goodloc = ~np.isnan(adata.X)

        self.condition_keys_ = condition_keys
        self.predictor_keys_ = predictor_keys

        if conditions is None:
            if condition_keys is not None:
                myset = set()
                for condition_key in condition_keys:
                    if len(set(adata.obs[condition_key]) & set(miss)) > 0:
                        raise ValueError(
                            "Condition key '{}' has missing values. the model can't deal \
                                with missing values in its condition keys for now, \
                                    you can run them as predictor to impute them from the data".format(
                                condition_key
                            )
                        )
                    group = set(adata.obs[condition_key]) - set(miss)
                    overlap = group & myset
                    if len(overlap) > 0:
                        adata.obs.replace(
                            {
                                condition_key: {
                                    val: condition_key + "_" + val for val in group
                                }
                            },
                            inplace=True,
                        )
                    myset = myset | set(adata.obs[condition_key])
                self.conditions_ = list(
                    set(adata.obs[condition_keys].values.flatten()) - set(miss)
                )
            else:
                self.conditions_ = []
        else:
            self.conditions_ = conditions

        # TODO: add a version when no condition_keys are provided
        if condition_keys is not None:
            self.condition_set_ = {
                key: set(adata.obs[key]) - set(miss) for key in condition_keys
            }
        # we only want the current's adata condition set.
        if predictors is None:
            if predictor_keys is not None:
                myset = set()
                for predictor_key in predictor_keys:
                    group = set(adata.obs[predictor_key]) - set(miss)
                    if len(group) == 0:
                        raise ValueError(
                            "Predictor key '{}' has no values. please check your obs".format(
                                predictor_key
                            )
                        )
                    overlap = group & myset
                    if len(overlap) > 0:
                        adata.obs.replace(
                            {
                                predictor_key: {
                                    val: predictor_key + "_" + val for val in group
                                }
                            },
                            inplace=True,
                        )
                    myset = myset | set(adata.obs[predictor_key])
                self.predictors_ = list(
                    set(adata.obs[predictor_keys].values.flatten()) - set(miss)
                )
            else:
                self.predictors_ = []

        else:
            self.predictors_ = predictors

        if predictor_keys is not None:
            self.predictor_set_ = {
                key: set(adata.obs[key]) - set(miss) for key in predictor_keys
            }
        else:
            self.predictor_set_ = {}
        for k, v in predictor_set.items():
            self.predictor_set_[k] = set(v) & set(self.predictor_set_[k])

        self.miss_ = miss
        self.hidden_layer_sizes_ = hidden_layer_sizes
        self.classifier_hidden_layer_sizes_ = classifier_hidden_layer_sizes
        self.latent_dim_ = latent_dim
        self.dr_rate_ = dr_rate
        self.use_mmd_ = use_mmd
        self.mmd_on_ = mmd_on
        self.mmd_boundary_ = mmd_boundary

        # expimap mode params
        self.expimap_mode_ = False
        if mask_key != "":
            mask = adata.varm[mask_key].T
        if mask is not None:
            mask = mask if isinstance(mask, list) else mask.tolist()
            self.mask_ = torch.tensor(mask).float()
            self.expimap_mode_ = True
            self.latent_dim_ = len(self.mask_) + n_unconstrained
        else:
            self.mask_ = None
        self.n_unconstrained_ = n_unconstrained
        self.use_hsic_ = use_hsic
        self.hsic_one_vs_all_ = hsic_one_vs_all
        self.ext_mask_ = ext_mask
        # end of expimap mode params

        self.recon_loss_ = recon_loss
        self.beta_ = beta
        self.betaclass_ = betaclass
        self.use_bn_ = use_bn
        self.use_ln_ = use_ln
        self.use_own_kl_ = use_own_kl

        self.n_expand_ = n_expand
        self.use_l_encoder_ = use_l_encoder
        self.ext_n_unconstrained_ = ext_n_unconstrained

        self.res_mult_ = res_mult
        self.graph_layers_ = graph_layers
        self.batch_knowledge_ = batch_knowledge

        self.input_dim_ = adata.n_vars
        self.apply_log_ = apply_log
        if main_dataset not in set(adata.obs[condition_keys].values.flatten()):
            print("main dataset not in conditions, removing..")
            self.main_dataset_ = None
        else:
            self.main_dataset_ = main_dataset

        self.model = Celligner2(
            self.input_dim_,
            self.hidden_layer_sizes_,
            self.latent_dim_,
            self.dr_rate_,
            self.use_own_kl_,
            self.recon_loss_,
            self.use_bn_,
            self.use_ln_,
            self.apply_log_,
            # condition params
            self.conditions_,
            self.use_mmd_,
            self.mmd_on_,
            self.mmd_boundary_,
            self.beta_,
            self.main_dataset_,
            self.batch_knowledge_,
            # predictor params
            self.predictors_,
            self.classifier_hidden_layer_sizes_,
            self.betaclass_,
            # GNN params
            self.graph_layers_,
            self.res_mult_,
            # expimap mode params
            self.n_expand_,
            self.expimap_mode_,
            self.mask_,
            self.ext_mask_,
            self.n_unconstrained_,
            self.ext_n_unconstrained_,
            self.use_l_encoder_,
            self.use_hsic_,
            self.hsic_one_vs_all_,
        )

        self.is_trained_ = False
        self.trainer = None

    def train(self, n_epochs: int = 400, lr: float = 1e-3, eps: float = 0.01, **kwargs):
        """Train the model.

        Parameters
        ----------
        n_epochs
             Number of epochs for training the model.
        lr
             Learning rate for training the model.
        eps
             torch.optim.Adam eps parameter
        kwargs
             kwargs for the TrVAE trainer.
        """
        self.trainer = Celligner2Trainer(
            self.model,
            self.adata,
            condition_keys=self.condition_keys_,
            predictor_keys=self.predictor_keys_,
            **kwargs
        )
        self.trainer.train(n_epochs, lr, eps)
        self.is_trained_ = True

    @classmethod
    def _get_init_params_from_dict(cls, dct):
        print(dct.keys())
        init_params = {
            "condition_keys": dct["condition_keys_"],
            "conditions": dct["conditions_"],
            "hidden_layer_sizes": dct["hidden_layer_sizes_"],
            "classifier_hidden_layer_sizes": dct["classifier_hidden_layer_sizes_"]
            if "classifier_hidden_layer_sizes_" in dct.keys()
            else [],
            "latent_dim": dct["latent_dim_"],
            "dr_rate": dct["dr_rate_"],
            "use_mmd": dct["use_mmd_"],
            "mmd_on": dct["mmd_on_"],
            "mmd_boundary": dct["mmd_boundary_"],
            "recon_loss": dct["recon_loss_"],
            "beta": dct["beta_"],
            "betaclass": dct["betaclass_"],
            "use_bn": dct["use_bn_"],
            "use_ln": dct["use_ln_"],
            "use_own_kl": dct["use_own_kl_"],
            "predictors": dct["predictors_"],
            "predictor_keys": dct["predictor_keys_"],
            "miss": dct["miss_"],
            "apply_log": dct["apply_log_"],
            "predictor_set": dct["predictor_set_"]
            if "predictor_set_" in dct.keys()
            else {},
            "condition_set": dct["condition_set_"]
            if "condition_set_" in dct.keys()
            else {},
            # main dataset mode params
            "main_dataset": dct["main_dataset_"]
            if "main_dataset_" in dct.keys()
            else None,
            # expimap mode params
            "mask": dct["mask_"] if "mask_" in dct.keys() else None,
            "mask_key": dct["mask_key_"] if "mask_key_" in dct.keys() else "",
            "n_unconstrained": dct["n_unconstrained_"]
            if "n_unconstrained_" in dct.keys()
            else 0,
            "use_hsic": dct["use_hsic_"] if "use_hsic_" in dct.keys() else False,
            "hsic_one_vs_all": dct["hsic_one_vs_all_"]
            if "hsic_one_vs_all_" in dct.keys()
            else False,
        }

        return init_params

    @classmethod
    def _validate_adata(cls, adata, dct):
        if adata.n_vars != dct["input_dim_"]:
            raise ValueError("Incorrect var dimension")

    def get_latent(
        self,
        adata: Optional[AnnData] = None,
        mean: bool = False,
        add_classpred: bool = False,
        get_fullpred: bool = False,
    ):
        """Map `x` in to the latent space. This function will feed data in encoder  and return  z for each sample in
        data.
        Parameters
        ----------
        x
             Numpy nd-array to be mapped to latent space. `x` has to be in shape [n_obs, input_dim].
             If None, then `self.adata.X` is used.
        c
             `numpy nd-array` of original (unencoded) desired labels for each sample.
        mean
             return mean instead of random sample from the latent space
        Returns
        -------
             Returns array containing latent space encoding of 'x'.
        """
        device = next(self.model.parameters()).device
        wasnull = False
        if adata is None:
            wasnull = True
            adata = self.adata
        conditions = label_encoder_2D(
            adata,
            encoder=self.model.condition_encoder,
            label_sets=self.condition_set_,
        )

        c = torch.tensor(conditions, dtype=torch.long)
        x = torch.tensor(adata.X)

        latents = []
        classes = []
        indices = torch.arange(x.size(0))
        subsampled_indices = indices.split(512)

        for batch in subsampled_indices:

            latent, classe = self.model.get_latent(
                x[batch, :].to(device), c[batch], mean, add_classpred
            )
            latents += [latent.cpu().detach()]

            if add_classpred:
                classes += [classe.cpu().detach()]

        if add_classpred:
            # import pdb; pdb.set_trace()
            predictor_decoder = {v: k for k, v in self.model.predictor_encoder.items()}
            predictions = np.array(torch.cat(classes))
            classes = []
            # TODO: use conditions_ if set is None:
            for _, v in self.predictor_set_.items():
                if len(v) == 0:
                    classes.append([""] * len(predictions))
                else:
                    loc = np.array([self.model.predictor_encoder[item] for item in v])
                    classes.append(
                        [
                            predictor_decoder[name]
                            for name in loc[np.argmax(predictions[:, loc], axis=1)]
                        ]
                    )
            # TODO: need to be careful when adding new predictions or conditions
            classes = pd.DataFrame(
                data=np.array(classes).T,
                columns=[i + "_pred" for i in self.predictor_set_.keys()],
                index=adata.obs.index,
            )
        else:
            classes = pd.DataFrame(index=adata.obs.index)

        res = AnnData(
            np.array(torch.cat(latents)),
            obs=pd.concat(
                [
                    adata.obs[self.condition_keys_ + self.predictor_keys_],
                    classes,
                ],
                axis=1,
            ),
        )
        if get_fullpred and add_classpred:
            return res, predictions
        else:
            return res, None

    def reconstruct(
        self,
        latent: np.ndarray,
        c: np.ndarray,
    ):
        """Map `x` in to the latent space. This function will feed data in encoder  and return  z for each sample in
        data.

        Parameters
        ----------
        adata: AnnData

        Returns
        -------
             Returns array containing latent space encoding of 'x'.
        """
        device = next(self.model.parameters()).device
        conditions = label_encoder_2D(
            adata=AnnData(latent, c),
            encoder=self.model.condition_encoder,
            label_sets=self.condition_set_,
        )

        c = torch.tensor(conditions, dtype=torch.long)
        latent = torch.tensor(latent)

        expressions = []
        indices = torch.arange(latent.size(0))
        subsampled_indices = indices.split(512)
        for batch in subsampled_indices:
            expression, _ = self.model.reconstructLatent(
                latent[batch, :].to(device), c[batch]
            )
            expressions += [expression.cpu().detach()]
        return pd.DataFrame(
            np.array(torch.cat(expressions)), columns=self.adata.var.index
        )

    @classmethod
    def load_query_data(
        cls,
        adata: AnnData,
        reference_model: Union[str, "CELLIGNER2"],
        freeze: bool = True,
        freeze_expression: bool = True,
        unfreeze_new: bool = True,
        remove_dropout: bool = True,
        new_unconstrained: Optional[int] = None,
        # new_constrained: Optional[int] = None,
        new_mask: Optional[Union[np.ndarray, list]] = None,
        **kwargs
    ):
        """Transfer Learning function for new data. Uses old trained model and expands it for new conditions.
        Parameters
        ----------
        adata
             Query anndata object.
        reference_model
             A model to expand or a path to a model folder.
        freeze: Boolean
             If 'True' freezes every part of the network except the first layers of encoder/decoder.
        freeze_expression: Boolean
             If 'True' freeze every weight in first layers except the condition weights.
        remove_dropout: Boolean
             If 'True' remove Dropout for Transfer Learning.
        unfreeze_new: Boolean
             If 'True' do not freeze weights for new nodes.
        new_nodes: Integer
             Number of new nodes to add to the reference model.
             Used for query mapping.
        new_unconstrained: Integer
             Number of new constrained extension terms to add to the reference model.
             Used for query mapping.
        new_constrained: Integer
             Number of new unconstrained extension terms to add to the reference model.
             Used for query mapping.
        new_ext_mask: Array or List
             Mask (similar to the mask argument) for new unconstrained extension terms.
        kwargs
             kwargs for the initialization of the EXPIMAP class for the query model.
        Returns
        -------
        new_model
             New (query) model to train on query data.
        """
        params = {}
        params["adata"] = adata
        params["reference_model"] = reference_model
        params["freeze"] = freeze
        params["freeze_expression"] = freeze_expression
        params["remove_dropout"] = remove_dropout

        if new_unconstrained is not None:
            params["new_unconstrained"] = new_unconstrained
        if new_mask is not None:
            params["new_mask"] = new_mask

        params.update(kwargs)

        new_model = super().load_query_data(**params)

        if freeze and unfreeze_new:
            for name, p in new_model.model.named_parameters():
                if "ext_L.weight" in name or "ext_L_m.weight" in name:
                    p.requires_grad = True
                if "expand_mean_encoder" in name or "expand_var_encoder" in name:
                    print(name)
                    p.requires_grad = True

        return new_model

    # expimap

    def nonzero_terms(self):
        """Return indices of active terms.
        Active terms are the terms which were not deactivated by the group lasso regularization.
        """
        return self.model.decoder.nonzero_terms()

    def update_terms(self, terms: Union[str, list] = "terms", adata=None):
        """Add extension terms' names to the terms."""
        if isinstance(terms, str):
            adata = self.adata if adata is None else adata
            key = terms
            terms = list(adata.uns[terms])
        else:
            adata = None
            key = None
            terms = list(terms)

        lat_mask_dim = self.latent_dim_ + self.n_ext_m_
        if len(terms) != self.latent_dim_ and len(terms) != lat_mask_dim + self.n_ext_:
            raise ValueError(
                "The list of terms should have the same length as the mask."
            )

        if len(terms) == self.latent_dim_:
            if self.n_ext_m_ > 0:
                terms += ["constrained_" + str(i) for i in range(self.n_ext_m_)]
            if self.n_ext_ > 0:
                terms += ["unconstrained_" + str(i) for i in range(self.n_ext_)]

        if adata is not None:
            adata.uns[key] = terms
        else:
            return terms

    def term_genes(self, term: Union[str, int], terms: Union[str, list] = "terms"):
        """Return the dataframe with genes belonging to the term after training sorted by absolute weights in the decoder."""
        if isinstance(terms, str):
            terms = list(self.adata.uns[terms])
        else:
            terms = list(terms)

        if len(terms) == self.latent_dim_:
            if self.n_ext_m_ > 0:
                terms += ["constrained_" + str(i) for i in range(self.n_ext_m_)]
            if self.n_ext_ > 0:
                terms += ["unconstrained_" + str(i) for i in range(self.n_ext_)]

        lat_mask_dim = self.latent_dim_ + self.n_ext_m_

        if len(terms) != self.latent_dim_ and len(terms) != lat_mask_dim + self.n_ext_:
            raise ValueError(
                "The list of terms should have the same length as the mask."
            )

        term = terms.index(term) if isinstance(term, str) else term

        if term < self.latent_dim_:
            weights = self.model.decoder.L0.expr_L.weight[:, term].data.cpu().numpy()
            mask_idx = self.mask_[term]
        elif term >= lat_mask_dim:
            term -= lat_mask_dim
            weights = self.model.decoder.L0.ext_L.weight[:, term].data.cpu().numpy()
            mask_idx = None
        else:
            term -= self.latent_dim_
            weights = self.model.decoder.L0.ext_L_m.weight[:, term].data.cpu().numpy()
            mask_idx = self.ext_mask_[term]

        abs_weights = np.abs(weights)
        srt_idx = np.argsort(abs_weights)[::-1][: (abs_weights > 0).sum()]

        result = pd.DataFrame()
        result["genes"] = self.adata.var_names[srt_idx].tolist()
        result["weights"] = weights[srt_idx]
        result["in_mask"] = False

        if mask_idx is not None:
            in_mask = np.isin(srt_idx, np.where(mask_idx)[0])
            result["in_mask"][in_mask] = True

        return result

    def mask_genes(self, terms: Union[str, list] = "terms"):
        """Return lists of genes belonging to the terms in the mask."""
        if isinstance(terms, str):
            terms = list(self.adata.uns[terms])
        else:
            terms = list(terms)

        I = np.array(self.mask_)

        if self.n_ext_m_ > 0:
            I = np.concatenate((I, self.ext_mask_))

            if len(terms) == self.latent_dim_:
                terms += ["constrained_" + str(i) for i in range(self.n_ext_m_)]
            elif len(terms) == self.latent_dim_ + self.n_ext_m_ + self.n_ext_:
                terms = terms[: (self.latent_dim_ + self.n_ext_m_)]
            else:
                raise ValueError(
                    "The list of terms should have the same length as the mask."
                )

        I = I.astype(bool)

        return {
            term: self.adata.var_names[I[i]].tolist() for i, term in enumerate(terms)
        }

    # TODO: move to eval folder
    def latent_directions(
        self, method="sum", get_confidence=False, adata=None, key_added="directions"
    ):
        """Get directions of upregulation for each latent dimension.
        Multipling this by raw latent scores ensures positive latent scores correspond to upregulation.
        Parameters
        ----------
        method: String
             Method of calculation, it should be 'sum' or 'counts'.
        get_confidence: Boolean
             Only for method='counts'. If 'True', also calculate confidence
             of the directions.
        adata: AnnData
             An AnnData object to store dimensions. If 'None', self.adata is used.
        key_added: String
             key of adata.uns where to put the dimensions.
        """
        if adata is None:
            adata = self.adata

        terms_weights = self.model.decoder.L0.expr_L.weight.data
        if self.n_ext_m_ > 0:
            terms_weights = torch.cat(
                [terms_weights, self.model.decoder.L0.ext_L_m.weight.data], dim=1
            )
        if self.n_ext_ > 0:
            terms_weights = torch.cat(
                [terms_weights, self.model.decoder.L0.ext_L.weight.data], dim=1
            )

        if method == "sum":
            signs = terms_weights.sum(0).cpu().numpy()
            signs[signs > 0] = 1.0
            signs[signs < 0] = -1.0
            confidence = None
        elif method == "counts":
            num_nz = torch.count_nonzero(terms_weights, dim=0)
            upreg_genes = torch.count_nonzero(terms_weights > 0, dim=0)
            signs = upreg_genes / (num_nz + (num_nz == 0))
            signs = signs.cpu().numpy()

            confidence = signs.copy()
            confidence = np.abs(confidence - 0.5) / 0.5
            confidence[num_nz == 0] = 0

            signs[signs > 0.5] = 1.0
            signs[signs < 0.5] = -1.0

            signs[signs == 0.5] = 0
            signs[num_nz == 0] = 0
        else:
            raise ValueError("Unrecognized method for getting the latent direction.")

        adata.uns[key_added] = signs
        if get_confidence and confidence is not None:
            adata.uns[key_added + "_confindence"] = confidence

    def latent_enrich(
        self,
        groups,
        comparison="rest",
        n_sample=5000,
        use_directions=False,
        directions_key="directions",
        select_terms=None,
        adata=None,
        exact=True,
        key_added="bf_scores",
    ):
        """Gene set enrichment test for the latent space. Test the hypothesis that latent scores
        for each term in one group (z_1) is bigger than in the other group (z_2).
        Puts results to `adata.uns[key_added]`. Results are a dictionary with
        `p_h0` - probability that z_1 > z_2, `p_h1 = 1-p_h0` and `bf` - bayes factors equal to `log(p_h0/p_h1)`.
        Parameters
        ----------
        groups: String or Dict
             A string with the key in `adata.obs` to look for categories or a dictionary
             with categories as keys and lists of cell names as values.
        comparison: String
             The category name to compare against. If 'rest', then compares each category against all others.
        n_sample: Integer
             Number of random samples to draw for each category.
        use_directions: Boolean
             If 'True', multiplies the latent scores by directions in `adata`.
        directions_key: String
             The key in `adata.uns` for directions.
        select_terms: Array
             If not 'None', then an index of terms to select for the test. Only does the test
             for these terms.
        adata: AnnData
             An AnnData object to use. If 'None', uses `self.adata`.
        exact: Boolean
             Use exact probabilities for comparisons.
        key_added: String
             key of adata.uns where to put the results of the test.
        """
        if adata is None:
            adata = self.adata

        if isinstance(groups, str):
            cats_col = adata.obs[groups]
            cats = cats_col.unique()
        elif isinstance(groups, dict):
            cats = []
            all_cells = []
            for group, cells in groups.items():
                cats.append(group)
                all_cells += cells
            adata = adata[all_cells]
            cats_col = pd.Series(index=adata.obs_names, dtype=str)
            for group, cells in groups.items():
                cats_col[cells] = group
        else:
            raise ValueError("groups should be a string or a dict.")

        if comparison != "rest" and set(comparison).issubset(cats):
            raise ValueError("comparison should be 'rest' or among the passed groups")

        scores = {}

        if comparison != "rest" and isinstance(comparison, str):
            comparison = [comparison]

        for cat in cats:
            if cat in comparison:
                continue

            cat_mask = cats_col == cat
            if comparison == "rest":
                others_mask = ~cat_mask
            else:
                others_mask = cats_col.isin(comparison)

            choice_1 = np.random.choice(cat_mask.sum(), n_sample)
            choice_2 = np.random.choice(others_mask.sum(), n_sample)

            adata_cat = adata[cat_mask][choice_1]
            adata_others = adata[others_mask][choice_2]

            if use_directions:
                directions = adata.uns[directions_key]
            else:
                directions = None

            z0 = self.get_latent(
                adata_cat.X,
                adata_cat.obs[self.condition_key_],
                mean=False,
                mean_var=exact,
            )
            z1 = self.get_latent(
                adata_others.X,
                adata_others.obs[self.condition_key_],
                mean=False,
                mean_var=exact,
            )

            if not exact:
                if directions is not None:
                    z0 *= directions
                    z1 *= directions

                if select_terms is not None:
                    z0 = z0[:, select_terms]
                    z1 = z1[:, select_terms]

                to_reduce = z0 > z1

                zeros_mask = (np.abs(z0).sum(0) == 0) | (np.abs(z1).sum(0) == 0)
            else:
                from scipy.special import erfc

                means0, vars0 = z0
                means1, vars1 = z1

                if directions is not None:
                    means0 *= directions
                    means1 *= directions

                if select_terms is not None:
                    means0 = means0[:, select_terms]
                    means1 = means1[:, select_terms]
                    vars0 = vars0[:, select_terms]
                    vars1 = vars1[:, select_terms]

                to_reduce = (means1 - means0) / np.sqrt(2 * (vars0 + vars1))
                to_reduce = 0.5 * erfc(to_reduce)

                zeros_mask = (np.abs(means0).sum(0) == 0) | (np.abs(means1).sum(0) == 0)

            p_h0 = np.mean(to_reduce, axis=0)
            p_h1 = 1.0 - p_h0
            epsilon = 1e-12
            bf = np.log(p_h0 + epsilon) - np.log(p_h1 + epsilon)

            p_h0[zeros_mask] = 0
            p_h1[zeros_mask] = 0
            bf[zeros_mask] = 0

            scores[cat] = dict(p_h0=p_h0, p_h1=p_h1, bf=bf)

        adata.uns[key_added] = s
