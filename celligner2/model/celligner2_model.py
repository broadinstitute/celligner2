import inspect
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
    """

    def __init__(
        self,
        adata: AnnData,
        condition_keys: list = None,
        conditions: Optional[list] = None,
        hidden_layer_sizes: list = [256, 64],
        classifier_hidden_layer_sizes: list = [64, 32],
        latent_dim: int = 10,
        dr_rate: float = 0.05,
        use_mmd: bool = True,
        mmd_on: str = "z",
        mmd_boundary: Optional[int] = None,
        recon_loss: Optional[str] = "nb",
        beta: float = 1,
        betaclass: float = 0.8,
        use_bn: bool = False,
        use_ln: bool = True,
        predictors: Optional[list] = None,
        predictor_keys: Optional[list] = None,
        use_own_kl: bool = False,
        miss: str = "U",
        apply_log: bool = True,
        mask: Optional[Union[np.ndarray, list]] = None,
        mask_key: str = "I",
        soft_mask: bool = False,
        main_dataset=None,
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
                self.condition_set_ = {
                    key: set(adata.obs[key]) - set(miss) for key in condition_keys
                }
            else:
                self.conditions_ = []
        else:
            self.conditions_ = conditions

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
                self.predictor_set_ = {
                    key: set(adata.obs[key]) - set(miss) for key in predictor_keys
                }
            else:
                self.predictors_ = []

        else:
            self.predictors_ = predictors

        self.miss_ = miss
        self.hidden_layer_sizes_ = hidden_layer_sizes
        self.classifier_hidden_layer_sizes_ = classifier_hidden_layer_sizes
        self.latent_dim_ = latent_dim
        self.dr_rate_ = dr_rate
        self.use_mmd_ = use_mmd
        self.mmd_on_ = mmd_on
        self.mmd_boundary_ = mmd_boundary

        if mask is not None:
            mask = mask if isinstance(mask, list) else mask.tolist()
            self.mask_ = torch.tensor(mask).float()
        else:
            self.mask_ = None
        self.soft_mask_ = soft_mask

        self.recon_loss_ = recon_loss
        self.beta_ = beta
        self.betaclass_ = betaclass
        self.use_bn_ = use_bn
        self.use_ln_ = use_ln
        self.use_own_kl_ = use_own_kl

        self.input_dim_ = adata.n_vars
        self.apply_log_ = apply_log
        self.main_dataset_ = main_dataset

        self.model = Celligner2(
            self.input_dim_,
            self.conditions_,
            self.predictors_,
            self.hidden_layer_sizes_,
            self.classifier_hidden_layer_sizes_,
            self.latent_dim_,
            self.dr_rate_,
            self.use_mmd_,
            self.use_own_kl_,
            self.mmd_on_,
            self.mmd_boundary_,
            self.recon_loss_,
            self.beta_,
            self.betaclass_,
            self.use_bn_,
            self.use_ln_,
            self.apply_log_,
            # self.soft_mask_,
            # self.mask_,
            self.main_dataset_,
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
            "classifier_hidden_layer_sizes": dct["classifier_hidden_layer_sizes_"],
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
            "input_dim": dct["input_dim_"],
            "apply_log": dct["apply_log_"],
            # "predictor_set": dct["predictor_set_"],
            # "condition_set": dct["condition_set_"],
            # "soft_mask": dct["soft_mask_"],
            # "mask": dct["mask_"],
            "main_dataset": dct["main_dataset_"],
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
            for _, v in self.predictor_set_.items():
                if len(v) == 0:
                    classes.append([""] * len(predictions))
                else:
                    loc = np.array([self.model.predictor_encoder[vv] for vv in v])
                    classes.append(
                        [
                            predictor_decoder[name]
                            for name in loc[np.argmax(predictions[:, loc], axis=1)]
                        ]
                    )
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
        reference_model: Union[str, "TRVAE"],
        freeze: bool = True,
        freeze_expression: bool = True,
        unfreeze_ext: bool = True,
        remove_dropout: bool = True,
        new_n_ext: Optional[int] = None,
        new_n_ext_m: Optional[int] = None,
        new_ext_mask: Optional[Union[np.ndarray, list]] = None,
        new_soft_ext_mask: bool = False,
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
        unfreeze_ext: Boolean
             If 'True' do not freeze weights for new constrained and unconstrained extension terms.
        new_n_ext: Integer
             Number of new unconstarined extension terms to add to the reference model.
             Used for query mapping.
        new_n_ext_m: Integer
             Number of new constrained extension terms to add to the reference model.
             Used for query mapping.
        new_ext_mask: Array or List
             Mask (similar to the mask argument) for new unconstarined extension terms.
        new_soft_ext_mask: Boolean
             Use the soft mask mode for training with the constarined extension terms.
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

        if new_n_ext is not None:
            params["n_ext"] = new_n_ext
        if new_n_ext_m is not None:
            params["n_ext_m"] = new_n_ext_m
            if new_ext_mask is None:
                raise ValueError("Provide new ext_mask")
            params["ext_mask"] = new_ext_mask
            params["soft_ext_mask"] = new_soft_ext_mask

        params.update(kwargs)

        new_model = super().load_query_data(**params)

        if freeze and unfreeze_ext:
            for name, p in new_model.model.named_parameters():
                if "ext_L.weight" in name or "ext_L_m.weight" in name:
                    p.requires_grad = True
                if "expand_mean_encoder" in name or "expand_var_encoder" in name:
                    p.requires_grad = True

        return new_model
