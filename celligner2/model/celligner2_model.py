import inspect
import os

from numpy.lib.function_base import _percentile_dispatcher
import torch
import pickle
import numpy as np
import pandas as pd

from anndata import AnnData
from typing import Optional

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
        mmd_on: str = 'z',
        mmd_boundary: Optional[int] = None,
        recon_loss: Optional[str] = 'nb',
        beta: float = 1,
        betaclass: float = 0.8,
        use_bn: bool = False,
        use_ln: bool = True,
        predictors: Optional[list] = None,
        predictor_keys: Optional[list] = None,
        use_own_kl: bool = False,
        miss: str = 'U',
    ):
        self.adata = adata

        self.condition_keys_ = condition_keys
        self.predictor_keys_ = predictor_keys

        if conditions is None:
            if condition_keys is not None:
                self.conditions_ = list(
                    set(adata.obs[condition_keys].values.flatten())-set(miss))
                if len(self.conditions_) != sum(
                    [len(set(adata.obs[condition_key])-set(miss)) for condition_key in condition_keys]):
                    raise ValueError('conditions need to be unique even amongst \
                        different columns')
            else:
                self.conditions_ = []
        else:
            self.conditions_ = conditions
        
        if predictors is None:
            if predictor_keys is not None:
                self.predictors_ = list(
                    set(adata.obs[predictor_keys].values.flatten()) - set(miss))
                if len(self.predictors_) != sum(
                    [len(set(adata.obs[predictor_key]) - set(miss)) for predictor_key in predictor_keys]):
                    raise ValueError('Predictors need to be unique even amongst \
                        different columns')
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

        self.recon_loss_ = recon_loss
        self.beta_ = beta
        self.betaclass_ = betaclass
        self.use_bn_ = use_bn
        self.use_ln_ = use_ln
        self.use_own_kl_ = use_own_kl

        self.input_dim_ = adata.n_vars

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
        )

        self.is_trained_ = False
        self.trainer = None

    def train(
        self,
        n_epochs: int = 400,
        lr: float = 1e-3,
        eps: float = 0.01,
        **kwargs
    ):
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
            **kwargs)
        self.trainer.train(n_epochs, lr, eps)
        self.is_trained_ = True

    @classmethod
    def _get_init_params_from_dict(cls, dct):
        init_params = {
            'condition_keys': dct['condition_keys_'],
            'conditions': dct['conditions_'],
            'hidden_layer_sizes': dct['hidden_layer_sizes_'],
            'latent_dim': dct['latent_dim_'],
            'dr_rate': dct['dr_rate_'],
            'use_mmd': dct['use_mmd_'],
            'mmd_on': dct['mmd_on_'],
            'mmd_boundary': dct['mmd_boundary_'],
            'recon_loss': dct['recon_loss_'],
            'beta': dct['beta_'],
            'betaclass': dct['betaclass_'],
            'use_bn': dct['use_bn_'],
            'use_ln': dct['use_ln_'],
            'use_own_kl': dct['use_own_kl_'],
            'predictors': dct['predictors_'],
            'predictor_keys': dct['predictor_keys_'],
            'miss': dct['miss_'],
        }

        return init_params

    @classmethod
    def _validate_adata(cls, adata, dct):
        if adata.n_vars != dct['input_dim_']:
            raise ValueError("Incorrect var dimension")

        adata_conditions = adata.obs[dct['condition_keys_']].unique().tolist()
        if not set(adata_conditions).issubset(dct['conditions_']):
            raise ValueError("Incorrect conditions")

    def get_latent(
        self,
        adata: Optional[AnnData] = None,
        mean: bool = False,
        add_classpred: bool = False,
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
            wasnull=True
            adata = self.adata
        import pdb; pdb.set_trace()
        condition_sets = {key: set(adata.obs[key]) for key in self.condition_keys_}
        conditions = label_encoder_2D(
            adata,
            encoder=self.model.condition_encoder,
            label_sets=condition_sets,
        )
        
        c = torch.tensor(conditions, dtype=torch.long)
        x = torch.tensor(adata.X)

        latents = []
        classes = []
        indices = torch.arange(x.size(0))
        subsampled_indices = indices.split(512)
        for batch in subsampled_indices:
            latent, classe = self.model.get_latent(x[batch,:].to(device), c[batch], mean, add_classpred)
            latents += [latent.cpu().detach()]
            if add_classpred:
                classes += [classe.cpu().detach()]

        if add_classpred:
            #import pdb; pdb.set_trace()
            predictor_set = {key: set(self.adata.obs[key])-set(self.miss_) for key in self.predictor_keys_}
            predictor_decoder = {v:k for k,v in self.model.predictor_encoder.items()}
            classes = np.array(torch.cat(classes))
            nclasses = []
            for _, v in predictor_set.items():
                loc = np.array([self.model.predictor_encoder[vv] for vv in v])
                nclasses.append([predictor_decoder[name] for name in loc[np.argmax(
                    classes[:, loc],
                axis=1)]])
            classes = pd.DataFrame(data=np.array(nclasses).T, columns=[i+'_pred' for i in predictor_set.keys()], index=self.adata.obs.index)
        else:
            classes = pd.DataFrame(index=self.adata.obs.index)
        return AnnData(np.array(torch.cat(latents)), obs=pd.concat([self.adata.obs, classes], axis=1)) if wasnull else np.array(torch.cat(latents))