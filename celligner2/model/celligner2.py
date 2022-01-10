from typing import Optional

import torch
import torch.nn as nn
from torch.distributions import Normal, kl_divergence
import torch.nn.functional as F

from .modules import Encoder, Decoder, Classifier
from .losses import mse, mmd, zinb, nb, classifier_er, classifier_loss1, classifier_hb_loss
from ._utils import one_hot_encoder
from celligner2.othermodels.base._base import CVAELatentsModelMixin


class Celligner2(nn.Module, CVAELatentsModelMixin):
    """ScArches model class. This class contains the implementation of Conditional Variational Auto-encoder.

        Parameters
        ----------
        input_dim: Integer
            Number of input features (i.e. gene in case of scRNA-seq).
        conditions: List
            List of Condition names that the used data will contain to get the right encoding when used after reloading.
        hidden_layer_sizes: List
            A list of hidden layer sizes for encoder network. Decoder network will be the reversed order.
        latent_dim: Integer
            Bottleneck layer (z)  size.
        dr_rate: Float
            Dropout rate applied to all layers, if `dr_rate`==0 no dropout will be applied.
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
            Scaling Factor for MMD loss. Higher beta values result in stronger batch-correction at a cost of worse biological variation.
        use_bn: Boolean
            If `True` batch normalization will be applied to layers.
        use_ln: Boolean
            If `True` layer normalization will be applied to layers.
        use_own_kl: Boolean
            If `True` the KL-Divergence will be calculated by the network itself. Otherwise torch
    """

    def __init__(self,
                 input_dim: int,
                 conditions: list,
                 predictors: list,
                 hidden_layer_sizes: list = [256, 64],
                 classifier_hidden_layer_sizes: list = [32],
                 latent_dim: int = 10,
                 dr_rate: float = 0.05,
                 use_mmd: bool = False,
                 use_own_kl: bool = False,
                 mmd_on: str = 'z',
                 mmd_boundary: Optional[int] = None,
                 recon_loss: Optional[str] = 'nb',
                 beta: float = 1,
                 betaclass: float =0.8,
                 use_bn: bool = False,
                 use_ln: bool = True,
                 ):
        super().__init__()
        assert isinstance(hidden_layer_sizes, list)
        assert isinstance(latent_dim, int)
        assert isinstance(conditions, list)
        assert recon_loss in ["mse", "nb", "zinb"], "'recon_loss' must be 'mse', 'nb' or 'zinb'"

        print("\nINITIALIZING NEW NETWORK..............")
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.n_predictors = len(predictors)
        self.n_conditions = len(conditions)
        self.use_own_kl = use_own_kl
        self.conditions = conditions
        self.predictors = predictors
        self.condition_encoder = {k: v for k, v in zip(conditions, range(len(conditions)))}
        self.predictor_encoder = {k: v for k, v in zip(predictors, range(len(predictors)))}
        
        self.cell_type_encoder = None
        self.recon_loss = recon_loss
        self.mmd_boundary = mmd_boundary
        self.use_mmd = use_mmd
        self.freeze = False
        self.beta = beta
        self.betaclass = betaclass
        self.use_bn = use_bn
        self.use_ln = use_ln
        self.mmd_on = mmd_on

        self.dr_rate = dr_rate
        if self.dr_rate > 0:
            self.use_dr = True
        else:
            self.use_dr = False

        if recon_loss in ["nb", "zinb"]:
            self.theta = torch.nn.Parameter(torch.randn(self.input_dim, self.n_conditions))
        else:
            self.theta = None

        self.hidden_layer_sizes = hidden_layer_sizes
        encoder_layer_sizes = self.hidden_layer_sizes.copy()
        encoder_layer_sizes.insert(0, self.input_dim)
        decoder_layer_sizes = self.hidden_layer_sizes.copy()
        decoder_layer_sizes.reverse()
        decoder_layer_sizes.append(self.input_dim)
        self.classifier_hidden_layer_sizes = classifier_hidden_layer_sizes
        self.encoder = Encoder(encoder_layer_sizes,
                               self.latent_dim,
                               self.use_bn,
                               self.use_ln,
                               self.use_dr,
                               self.dr_rate,
                               self.n_conditions)
        self.classifier = Classifier(self.classifier_hidden_layer_sizes,
                                    self.latent_dim,
                                    self.dr_rate,
                                    self.use_dr,
                                    self.use_bn,
                                    self.use_ln,
                                    self.n_predictors
                                    )
        self.decoder = Decoder(decoder_layer_sizes,
                               self.latent_dim,
                               self.recon_loss,
                               self.use_bn,
                               self.use_ln,
                               self.use_dr,
                               self.dr_rate,
                               self.n_conditions)

    def forward(self, x=None, batch=None, sizefactor=None, classes=None, labeled=None, weight=None, celltypes=None):
        x_log = torch.log(1 + x)
        if self.recon_loss == 'mse':
            x_log = x

        z1_mean, z1_log_var = self.encoder(x_log, batch)
        #print(z1_mean.mean(), z1_log_var.mean())
        z1 = self.sampling(z1_mean, z1_log_var)
        outputs = self.decoder(z1, batch)
        pred_classes = self.classifier(z1_mean, batch)

        if self.recon_loss == "mse":
            recon_x, y1 = outputs
            recon_loss = mse(recon_x, x_log).sum(dim=-1).mean()
        elif self.recon_loss == "zinb":
            dec_mean_gamma, dec_dropout, y1 = outputs
            size_factor_view = sizefactor.unsqueeze(1).expand(dec_mean_gamma.size(0), dec_mean_gamma.size(1))
            dec_mean = dec_mean_gamma * size_factor_view
            dispersion = F.linear(one_hot_encoder(batch, self.n_conditions), self.theta)
            dispersion = torch.exp(dispersion)
            recon_loss = -zinb(x=x, mu=dec_mean, theta=dispersion, pi=dec_dropout).sum(dim=-1).mean()
        elif self.recon_loss == "nb":
            dec_mean_gamma, y1 = outputs
            size_factor_view = sizefactor.unsqueeze(1).expand(dec_mean_gamma.size(0), dec_mean_gamma.size(1))
            dec_mean = dec_mean_gamma * size_factor_view
            dispersion = F.linear(one_hot_encoder(batch, self.n_conditions), self.theta)
            dispersion = torch.exp(dispersion)
            recon_loss = -nb(x=x, mu=dec_mean, theta=dispersion).sum(dim=-1).mean()

        z1_var = torch.exp(z1_log_var) + 1e-4
        if self.use_own_kl:
            kl_div = -0.5 * torch.sum(1 + z1_log_var - z1_mean ** 2 - z1_var)
        else:
            kl_div = kl_divergence(
                Normal(z1_mean, torch.sqrt(z1_var)),
                Normal(torch.zeros_like(z1_mean), torch.ones_like(z1_var))
            ).sum(dim=1).mean()

        mmd_loss = torch.tensor(0.0, device=z1.device)

        if self.use_mmd:
            batch = torch.argmax(batch, dim=1)
            if self.mmd_on == "z":
                mmd_loss = mmd(z1, batch, self.n_conditions, self.beta, self.mmd_boundary)
            else:
                mmd_loss = mmd(y1, batch, self.n_conditions, self.beta, self.mmd_boundary)

        class_ce_loss = classifier_hb_loss(pred_classes, classes, beta=self.betaclass, weight=weight)
        return recon_loss, kl_div, mmd_loss, class_ce_loss


    def get_latent(self, x, c=None, mean=False, add_classpred=False):
        """Map `x` in to the latent space. This function will feed data in encoder  and return  z for each sample in
           data.

           Parameters
           ----------
           x:  torch.Tensor
                Torch Tensor to be mapped to latent space. `x` has to be in shape [n_obs, input_dim].
           c: torch.Tensor
                Torch Tensor of condition labels for each sample.
           mean: boolean

           Returns
           -------
           Returns Torch Tensor containing latent space encoding of 'x'.
        """
        x_ = torch.log(1 + x)
        if self.recon_loss == 'mse':
            x_ = x
        z_mean, z_log_var = self.encoder(x_, c)
        latent = self.sampling(z_mean, z_log_var)
        if add_classpred:
            classes = self.classifier(z_mean)
        if mean:
            return z_mean if not add_classpred else (z_mean, classes)
        return latent if not add_classpred else (latent, classes)