from typing import Optional, Union

import torch
import torch.nn as nn
from torch.distributions import Normal, kl_divergence
import torch.nn.functional as F
import numpy as np
from captum.attr import LRP
from .modules import Encoder, Decoder, Classifier, MaskedLinearDecoder  # , ExtEncoder
from .losses import mse, mmd, zinb, nb, classifier_hb_loss, hsic
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

    def __init__(
        self,
        input_dim: int,
        hidden_layer_sizes: list = [256, 64],
        latent_dim: int = 10,
        dr_rate: float = 0.05,
        use_own_kl: bool = False,
        recon_loss: Optional[str] = "nb",
        use_bn: bool = False,
        use_ln: bool = True,
        applylog: bool = True,
        # conditional part
        conditions: list = [],
        use_mmd: bool = False,
        mmd_on: str = "z",
        mmd_boundary: Optional[int] = None,
        beta: float = 1,
        main_dataset: str = None,
        batch_knowledge: bool = True,
        # classifier part
        predictors: list = [],
        classifier_hidden_layer_sizes: list = [32],
        betaclass: float = 0.8,
        # GNN part
        graph_layers: int = 0,
        res_mult: int = 0,
        # expimap part
        n_expand: int = 0,
        expimap_mode: bool = None,
        mask: Optional[Union[np.ndarray, list]] = None,
        ext_mask: Optional[Union[np.ndarray, list]] = None,
        n_unconstrained: int = 0,
        ext_n_unconstrained: int = 0,
        use_l_encoder: bool = False,
        use_hsic: bool = False,
        hsic_one_vs_all: bool = False,
    ):
        super().__init__()
        assert isinstance(hidden_layer_sizes, list)
        assert isinstance(latent_dim, int)
        assert isinstance(conditions, list)
        assert recon_loss in [
            "mse",
            "nb",
            "zinb",
        ], "'recon_loss' must be 'mse', 'nb' or 'zinb'"

        print("\nINITIALIZING NEW NETWORK..............")
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.n_predictors = len(predictors)
        self.n_conditions = len(conditions)
        self.use_own_kl = use_own_kl
        self.conditions = conditions
        self.predictors = predictors
        self.applylog = applylog

        self.condition_encoder = {
            k: v for k, v in zip(conditions, range(len(conditions)))
        }
        self.predictor_encoder = {
            k: v for k, v in zip(predictors, range(len(predictors)))
        }

        self.use_hsic = use_hsic and self.n_ext_decoder > 0
        self.hsic_one_vs_all = hsic_one_vs_all
        self.n_unconstrained = n_unconstrained

        self.use_l_encoder = use_l_encoder

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
        self.main_dataset = main_dataset

        self.batch_knowledge = batch_knowledge

        self.res_mult = res_mult
        self.graph_layers = graph_layers

        self.dr_rate = dr_rate
        if self.dr_rate > 0:
            self.use_dr = True
        else:
            self.use_dr = False

        if recon_loss in ["nb", "zinb"]:
            self.theta = torch.nn.Parameter(
                torch.randn(self.input_dim, self.n_conditions)
            )
        else:
            self.theta = None

        self.hidden_layer_sizes = hidden_layer_sizes
        encoder_layer_sizes = self.hidden_layer_sizes.copy()
        encoder_layer_sizes.insert(0, self.input_dim)
        decoder_layer_sizes = self.hidden_layer_sizes.copy()
        decoder_layer_sizes.reverse()
        decoder_layer_sizes.append(self.input_dim)
        self.classifier_hidden_layer_sizes = classifier_hidden_layer_sizes

        if mask is not None:
            self.n_inact_genes = (1 - mask).sum().item()
            soft_shape = mask.shape
            if soft_shape[0] != latent_dim or soft_shape[1] != input_dim:
                raise ValueError("Incorrect shape of the soft mask.")
            self.mask = mask.t()
            mask = None
        else:
            self.mask = None

        if ext_mask is not None:
            self.n_inact_ext_genes = (1 - ext_mask).sum().item()
            ext_shape = ext_mask.shape
            if ext_shape[0] != self.n_ext_m_decoder:
                raise ValueError(
                    "Dim 0 of ext_mask should be the same as n_ext_m_decoder."
                )
            if ext_shape[1] != self.input_dim:
                raise ValueError("Dim 1 of ext_mask should be the same as input_dim.")
            self.ext_mask = ext_mask.t()
            ext_mask = None
        else:
            self.ext_mask = None

        self.n_expand = n_expand

        if self.use_l_encoder:
            self.l_encoder = Encoder(
                [self.input_dim, 128],
                1,
                self.use_bn,
                self.use_ln,
                self.use_dr,
                self.dr_rate,
                self.n_conditions if self.batch_knowledge else 0,
                self.n_expand,
            )
        self.encoder = Encoder(
            encoder_layer_sizes,
            self.latent_dim,
            self.use_bn,
            self.use_ln,
            self.use_dr,
            self.dr_rate,
            self.n_conditions if self.batch_knowledge else 0,
            self.n_expand,
        )
        self.classifier = Classifier(
            self.classifier_hidden_layer_sizes,
            self.latent_dim,
            self.dr_rate,
            self.use_bn,
            self.use_ln,
            self.use_dr,
            self.n_predictors,
            self.n_expand,
        )
        if expimap_mode:
            print("expimap mode")
            self.decoder = MaskedLinearDecoder(
                self.latent_dim,
                self.input_dim,
                self.n_conditions,
                mask,
                self.recon_loss,
                self.n_unconstrained,
                self.n_expand,
            )

        else:
            self.decoder = Decoder(
                decoder_layer_sizes,
                self.latent_dim,
                self.recon_loss,
                self.use_bn,
                self.use_ln,
                self.use_dr,
                self.dr_rate,
                self.n_conditions if self.batch_knowledge else 0,
                self.n_expand,
            )

    def forward(
        self,
        x: torch.Tensor = None,
        batch: torch.Tensor = None,
        sizefactor: torch.Tensor = None,
        classes: torch.Tensor = None,
        weight: torch.Tensor = None,
        goodloc: torch.Tensor = None,
        main_dataset: str = None,
    ):
        if self.applylog:
            x_log = torch.log(1 + x)
            if self.recon_loss == "mse":
                x_log = x
        else:
            x_log = x

        z1_mean, z1_log_var = self.encoder(x_log, batch)
        # print(z1_mean.mean(), z1_log_var.mean())
        z1 = self.sampling(z1_mean, z1_log_var)
        outputs = self.decoder(z1, batch)
        if classes is not None:
            pred_classes = self.classifier(z1)

        if self.recon_loss == "mse":
            recon_x, y1 = outputs
            recon_loss = mse(recon_x, x_log).sum(dim=-1, where=goodloc).mean()
        elif self.recon_loss == "zinb":
            dec_mean_gamma, dec_dropout, y1 = outputs
            size_factor_view = sizefactor.unsqueeze(1).expand(
                dec_mean_gamma.size(0), dec_mean_gamma.size(1)
            )
            dec_mean = dec_mean_gamma * size_factor_view
            dispersion = F.linear(one_hot_encoder(batch, self.n_conditions), self.theta)
            dispersion = torch.exp(dispersion)
            recon_loss = zinb(x=x, mu=dec_mean, theta=dispersion, pi=dec_dropout)
            recon_loss[~goodloc] = 0
            recon_loss = -recon_loss.sum(dim=-1).mean()

        elif self.recon_loss == "nb":
            dec_mean_gamma, y1 = outputs
            size_factor_view = sizefactor.unsqueeze(1).expand(
                dec_mean_gamma.size(0), dec_mean_gamma.size(1)
            )
            dec_mean = dec_mean_gamma * size_factor_view
            dispersion = torch.exp(
                F.linear(one_hot_encoder(batch, self.n_conditions), self.theta)
            )
            recon_loss = nb(x=x, mu=dec_mean, theta=dispersion)
            recon_loss[~goodloc] = 0
            recon_loss = -recon_loss.sum(dim=-1).mean()

        z1_var = torch.exp(z1_log_var) + 1e-4
        if self.use_own_kl:
            kl_div = -0.5 * torch.sum(1 + z1_log_var - z1_mean**2 - z1_var)
        else:
            kl_div = (
                kl_divergence(
                    Normal(z1_mean, torch.sqrt(z1_var)),
                    Normal(torch.zeros_like(z1_mean), torch.ones_like(z1_var)),
                )
                .sum(dim=1)
                .mean()
            )

        mmd_loss = torch.tensor(0.0, device=z1.device)
        # this is a debugger line
        if self.use_mmd:
            if self.mmd_on == "z":
                mmd_loss = mmd(
                    z1,
                    batch,
                    self.n_conditions,
                    self.beta,
                    self.mmd_boundary,
                    self.condition_encoder[self.main_dataset]
                    if self.main_dataset
                    else None,
                )
            else:
                mmd_loss = mmd(
                    y1,
                    batch,
                    self.n_conditions,
                    self.beta,
                    self.mmd_boundary,
                    self.condition_encoder[self.main_dataset]
                    if self.main_dataset
                    else None,
                )

        class_ce_loss = (
            classifier_hb_loss(
                pred_classes, classes, beta=self.betaclass, weight=weight
            )
            if classes is not None
            else torch.tensor(0.0, device=kl_div.device)
        )
        if self.use_hsic:
            if not self.hsic_one_vs_all:
                z_ann = z1[:, : -self.n_ext_decoder]
                z_ext = z1[:, -self.n_ext_decoder :]
                hsic_loss = hsic(z_ann, z_ext)
            else:
                hsic_loss = 0.0
                sz = self.latent_dim + self.n_ext_encoder
                shift = self.latent_dim + self.n_ext_m_decoder
                for i in range(self.n_ext_decoder):
                    sel_cols = torch.full((sz,), True, device=z1.device)
                    sel_cols[shift + i] = False
                    rest = z1[:, sel_cols]
                    term = z1[:, ~sel_cols]
                    hsic_loss = hsic_loss + hsic(term, rest)
        else:
            hsic_loss = torch.tensor(0.0, device=z1.device)
        return recon_loss, kl_div, mmd_loss, class_ce_loss

    def get_latent(
        self,
        x: torch.Tensor,
        c: torch.Tensor = None,
        mean: bool = False,
        add_classpred: bool = False,
    ):
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
        if self.applylog:
            x = torch.log(1 + x)

        z_mean, z_log_var = self.encoder(x, c)

        latent = self.sampling(z_mean, z_log_var)
        if add_classpred:
            classes = self.classifier(z_mean)

        if mean:
            return (z_mean, None) if not add_classpred else (z_mean, classes)
        return (latent, None) if not add_classpred else (latent, classes)

    def reconstructLatent(self, latent: np.array, c=None):
        """reconstructLatent recontruct the expression matrix from latent space.

        Args:
            latent (np.array): latent space encoding of data

        Returns:
            torch.Tensor: reconstructed data
        """
        return self.decoder(latent, c)
