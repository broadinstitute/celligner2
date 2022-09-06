import torch
import torch.nn as nn
import numpy as np
from dgl.nn import GraphConv

from typing import Optional, Union
from ._utils import one_hot_encoder


class mySequential(nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs


class MaskedLinear(nn.Linear):
    def __init__(self, n_in, n_out, mask, bias=True):
        # mask should have the same dimensions as the transposed linear weight
        # n_input x n_output_nodes
        if n_in != mask.shape[0] or n_out != mask.shape[1]:
            raise ValueError("Incorrect shape of the mask.")

        super().__init__(n_in, n_out, bias)

        self.register_buffer("mask", mask.t())

        # zero out the weights for group lasso
        # gradient descent won't change these zero weights
        self.weight.data *= self.mask

    def forward(self, input):
        return nn.functional.linear(input, self.weight * self.mask, self.bias)


class MaskedCondLayers(nn.Module):
    def __init__(
        self,
        n_in: int,
        n_out: int,
        n_cond: int,
        bias: bool,
        n_ext: int = 0,
        n_ext_m: int = 0,
        graph: bool = False,
        mask: Optional[torch.Tensor] = None,
        ext_mask: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.n_cond = n_cond
        self.n_ext = n_ext
        self.n_ext_m = n_ext_m
        self.graph = graph

        if mask is not None:
            self.expr_L = MaskedLinear(n_in, n_out, mask, bias=bias)
        elif graph:
            self.expr_L = GraphConv(n_in, n_out)
        else:
            self.expr_L = nn.Linear(n_in, n_out, bias=bias)

        if self.n_cond != 0:
            self.cond_L = nn.Linear(self.n_cond, n_out, bias=False)

        if self.n_ext != 0:
            self.ext_L = nn.Linear(self.n_ext, n_out, bias=False)

        if self.n_ext_m != 0:
            if ext_mask is not None:
                self.ext_L_m = MaskedLinear(self.n_ext_m, n_out, ext_mask, bias=False)
            else:
                self.ext_L_m = nn.Linear(self.n_ext_m, n_out, bias=False)

    def forward(self, x: torch.Tensor, g: Optional[torch.Tensor] = None):
        if self.n_cond == 0:
            expr, cond = x, None
        else:
            expr, cond = torch.split(x, [x.shape[1] - self.n_cond, self.n_cond], dim=1)

        if self.n_ext == 0:
            ext = None
        else:
            expr, ext = torch.split(
                expr, [expr.shape[1] - self.n_ext, self.n_ext], dim=1
            )

        if self.n_ext_m == 0:
            ext_m = None
        else:
            expr, ext_m = torch.split(
                expr, [expr.shape[1] - self.n_ext_m, self.n_ext_m], dim=1
            )

        if self.graph:
            out = self.expr_L(expr, g)
        else:
            out = self.expr_L(expr)
        if ext is not None:
            out = out + self.ext_L(ext)
        if ext_m is not None:
            out = out + self.ext_L_m(ext_m)
        if cond is not None:
            out = out + self.cond_L(cond)
        return out


class CondLayers(nn.Module):
    def __init__(
        self,
        n_in: int,
        n_out: int,
        n_cond: int,
        bias: bool,
    ):
        super().__init__()
        self.n_cond = n_cond
        self.expr_L = nn.Linear(n_in, n_out, bias=bias)
        if self.n_cond != 0:
            self.cond_L = nn.Linear(self.n_cond, n_out, bias=False)

    def forward(self, x: torch.Tensor):
        if self.n_cond == 0:
            out = self.expr_L(x)
        else:
            expr, cond = torch.split(x, [x.shape[1] - self.n_cond, self.n_cond], dim=1)
            out = self.expr_L(expr) + self.cond_L(cond)
        return out


class MaskedLinearDecoder(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        n_cond,
        mask,
        ext_mask,
        recon_loss,
        last_layer=None,
        n_ext=0,
        n_ext_m=0,
    ):
        super().__init__()

        if recon_loss == "mse":
            if last_layer == "softmax":
                raise ValueError("Can't specify softmax last layer with mse loss.")
            last_layer = "identity" if last_layer is None else last_layer
        elif recon_loss == "nb":
            last_layer = "softmax" if last_layer is None else last_layer
        else:
            raise ValueError("Unrecognized loss.")

        print("Decoder Architecture:")
        print(
            "\tMasked linear layer in, ext_m, ext, cond, out: ",
            in_dim,
            n_ext_m,
            n_ext,
            n_cond,
            out_dim,
        )
        if mask is not None:
            print("\twith hard mask.")
        else:
            print("\twith soft mask.")

        self.n_ext = n_ext
        self.n_ext_m = n_ext_m

        self.n_cond = 0
        if n_cond is not None:
            self.n_cond = n_cond

        self.L0 = MaskedCondLayers(
            in_dim,
            out_dim,
            n_cond,
            bias=False,
            n_ext=n_ext,
            n_ext_m=n_ext_m,
            mask=mask,
            ext_mask=ext_mask,
        )

        if last_layer == "softmax":
            self.mean_decoder = nn.Softmax(dim=-1)
        elif last_layer == "softplus":
            self.mean_decoder = nn.Softplus()
        elif last_layer == "exp":
            self.mean_decoder = torch.exp
        elif last_layer == "relu":
            self.mean_decoder = nn.ReLU()
        elif last_layer == "identity":
            self.mean_decoder = lambda a: a
        else:
            raise ValueError("Unrecognized last layer.")

        print("Last Decoder layer:", last_layer)

    def forward(self, z, batch=None):
        if batch is not None:
            batch = one_hot_encoder(batch, n_cls=self.n_cond)
            z_cat = torch.cat((z, batch), dim=-1)
            dec_latent = self.L0(z_cat)
        else:
            dec_latent = self.L0(z)

        recon_x = self.mean_decoder(dec_latent)

        return recon_x, dec_latent

    def nonzero_terms(self):
        v = self.L0.expr_L.weight.data
        nz = (v.norm(p=1, dim=0) > 0).cpu().numpy()
        nz = np.append(nz, np.full(self.n_ext_m, True))
        nz = np.append(nz, np.full(self.n_ext, True))
        return nz

    def n_inactive_terms(self):
        n = (~self.nonzero_terms()).sum()
        return int(n)


class Encoder(nn.Module):
    """ScArches Encoder class. Constructs the encoder sub-network of Celligner2 and CVAE. It will transform primary space
    input to means and log. variances of latent space with n_dimensions = z_dimension.

    Parameters
    ----------
    layer_sizes: List
         List of first and hidden layer sizes
    latent_dim: Integer
         Bottleneck layer (z)  size.
    use_bn: Boolean
         If `True` batch normalization will be applied to layers.
    use_ln: Boolean
         If `True` layer normalization will be applied to layers.
    use_dr: Boolean
         If `True` dropout will applied to layers.
    dr_rate: Float
         Dropput rate applied to all layers, if `dr_rate`==0 no dropput will be applied.
    num_classes: Integer
         Number of classes (conditions) the data contain. if `None` the model will be a normal VAE instead of
         conditional VAE.
    """

    def __init__(
        self,
        layer_sizes: list,
        latent_dim: int,
        use_bn: bool,
        use_ln: bool,
        use_dr: bool,
        dr_rate: float,
        num_classes: int = 0,
        n_expand: int = 0,
        graph_layers: int = 0,
    ):
        super().__init__()
        self.n_classes = num_classes
        self.FC = mySequential()
        self.n_expand = n_expand

        if len(layer_sizes) > 1:
            print("Encoder Architecture:")
            for i, (in_size, out_size) in enumerate(
                zip(layer_sizes[:-1], layer_sizes[1:])
            ):
                if i == 0:
                    print(
                        "\tInput " + "graph " if graph_layers > 0 else "",
                        "Layer in, out and cond:",
                        in_size,
                        out_size,
                        self.n_classes,
                    )
                    self.FC.add_module(
                        name="L{:d}".format(i),
                        module=MaskedCondLayers(
                            in_size,
                            out_size,
                            self.n_classes,
                            bias=True,
                            graph=(i < graph_layers),
                        ),
                    )
                else:
                    print("\tHidden Layer", i, "in/out:", in_size, out_size)
                    if i < graph_layers:
                        node = GraphConv(in_size, out_size)
                    else:
                        node = nn.Linear(in_size, out_size, bias=True)
                    self.FC.add_module(name="L{:d}".format(i), module=node)
                if use_bn:
                    self.FC.add_module(
                        "N{:d}".format(i),
                        module=nn.BatchNorm1d(out_size, affine=True),
                    )
                elif use_ln:
                    self.FC.add_module(
                        "N{:d}".format(i),
                        module=nn.LayerNorm(out_size, elementwise_affine=False),
                    )
                self.FC.add_module(name="A{:d}".format(i), module=nn.ReLU())
                if use_dr:
                    self.FC.add_module(
                        name="D{:d}".format(i), module=nn.Dropout(p=dr_rate)
                    )
        print("\tMean/Var Layer in/out:", layer_sizes[-1], latent_dim)
        self.mean_encoder = nn.Linear(layer_sizes[-1], latent_dim)
        self.log_var_encoder = nn.Linear(layer_sizes[-1], latent_dim)

        if self.n_expand > 0:
            print("\tExpanded Mean/Var Layer in/out:", layer_sizes[-1], self.n_expand)
            self.expand_mean_encoder = nn.Linear(layer_sizes[-1], self.n_expand)
            self.expand_var_encoder = nn.Linear(layer_sizes[-1], self.n_expand)

    def forward(self, x, batch=None, g: Optional[torch.Tensor] = None):
        if batch is not None:
            x = torch.cat((x, batch), dim=-1)
        if self.FC is not None:
            x = self.FC(x)
        means = self.mean_encoder(x)
        log_vars = self.log_var_encoder(x)
        if self.n_expand > 0:
            means = torch.cat((means, self.expand_mean_encoder(x)), dim=-1)
            log_vars = torch.cat((log_vars, self.expand_var_encoder(x)), dim=-1)
        return means, log_vars


class Decoder(nn.Module):
    """ScArches Decoder class. Constructs the decoder sub-network of Celligner2 or CVAE networks. It will transform the
    constructed latent space to the previous space of data with n_dimensions = x_dimension.

    Parameters
    ----------
    layer_sizes: List
         List of hidden and last layer sizes
    latent_dim: Integer
         Bottleneck layer (z)  size.
    recon_loss: String
         Definition of Reconstruction-Loss-Method, 'mse', 'nb' or 'zinb'.
    use_bn: Boolean
         If `True` batch normalization will be applied to layers.
    use_ln: Boolean
         If `True` layer normalization will be applied to layers.
    use_dr: Boolean
         If `True` dropout will applied to layers.
    dr_rate: Float
         Dropput rate applied to all layers, if `dr_rate`==0 no dropput will be applied.
    num_classes: Integer
         Number of classes (conditions) the data contain. if `None` the model will be a normal VAE instead of
         conditional VAE.
    """

    def __init__(
        self,
        layer_sizes: list,
        latent_dim: int,
        recon_loss: str,
        use_bn: bool,
        use_ln: bool,
        use_dr: bool,
        dr_rate: float,
        num_classes: int = 0,
        n_expand: int = 0,
    ):
        super().__init__()
        self.use_dr = use_dr
        self.recon_loss = recon_loss
        self.n_classes = num_classes
        layer_sizes = [latent_dim] + layer_sizes
        print("Decoder Architecture:")
        # Create first Decoder layer
        self.FirstL = mySequential()
        print(
            "\tFirst Layer in, out and cond: ",
            layer_sizes[0],
            layer_sizes[1],
            self.n_classes,
        )
        self.n_expand = n_expand
        if self.n_expand:
            self.FirstL_add = mySequential()
            self.FirstL_add.add_module(
                name="L_add{:d}".format(0),
                module=CondLayers(
                    self.n_expand,
                    layer_sizes[1],
                    self.n_classes,
                    bias=True,
                ),
            )
        self.FirstL.add_module(
            name="L0",
            module=CondLayers(
                layer_sizes[0], layer_sizes[1], self.n_classes, bias=False
            ),
        )

        if use_bn:
            self.FirstL.add_module(
                "N0", module=nn.BatchNorm1d(layer_sizes[1], affine=True)
            )
            if self.n_expand:
                self.FirstL_add.add_module(
                    "N_add{:d}".format(0),
                    module=nn.BatchNorm1d(layer_sizes[1], affine=True),
                )
        elif use_ln:
            self.FirstL.add_module(
                "N0", module=nn.LayerNorm(layer_sizes[1], elementwise_affine=False)
            )
            if self.n_expand:
                self.FirstL_add.add_module(
                    "N_add{:d}".format(0),
                    module=nn.LayerNorm(layer_sizes[1], elementwise_affine=False),
                )
        self.FirstL.add_module(name="A0", module=nn.ReLU())
        if self.n_expand:
            self.FirstL_add.add_module(name="A_add{:d}".format(0), module=nn.ReLU())
        if self.use_dr:
            self.FirstL.add_module(name="D0", module=nn.Dropout(p=dr_rate))
            if self.n_expand:
                self.FirstL_add.add_module(
                    name="D_add{:d}".format(0), module=nn.Dropout(p=dr_rate)
                )

        # Create all Decoder hidden layers
        if len(layer_sizes) > 2:
            self.HiddenL = mySequential()
            for i, (in_size, out_size) in enumerate(
                zip(layer_sizes[1:-1], layer_sizes[2:])
            ):
                if i + 3 < len(layer_sizes):
                    print("\tHidden Layer", i + 1, "in/out:", in_size, out_size)
                    self.HiddenL.add_module(
                        name="L{:d}".format(i + 1),
                        module=nn.Linear(in_size, out_size, bias=False),
                    )
                    if use_bn:
                        self.HiddenL.add_module(
                            "N{:d}".format(i + 1),
                            module=nn.BatchNorm1d(out_size, affine=True),
                        )
                    elif use_ln:
                        self.HiddenL.add_module(
                            "N{:d}".format(i + 1),
                            module=nn.LayerNorm(out_size, elementwise_affine=False),
                        )
                    self.HiddenL.add_module(
                        name="A{:d}".format(i + 1), module=nn.ReLU()
                    )
                    if self.use_dr:
                        self.HiddenL.add_module(
                            name="D{:d}".format(i + 1), module=nn.Dropout(p=dr_rate)
                        )
        else:
            self.HiddenL = None

        # Create Output Layers
        print("\tOutput Layer in/out: ", layer_sizes[-2], layer_sizes[-1], "\n")
        if self.recon_loss == "mse":
            self.recon_decoder = nn.Sequential(
                nn.Linear(layer_sizes[-2], layer_sizes[-1]), nn.ReLU()
            )
        if self.recon_loss == "zinb":
            # mean gamma
            self.mean_decoder = nn.Sequential(
                nn.Linear(layer_sizes[-2], layer_sizes[-1]), nn.Softmax(dim=-1)
            )
            # dropout
            self.dropout_decoder = nn.Linear(layer_sizes[-2], layer_sizes[-1])
        if self.recon_loss == "nb":
            # mean gamma
            self.mean_decoder = nn.Sequential(
                nn.Linear(layer_sizes[-2], layer_sizes[-1]), nn.Softmax(dim=-1)
            )

    def forward(self, z, batch=None):
        # Add Condition Labels to Decoder Input
        if batch is not None:
            z = torch.cat((z, batch), dim=-1)
        if self.n_expand > 0:
            z, z_add = torch.split(
                z, [z.shape[1] - self.n_expand, self.n_expand], dim=1
            )
            dec_latent_add = self.FirstL_add(z_add)
        dec_latent = self.FirstL(z)

        if self.n_expand > 0:
            dec_latent = dec_latent + dec_latent_add
        # Compute Hidden Output
        if self.HiddenL is not None:
            x = self.HiddenL(dec_latent)
        else:
            x = dec_latent

        # Compute Decoder Output
        if self.recon_loss == "mse":
            recon_x = self.recon_decoder(x)
            return recon_x, dec_latent
        elif self.recon_loss == "zinb":
            dec_mean_gamma = self.mean_decoder(x)
            dec_dropout = self.dropout_decoder(x)
            return dec_mean_gamma, dec_dropout, dec_latent
        elif self.recon_loss == "nb":
            dec_mean_gamma = self.mean_decoder(x)
            return dec_mean_gamma, dec_latent


class Classifier(nn.Module):
    """
    Classifier for the Conditional VAE.
    """

    def __init__(
        self,
        layer_sizes: list,
        latent_dim: int,
        dr_rate: float,
        use_bn: bool,
        use_ln: bool,
        use_dr: bool,
        num_classes: int = 0,
        n_expand: int = 0,
    ):
        super().__init__()
        self.use_dr = use_dr
        self.n_classes = num_classes
        layer_sizes = [latent_dim] + layer_sizes + [num_classes]
        print("Classifier Architecture:")
        # Create first Classifier layer
        self.FirstL = nn.Sequential()

        self.n_expand = n_expand
        if self.n_expand > 0:
            self.FirstL_add = nn.Sequential()
            self.FirstL_add.add_module(
                name="L_add0",
                module=nn.Linear(self.n_expand, layer_sizes[1], bias=False),
            )

        print(
            "\tFirst Layer in/out/expand: ",
            layer_sizes[0],
            layer_sizes[1],
            self.n_expand,
        )
        self.FirstL.add_module(
            name="L0", module=nn.Linear(layer_sizes[0], layer_sizes[1], bias=False)
        )
        if use_bn:
            self.FirstL.add_module(
                "N0",
                module=nn.BatchNorm1d(layer_sizes[1], affine=True),
            )
            if self.n_expand > 0:
                self.FirstL_add.add_module(
                    name="N_add0",
                    module=nn.BatchNorm1d(layer_sizes[1], affine=True),
                )
        elif use_ln:
            self.FirstL.add_module(
                "N0", module=nn.LayerNorm(layer_sizes[1], elementwise_affine=False)
            )
            if self.n_expand > 0:
                self.FirstL_add.add_module(
                    name="N_add0",
                    module=nn.LayerNorm(layer_sizes[1], elementwise_affine=False),
                )
        self.FirstL.add_module(name="A0", module=nn.ReLU())
        if self.n_expand > 0:
            self.FirstL_add.add_module(name="A_add0", module=nn.ReLU())
        if self.use_dr:
            self.FirstL.add_module(name="D0", module=nn.Dropout(p=dr_rate))
            if self.n_expand > 0:
                self.FirstL_add.add_module(name="D_add0", module=nn.Dropout(p=dr_rate))

        # Create all Classifier hidden layers
        if len(layer_sizes) > 3:
            self.HiddenL = nn.Sequential()
            for i, (in_size, out_size) in enumerate(
                zip(layer_sizes[1:-2], layer_sizes[2:-1])
            ):
                print("\tHidden Layer", i + 1, "in/out:", in_size, out_size)
                self.HiddenL.add_module(
                    name="L{:d}".format(i + 1),
                    module=nn.Linear(in_size, out_size, bias=False),
                )
                # https://stats.stackexchange.com/questions/361700/lack-of-batch-normalization-before-last-fully-connected-layer
                if use_bn and i + 3 < len(layer_sizes):
                    self.HiddenL.add_module(
                        "N{:d}".format(i + 1),
                        module=nn.BatchNorm1d(out_size, affine=True),
                    )
                elif use_ln:
                    self.HiddenL.add_module(
                        "N{:d}".format(i + 1),
                        module=nn.LayerNorm(out_size, elementwise_affine=False),
                    )
                self.HiddenL.add_module(name="A{:d}".format(i + 1), module=nn.ReLU())
                if self.use_dr:
                    self.HiddenL.add_module(
                        name="D{:d}".format(i + 1), module=nn.Dropout(p=dr_rate)
                    )
        else:
            self.HiddenL = None

        # Create Output Layers
        print("\tOutput Layer in/out: ", layer_sizes[-2], layer_sizes[-1], "\n")
        # https://glassboxmedicine.com/2019/05/26/classification-sigmoid-vs-softmax/
        self.classifier = nn.Sequential(nn.Linear(layer_sizes[-2], layer_sizes[-1]))

    def forward(self, z):
        # predicts class probabilities from latent space
        if self.n_expand > 0:
            z, z_add = torch.split(
                z, [z.shape[1] - self.n_expand, self.n_expand], dim=1
            )
            zL_add = self.FirstL_add(z_add)
        zL = self.FirstL(z)
        if self.n_expand > 0:
            zL = zL + zL_add
        if self.HiddenL is not None:
            x = self.HiddenL(zL)
        else:
            x = zL
        return self.classifier(x)
