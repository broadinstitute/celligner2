import torch
from torch.autograd import Variable
import torch.nn.functional as F

from ._utils import partition


def mse(recon_x, x):
    """Computes MSE loss between reconstructed data and ground truth data.

    Parameters
    ----------
    recon_x: torch.Tensor
         Torch Tensor of reconstructed data
    x: torch.Tensor
         Torch Tensor of ground truth data

    Returns
    -------
    MSE loss value
    """
    mse_loss = torch.nn.functional.mse_loss(recon_x, x, reduction="none")
    return mse_loss


def nb(x: torch.Tensor, mu: torch.Tensor, theta: torch.Tensor, eps=1e-8):
    """
    This negative binomial function was taken from:
    Title: scvi-tools
    Authors: Romain Lopez <romain_lopez@gmail.com>,
             Adam Gayoso <adamgayoso@berkeley.edu>,
             Galen Xing <gx2113@columbia.edu>
    Date: 16th November 2020
    Code version: 0.8.1
    Availability: https://github.com/YosefLab/scvi-tools/blob/8f5a9cc362325abbb7be1e07f9523cfcf7e55ec0/scvi/core/distributions/_negative_binomial.py

    Computes negative binomial loss.
    Parameters
    ----------
    x: torch.Tensor
         Torch Tensor of ground truth data.
    mu: torch.Tensor
         Torch Tensor of means of the negative binomial (has to be positive support).
    theta: torch.Tensor
         Torch Tensor of inverse dispersion parameter (has to be positive support).
    eps: Float
         numerical stability constant.

    Returns
    -------
    If 'mean' is 'True' NB loss value gets returned, otherwise Torch tensor of losses gets returned.
    """
    if theta.ndimension() == 1:
        theta = theta.view(1, theta.size(0))

    log_theta_mu_eps = torch.log(theta + mu + eps)
    res = (
        theta * (torch.log(theta + eps) - log_theta_mu_eps)
        + x * (torch.log(mu + eps) - log_theta_mu_eps)
        + torch.lgamma(x + theta)
        - torch.lgamma(theta)
        - torch.lgamma(x + 1)
    )
    return res


def zinb(
    x: torch.Tensor, mu: torch.Tensor, theta: torch.Tensor, pi: torch.Tensor, eps=1e-8
):
    """
    This zero-inflated negative binomial function was taken from:
    Title: scvi-tools
    Authors: Romain Lopez <romain_lopez@gmail.com>,
             Adam Gayoso <adamgayoso@berkeley.edu>,
             Galen Xing <gx2113@columbia.edu>
    Date: 16th November 2020
    Code version: 0.8.1
    Availability: https://github.com/YosefLab/scvi-tools/blob/8f5a9cc362325abbb7be1e07f9523cfcf7e55ec0/scvi/core/distributions/_negative_binomial.py

    Computes zero inflated negative binomial loss.
    Parameters
    ----------
    x: torch.Tensor
         Torch Tensor of ground truth data.
    mu: torch.Tensor
         Torch Tensor of means of the negative binomial (has to be positive support).
    theta: torch.Tensor
         Torch Tensor of inverses dispersion parameter (has to be positive support).
    pi: torch.Tensor
         Torch Tensor of logits of the dropout parameter (real support)
    eps: Float
         numerical stability constant.

    Returns
    -------
    If 'mean' is 'True' ZINB loss value gets returned, otherwise Torch tensor of losses gets returned.
    """
    # theta is the dispersion rate. If .ndimension() == 1, it is shared for all cells (regardless of batch or labels)
    if theta.ndimension() == 1:
        theta = theta.view(
            1, theta.size(0)
        )  # In this case, we reshape theta for broadcasting

    softplus_pi = F.softplus(-pi)  #  uses log(sigmoid(x)) = -softplus(-x)
    log_theta_eps = torch.log(theta + eps)
    log_theta_mu_eps = torch.log(theta + mu + eps)
    pi_theta_log = -pi + theta * (log_theta_eps - log_theta_mu_eps)

    case_zero = F.softplus(pi_theta_log) - softplus_pi
    mul_case_zero = torch.mul((x < eps).type(torch.float32), case_zero)

    case_non_zero = (
        -softplus_pi
        + pi_theta_log
        + x * (torch.log(mu + eps) - log_theta_mu_eps)
        + torch.lgamma(x + theta)
        - torch.lgamma(theta)
        - torch.lgamma(x + 1)
    )
    mul_case_non_zero = torch.mul((x > eps).type(torch.float32), case_non_zero)

    res = mul_case_zero + mul_case_non_zero
    return res


def pairwise_distance(x, y):
    x = x.view(x.shape[0], x.shape[1], 1)
    y = torch.transpose(y, 0, 1)
    output = torch.sum((x - y) ** 2, 1)
    output = torch.transpose(output, 0, 1)

    return output


def gaussian_kernel_matrix(x, y, alphas):
    """Computes multiscale-RBF kernel between x and y.

    Parameters
    ----------
    x: torch.Tensor
         Tensor with shape [batch_size, z_dim].
    y: torch.Tensor
         Tensor with shape [batch_size, z_dim].
    alphas: Tensor

    Returns
    -------
    Returns the computed multiscale-RBF kernel between x and y.
    """

    dist = pairwise_distance(x, y).contiguous()
    dist_ = dist.view(1, -1)

    alphas = alphas.view(alphas.shape[0], 1)
    beta = 1.0 / (2.0 * alphas)

    s = torch.matmul(beta, dist_)

    return torch.sum(torch.exp(-s), 0).view_as(dist)


def mmd_loss_calc(source_features, target_features):
    """Initializes Maximum Mean Discrepancy(MMD) between source_features and target_features.

    - Gretton, Arthur, et al. "A Kernel Two-Sample Test". 2012.

    Parameters
    ----------
    source_features: torch.Tensor
         Tensor with shape [batch_size, z_dim]
    target_features: torch.Tensor
         Tensor with shape [batch_size, z_dim]

    Returns
    -------
    Returns the computed MMD between x and y.
    """
    alphas = [
        1e-6,
        1e-5,
        1e-4,
        1e-3,
        1e-2,
        1e-1,
        1,
        5,
        10,
        15,
        20,
        25,
        30,
        35,
        100,
        1e3,
        1e4,
        1e5,
        1e6,
    ]
    alphas = Variable(torch.FloatTensor(alphas)).to(device=source_features.device)

    cost = torch.mean(gaussian_kernel_matrix(source_features, source_features, alphas))
    cost += torch.mean(gaussian_kernel_matrix(target_features, target_features, alphas))
    cost -= 2 * torch.mean(
        gaussian_kernel_matrix(source_features, target_features, alphas)
    )

    return cost


def mmd(y, c, n_conditions, beta, boundary, maindataset):
    """Initializes Maximum Mean Discrepancy(MMD) between every different condition.

    Parameters
    ----------
    n_conditions: integer
         Number of classes (conditions) the data contain.
    beta: float
         beta coefficient for MMD loss.
    y: torch.Tensor
         Torch Tensor of computed latent data.
    c: torch.Tensor
         Torch Tensor of condition labels.

    Returns
    -------
    Returns MMD loss.
    """
    # partition separates y into num_cls subsets w.r.t. their labels c
    conditions_mmd = partition(y, c, n_conditions, maindataset)
    loss = torch.tensor(0.0, device=y.device)
    if boundary is not None:
        for i in range(boundary):
            # if conditions_mmd ==
            for j in range(boundary, n_conditions):
                if conditions_mmd[i].size(0) < 2 or conditions_mmd[j].size(0) < 2:
                    continue
                loss += mmd_loss_calc(conditions_mmd[i], conditions_mmd[j])
    else:
        for i in range(len(conditions_mmd)):
            if conditions_mmd[i].size(0) < 1:
                continue
            for j in range(i):
                if conditions_mmd[j].size(0) < 1 or i == j:
                    continue
                if maindataset is not None and j != 0:
                    continue
                loss += mmd_loss_calc(conditions_mmd[i], conditions_mmd[j])
    return beta * loss


def kl(z1_log_var, z1_mean):
    """kl loss for VAE

    Parameters
    ----------
    z1_log_var: torch.Tensor
        Tensor of shape [batch_size, z_dim]
    z1_mean: torch.Tensor
        Tensor of shape [batch_size, z_dim]

    Returns
    -------
    Returns the computed kl loss.
    """
    z1_var = torch.exp(z1_log_var) + 1e-4
    return -0.5 * torch.sum(1 + z1_log_var - z1_mean ** 2 - z1_var)


def cnv_recon_loss(recon_x, x):
    loss = F.binary_cross_entropy(recon_x[0], x[0], reduction="mean")
    return loss


def expr_recon_loss(recon_x, x):
    loss = F.binary_cross_entropy(recon_x[1], x[1], reduction="mean")
    return loss


def classifier_er(pred_y, y, beta=0.5):
    bootstrap = -(1.0 - beta) * torch.sum(
        F.softmax(pred_y, dim=1) * F.log_softmax(pred_y, dim=1), dim=1
    )
    return torch.sum(bootstrap)


def kl_loss(mean, log_var):
    loss = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    return loss


def classifier_loss1(pred_y, y):
    loss = F.cross_entropy(pred_y, y, reduction="sum")
    return loss


def classifier_loss(pred_y, y, tmpidx, beta=0.5):
    return beta * F.cross_entropy(pred_y[tmpidx], y[tmpidx], reduction="sum")


def classifier_sb_loss(pred_y, y, tmpidx, beta=0.5):
    # cross_entropy=- t * log(p)
    utmpidx = [y_ for y_ in range(len(y)) if y[y_] not in tmpidx]
    # if (len(utmpidx)>0):
    _, z = torch.max(F.softmax(pred_y[utmpidx], dim=1), dim=1)
    z = z.view(-1, 1)
    bootstrap = F.log_softmax(pred_y[utmpidx], dim=1).gather(1, z).view(-1)
    # second term=(1 - beta) * z * log(p)
    bootstrap = -(1.0 - beta) * bootstrap
    loss = beta * F.cross_entropy(
        pred_y[tmpidx], y[tmpidx], reduction="sum"
    ) + torch.sum(bootstrap)
    return loss
    # else:
    # return beta*F.cross_entropy(pred_y[tmpidx], y[tmpidx], reduction='sum')


def classifier_hb_loss(pred_y, y, ignore_index=-1, beta=0.7, weight=None):
    if weight is None:
        weight = torch.ones(y.shape).to(device=pred_y.device)
    else:
        weight = torch.tensor(weight).to(device=pred_y.device)

    weight[y == ignore_index] = 0
    y[y == ignore_index] = 0
    bootstrap = -(1.0 - beta) * torch.sum(
        F.softmax(pred_y, dim=1) * F.log_softmax(pred_y, dim=1), dim=1
    )
    ce = F.binary_cross_entropy_with_logits(
        pred_y, y.float(), reduction="sum", weight=weight
    )
    # import pdb; pdb.set_trace()
    return beta * ce + torch.sum(bootstrap)
