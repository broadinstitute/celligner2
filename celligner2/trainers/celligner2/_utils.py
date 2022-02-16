import sys
import numpy as np
import re
import torch
import collections.abc as container_abcs
from torch.utils.data import DataLoader
from scipy import sparse

from celligner2.dataset import celligner2Dataset
from celligner2.dataset import remove_sparsity


def print_progress(epoch, logs, n_epochs=10000, only_val_losses=True):
    """Creates Message for '_print_progress_bar'.

    Parameters
    ----------
    epoch: Integer
         Current epoch iteration.
    logs: Dict
         Dictionary of all current losses.
    n_epochs: Integer
         Maximum value of epochs.
    only_val_losses: Boolean
         If 'True' only the validation dataset losses are displayed, if 'False' additionally the training dataset
         losses are displayed.

    Returns
    -------
    """
    message = ""
    for key in logs:
        if only_val_losses:
            if "val_" in key and "unweighted" not in key:
                message += f" - {key:s}: {logs[key][-1]:7.10f}"
        else:
            if "unweighted" not in key:
                message += f" - {key:s}: {logs[key][-1]:7.10f}"

    _print_progress_bar(
        epoch + 1, n_epochs, prefix="", suffix=message, decimals=1, length=20
    )


def _print_progress_bar(
    iteration, total, prefix="", suffix="", decimals=1, length=100, fill="█"
):
    """Prints out message with a progress bar.

    Parameters
    ----------
    iteration: Integer
         Current epoch.
    total: Integer
         Maximum value of epochs.
    prefix: String
         String before the progress bar.
    suffix: String
         String after the progress bar.
    decimals: Integer
         Digits after comma for all the losses.
    length: Integer
         Length of the progress bar.
    fill: String
         Symbol for filling the bar.

    Returns
    -------
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_len = int(length * iteration // total)
    bar = fill * filled_len + "-" * (length - filled_len)
    sys.stdout.write("\r%s |%s| %s%s %s" % (prefix, bar, percent, "%", suffix)),
    if iteration == total:
        sys.stdout.write("\n")
    sys.stdout.flush()


def train_test_split(adata, train_frac=0.85, cell_type_key=None):
    """Splits 'Anndata' object into training and validation data.

    Parameters
    ----------
    adata: `~anndata.AnnData`
         `AnnData` object for training the model.
    train_frac: float
         Train-test split fraction. the model will be trained with train_frac for training
         and 1-train_frac for validation.

    Returns
    -------
    `AnnData` objects for training and validating the model.
    """
    if train_frac == 1:
        return adata, None
    else:
        indices = np.arange(adata.shape[0])
        if cell_type_key is not None:
            train_idx = []
            val_idx = []
            cell_types = adata.obs[cell_type_key].unique().tolist()
            for cell_type in cell_types:
                ct_idx = indices[adata.obs[cell_type_key] == cell_type]
                n_train_samples = int(np.ceil(train_frac * len(ct_idx)))
                np.random.shuffle(ct_idx)
                train_idx.append(ct_idx[:n_train_samples])
                val_idx.append(ct_idx[n_train_samples:])
            train_idx = np.concatenate(train_idx)
            val_idx = np.concatenate(val_idx)
        else:
            n_train_samples = int(np.ceil(train_frac * len(indices)))
            np.random.shuffle(indices)
            train_idx = indices[:n_train_samples]
            val_idx = indices[n_train_samples:]

        return train_idx, val_idx


def make_dataset(
    adata,
    train_frac=0.9,
    condition_keys=None,
    cell_type_key=None,
    condition_encoder=None,
    cell_type_encoder=None,
    predictor_keys=None,
    predictor_encoder=None,
    min_weight=0.0,
):
    """Splits 'adata' into train and validation data and converts them into 'CustomDatasetFromAdata' objects.

    Parameters
    ----------

    Returns
    -------
    Training 'CustomDatasetFromAdata' object, Validation 'CustomDatasetFromAdata' object
    """
    if sparse.issparse(adata.X):
        adata = remove_sparsity(adata)

    # if data contains nan, replace them with mean of the column
    goodloc = torch.tensor(~np.isnan(adata.X))
    if np.isnan(adata.X).any():
        nanme = np.nanmean(adata.X, axis=0)
        for i, val in enumerate(np.isnan(adata.X).T):
            adata.X[val, i] = nanme[i]

    size_factors = adata.X.sum(1)
    if len(size_factors.shape) < 2:
        size_factors = np.expand_dims(size_factors, axis=1)
    adata.obs["celligner2_size_factors"] = size_factors

    train_idx, val_idx = train_test_split(adata, train_frac, cell_type_key)
    goodloc_train = goodloc[train_idx, :]
    goodloc_val = goodloc[val_idx, :]
    train_adata = adata[train_idx, :]
    validation_adata = adata[val_idx, :]

    data_set_train = celligner2Dataset(
        train_adata,
        goodloc_train,
        condition_keys=condition_keys,
        cell_type_key=cell_type_key,
        condition_encoder=condition_encoder,
        cell_type_encoder=cell_type_encoder,
        predictor_encoder=predictor_encoder,
        predictor_keys=predictor_keys,
        minweight=min_weight,
    )
    if train_frac == 1:
        return data_set_train, None
    else:
        data_set_valid = celligner2Dataset(
            validation_adata,
            goodloc_val,
            condition_keys=condition_keys,
            cell_type_key=cell_type_key,
            condition_encoder=condition_encoder,
            cell_type_encoder=cell_type_encoder,
            predictor_encoder=predictor_encoder,
            predictor_keys=predictor_keys,
            minweight=min_weight,
        )
        return data_set_train, data_set_valid


def custom_collate(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""
    np_str_obj_array_pattern = re.compile(r"[SaUO]")
    default_collate_err_msg_format = (
        "default_collate: batch must contain tensors, numpy arrays, numbers, "
        "dicts or lists; found {}"
    )

    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.stack(batch, 0, out=out)

    elif (
        elem_type.__module__ == "numpy"
        and elem_type.__name__ != "str_"
        and elem_type.__name__ != "string_"
    ):
        elem = batch[0]
        if elem_type.__name__ == "ndarray" or elem_type.__name__ == "memmap":
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))

            return custom_collate([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)

    elif isinstance(elem, container_abcs.Mapping):
        output = {key: custom_collate([d[key] for d in batch]) for key in elem}
        return output
