import numpy as np


def label_encoder(adata, encoder, label_key=None):
    """Encode labels of Annotated `adata` matrix.
       Parameters
       ----------
       adata: : `~anndata.AnnData`
            Annotated data matrix.
       encoder: Dict
            dictionary of encoded labels.
       label_key: String
            column name of conditions in `adata.obs` data frame.

       Returns
       -------
       labels: `~numpy.ndarray`
            Array of encoded labels
       label_encoder: Dict
            dictionary with labels and encoded labels as key, value pairs.
    """
    unique_conditions = list(np.unique(adata.obs[label_key]))
    labels = np.zeros(adata.shape[0])
    if not set(unique_conditions).issubset(set(encoder.keys())):
        print(f"Warning: Labels: {set(unique_conditions)-set(encoder.keys())} in \
adata.obs[{label_key}] is not a subset of label-encoder!")
        print("Therefore integer value of those labels is set to -1")
        for data_cond in unique_conditions:
            if data_cond not in encoder.keys():
                labels[adata.obs[label_key] == data_cond] = -1

    for condition, label in encoder.items():
        labels[adata.obs[label_key] == condition] = label
    return labels

def label_encoder_2D(adata, encoder, label_sets):
    """Encode labels of Annotated `adata` matrix.
       Parameters
       ----------
       adata: : `~anndata.AnnData`
            Annotated data matrix.
       encoder: Dict
            dictionary of encoded labels.
       label_sets: Dict
            column name of conditions in `adata.obs` data frame.

       Returns
       -------
       labels: `~numpy.ndarray`
            Array of encoded labels
    """
    labels = np.zeros((adata.shape[0], len(encoder.keys())))
    
    miss = set()
    for label_key in label_sets.keys():
        unique_conditions = list(np.unique(adata.obs[label_key]))

        if not set(unique_conditions).issubset(set(encoder.keys())):
            miss = miss | set(unique_conditions)-set(encoder.keys())
            print(f"Warning: Labels: {miss} in \
    adata.obs[{label_key}] is not a subset of label-encoder!")
            print("Therefore integer value of those labels is set to -1")
    
    for k, values in label_sets.items():
        for val in values:
            if val in miss:
                # setting all values of that column for samples having missing labels to -1
                for toset in set(values)-miss:
                    labels[adata.obs[k] == val, encoder[toset]] = -1
            else:
                labels[adata.obs[k] == val, encoder[val]] = 1
    return labels