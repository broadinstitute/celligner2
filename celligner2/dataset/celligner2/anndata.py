import numpy as np
import torch
from torch.utils.data import Dataset
from scipy import sparse

from .data_handling import remove_sparsity
from ._utils import label_encoder


class AnnotatedDataset(Dataset):
    """Dataset handler for celligner2 model and trainer.
       Parameters
       ----------
       adata: : `~anndata.AnnData`
            Annotated data matrix.
       condition_keys: List[String]
            column name of conditions in `adata.obs` data frame.
       condition_encoder: List[Dict]
            dictionary of encoded conditions.
       cell_type_key: String
            column name of different celltype hierarchies in `adata.obs` data frame.
       cell_type_encoder: List[Dict]
            dictionary of encoded celltypes.
       predictor_encoder: List[Dict]
            dictionary of encoded predictors. 
       predictor_keys: List[String]
            column name of predictors in `adata.obs` data frame.
    """
    def __init__(self,
                 adata,
                 condition_keys=None,
                 condition_encoder=None,
                 cell_type_key=None,
                 cell_type_encoder=None,
                 predictor_encoder=None,
                 predictor_keys=None,
                 ):

        self.X_norm = None

        self.condition_keys = condition_keys
        self.condition_encoder = condition_encoder
        self.cell_type_key = cell_type_key
        self.cell_type_encoder = cell_type_encoder
        self.predictor_encoder = predictor_encoder
        self.predictor_keys = predictor_keys

        if sparse.issparse(adata.X):
            adata = remove_sparsity(adata)
        self.data = torch.tensor(adata.X)

        self.size_factors = torch.tensor(adata.obs['celligner2_size_factors'])
        self.labeled_vector = torch.tensor(adata.obs['celligner2_labeled'])

        # Encode condition strings to integer
        if self.condition_keys is not None:
            conditions = list()
            for key in self.condition_keys:
                condition = label_encoder(
                    adata,
                    encoder=self.condition_encoder,
                    label_key=key,
                )
                conditions.append(condition)
            conditions = np.stack(conditions)
            self.conditions = torch.tensor(conditions, dtype=torch.long)

        # Encode predictors strings to integer
        if self.predictor_keys is not None:
            predictors = list()
            for key in self.predictor_keys:
                predictor = label_encoder(
                    adata,
                    encoder=self.predictor_encoder,
                    label_key=key,
                )
                predictors.append(predictor)
            predictors = np.stack(predictors)
            self.predictors = torch.tensor(predictors, dtype=torch.long)

        # Encode cell type strings to integer
        if self.cell_type_key is not None:
            cell_types = list()
            level_cell_types = label_encoder(
                adata,
                encoder=self.cell_type_encoder,
                label_key=cell_type_key,
            )
            cell_types.append(level_cell_types)
            cell_types = np.stack(cell_types)
            self.cell_types = torch.tensor(cell_types, dtype=torch.long)

    def __getitem__(self, index):
        outputs = dict()

        outputs["x"] = self.data[index, :]
        outputs["labeled"] = self.labeled_vector[index]
        outputs["sizefactor"] = self.size_factors[index]

        if self.condition_keys:
            outputs["batch"] = self.conditions[index,:]

        if self.cell_type_key:
            outputs["celltypes"] = self.cell_types[index]

        if self.predictor_keys:
            outputs["predictors"] = self.predictors[self.predictor_keys,:]

        return outputs

    def __len__(self):
        return self.data.size(0)

    @property
    def condition_label_encoder(self) -> dict:
        return self.condition_encoder

    @condition_label_encoder.setter
    def condition_label_encoder(self, value: dict):
        if value is not None:
            self.condition_encoder = value

    @property
    def predictor_label_encoder(self) -> dict:
        return self.predictor_encoder

    @predictor_label_encoder.setter
    def predictor_label_encoder(self, value: dict):
        if value is not None:
            self.predictor_encoder = value

    @property
    def cell_type_label_encoder(self) -> dict:
        return self.cell_type_encoder

    @cell_type_label_encoder.setter
    def cell_type_label_encoder(self, value: dict):
        if value is not None:
            self.cell_type_encoder = value

    @property
    def stratifier_weights(self):
        maincond = self.conditions[:, 0]
        conditions = maincond.detach().cpu().numpy()
        condition_coeff = 1 / len(conditions)
        weights_per_condition = list()
        for i in range(len(maincond)):
            samples_per_condition = np.count_nonzero(conditions == i)
            if samples_per_condition == 0:
                weights_per_condition.append(0)
            else:
                weights_per_condition.append((1 / samples_per_condition) * condition_coeff)
        strat_weights = np.copy(conditions)
        for i in range(len(conditions)):
            strat_weights = np.where(strat_weights == i, weights_per_condition[i], strat_weights)

        return strat_weights.astype(float)
