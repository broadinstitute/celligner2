import numpy as np
import torch
from torch.utils.data import Dataset

from ._utils import label_encoder, label_encoder_2D


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

    def __init__(
        self,
        adata,
        goodloc=None,
        condition_keys=None,
        condition_encoder=None,
        cell_type_key=None,
        cell_type_encoder=None,
        predictor_encoder=None,
        predictor_keys=None,
        minweight=0.0,
    ):

        self.X_norm = None
        self.minweight = minweight
        self.condition_keys = condition_keys
        self.condition_encoder = condition_encoder
        self.cell_type_key = cell_type_key
        self.cell_type_encoder = cell_type_encoder
        self.predictor_encoder = predictor_encoder
        self.predictor_keys = predictor_keys

        self.size_factors = torch.tensor(adata.obs["celligner2_size_factors"])
        self.data = torch.tensor(adata.X)
        if goodloc is not None:
            self.goodloc = torch.tensor(goodloc, dtype=torch.bool)
        else:
            self.goodloc = torch.ones(self.data.shape, dtype=torch.bool)

        # Encode condition strings to integer
        if self.condition_keys is not None:
            condition_sets = {key: set(adata.obs[key]) for key in self.condition_keys}
            conditions = label_encoder_2D(
                adata,
                encoder=self.condition_encoder,
                label_sets=condition_sets,
            )
            self.conditions = torch.tensor(conditions, dtype=torch.long)

        # Encode predictors strings to integer
        if self.predictor_keys is not None:
            predictor_set = {key: set(adata.obs[key]) for key in self.predictor_keys}
            predictors = label_encoder_2D(
                adata,
                encoder=self.predictor_encoder,
                label_sets=predictor_set,
            )
            self.predictors = torch.tensor(predictors, dtype=torch.long)
            # import pdb; pdb.set_trace()
            # predictors[predictors == -1] = 0
            # weights = predictors.sum(0)
            # self.weights =  (( 1 + minweight ) - weights / weights.max()) / (1 + minweight)

        # Encode cell type strings to integer
        # if self.cell_type_key is not None:
        #    level_cell_types = label_encoder(
        #        adata,
        #        encoder=self.cell_type_encoder,
        #        label_key=cell_type_key,
        #    )
        #    self.cell_types = torch.tensor(level_cell_types, dtype=torch.long)

    def __getitem__(self, index):
        outputs = dict()
        outputs["x"] = self.data[index, :]
        outputs["sizefactor"] = self.size_factors[index]
        outputs["goodloc"] = self.goodloc[index, :]

        if self.condition_keys:
            outputs["batch"] = self.conditions[index]

        # if self.cell_type_key:
        #    outputs["celltypes"] = self.cell_types[index]

        if self.predictor_keys:
            outputs["classes"] = self.predictors[index]
            # outputs['weight'] = self.weights

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
        strat_weights = np.zeros(len(self.conditions)).astype(float)
        # for conditions
        conditions = self.conditions.detach().cpu().numpy().T
        weights_per_condition = list()
        for i in range(conditions.shape[0]):
            samples_per_condition = np.count_nonzero(conditions[i])
            if samples_per_condition == 0:
                weights_per_condition.append(0)
            else:
                weights_per_condition.append(
                    1 / (samples_per_condition * conditions.shape[0])
                )
        for i in range(conditions.shape[0]):
            strat_weights += np.where(conditions[i], weights_per_condition[i], 0)
        # for cell types
        predictions = self.predictors.detach().cpu().numpy().T
        weights_per_prediction = list()
        for i in range(predictions.shape[0]):
            samples_per_prediction = np.count_nonzero(predictions[i])
            if samples_per_prediction == 0:
                weights_per_prediction.append(0)
            else:
                weights_per_prediction.append(
                    1 / (samples_per_prediction * predictions.shape[0])
                )
        for i in range(predictions.shape[0]):
            strat_weights += np.where(predictions[i], weights_per_prediction[i], 0)
        # import pdb; pdb.set_trace()
        return strat_weights + self.minweight
