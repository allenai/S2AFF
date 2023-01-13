"""
This file contains the classes required by Semantic Scholar's
TIMO tooling.

You must provide a wrapper around your model, as well
as a definition of the objects it expects, and those it returns.
"""

from typing import List
from os.path import join, basename
import torch

from pydantic import BaseModel, BaseSettings, Field

from s2aff import S2AFF
from s2aff.ror import RORIndex
from s2aff.model import NERPredictor, PairwiseRORLightGBMReranker
from s2aff.consts import PATHS


class Instance(BaseModel):
    """
    Describes one Instance over which the model performs inference.

    The fields below are examples only; please replace them with
    appropriate fields for your model.

    To learn more about declaring pydantic model fields, please see:
    https://pydantic-docs.helpmanual.io/
    """

    raw_affiliation: str = Field(description="Raw affiliation string")



class Prediction(BaseModel):
    """
    Describes the outcome of inference for one Instance

    The fields below are examples only; please replace them with
    appropriate fields for your model.

    To learn more about declaring pydantic model fields, please see:
    https://pydantic-docs.helpmanual.io/
    """

    ror_id: str = Field(description="ROR id for the top result")
    display_name: str = Field(description="Standardized name for the top result")
    score: float = Field(description="Score from the LightGBM stage model for the top result")
    main: List[str] = Field(description="Main affiliation strings from NER step")
    child: List[str] = Field(description="Child affiliation strings from NER step")
    address: List[str] = Field(description="Address affiliation strings from NER step")

class PredictorConfig(BaseSettings):
    """
    Configuration required by the model to do its work.
    Uninitialized fields will be set via Environment variables.

    The fields below are examples only; please replace them with ones
    appropriate for your model. These serve as a record of the ENV
    vars the consuming application needs to set.
    """


class Predictor:
    """
    Interface on to your underlying model.

    This class is instantiated at application startup as a singleton.
    You should initialize your model inside of it, and implement
    prediction methods.

    If you specified an artifacts.tar.gz for your model, it will
    have been extracted to `artifacts_dir`, provided as a constructor
    arg below.
    """

    _config: PredictorConfig
    _artifacts_dir: str

    def __init__(self, config: PredictorConfig, artifacts_dir: str):
        self._config = config
        self._artifacts_dir = artifacts_dir
        self._load_model()

    def _load_model(self) -> None:
        """
        Perform whatever start-up operations are required to get your
        model ready for inference. This operation is performed only once
        during the application life-cycle.
        """
        ner_predictor = NERPredictor(
            model_path=join(self._artifacts_dir, basename(PATHS["ner_model"])), use_cuda=torch.cuda.is_available()
        )
        ror_index = RORIndex(
            ror_data_path=join(self._artifacts_dir, basename(PATHS["ror_data"])),
            country_info_path=join(self._artifacts_dir, basename(PATHS["country_info"])),
            works_counts_path=join(self._artifacts_dir, basename(PATHS["openalex_works_counts"])),
        )
        pairwise_model = PairwiseRORLightGBMReranker(
            ror_index,
            model_path=join(self._artifacts_dir, basename(PATHS["lightgbm_model"])),
            kenlm_model_path=join(self._artifacts_dir, basename(PATHS["kenlm_model"])),
        )
        self.s2aff = S2AFF(ner_predictor, ror_index, pairwise_model)

    @staticmethod
    def convert_raw_prediction_to_Prediction(prediction) -> Prediction:
        if len(prediction["stage2_candidates"]) == 0:
            prediction_instance = Prediction(ror_id="", score=0, main="", child="", address="")
        else:
            prediction_instance = Prediction(
                ror_id=prediction["stage2_candidates"][0],
                display_name=prediction["top_candidate_display_name"]
                score=prediction["stage2_scores"][0],
                main=prediction["main_from_ner"],
                child=prediction["child_from_ner"],
                address=prediction["address_from_ner"],
            )

        return prediction_instance

    def predict_batch(self, instances: List[Instance]) -> List[Prediction]:
        """
        Method called by the client application. One or more Instances will
        be provided, and the caller expects a corresponding Prediction for
        each one.

        If your model gets performance benefits from batching during inference,
        implement that here, explicitly.

        Otherwise, you can leave this method as-is and just implement
        `predict_one()` above. The default implementation here passes
        each Instance into `predict_one()`, one at a time.

        The size of the batches passed into this method is configurable
        via environment variable by the calling application.
        """
        predictions = self.s2aff.predict([i.raw_affiliation for i in instances])
        return [self.convert_raw_prediction_to_Prediction(i) for i in predictions]
