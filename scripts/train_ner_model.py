import pickle
from s2aff.model import NERPredictor
from s2aff.consts import PATHS

# load provided training data
# note: how this was made is not included in the release repo
# as it is messy and hard to reproduce
with open(PATHS["ner_training_data"], "rb") as f:
    df_training, df_validation, df_test = pickle.load(f)

ner_model = NERPredictor()

validation_metrics = ner_model.fit(df_training, df_validation)
print(validation_metrics)

# if you want to overwrite the provided model uncomment here
# ner_model.save_model(PATHS["ner_model"])
