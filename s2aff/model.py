import os
import torch

# have to do this to avoid weird bugs
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.multiprocessing.set_sharing_strategy("file_system")

import gc
import numpy as np
import lightgbm as lgb
import kenlm
from s2aff.text import fix_text
from s2aff.features import make_lightgbm_features, parse_ror_entry_into_single_string_lightgbm, FEATURE_NAMES
from simpletransformers.ner import NERModel, NERArgs
from blingfire import text_to_words
from s2aff.consts import PATHS
from s2aff.text import fix_text, CERTAINLY_MAIN


FEATURE_NAMES = list(FEATURE_NAMES)


def parse_ner_prediction(ner_prediction, ror_index):
    """Parse the NER prediction that comes out of a NERPredictor
    into main entity, child entities, address, and early candidates for the
    next retrieval stage.

    Args:
        ner_prediction (list of dicts): Output of a NERPredictor.predict. Looks like this:
            [{'Chinese': 'I-MAIN'},
                {'Academy': 'I-MAIN'},
                {'of': 'I-MAIN'},
                {'Sciences': 'I-MAIN'},
                {'PRC': 'I-ADDRESS'}]

                Note that it's one prediction and not a list of predictions.

    Returns:
        main (str): Main entity of the child affiliation string.
        child (list[str]): Child entities of the raw affiliation string.
        address (list): Address part of the raw affiliation string.
        early_candidates (list[str]): set of candidates based off child entities.
            These appear in the direct ROR look-up table.
    """
    main = ""
    child = ""
    main_child = ""
    address = ""
    previous_tag_suffix = "not a tag"
    for d in ner_prediction:
        word, tag = list(d.items())[0]
        # if a second affiliation starts, we'll just take the first one only
        # TODO: actually return the second affiliation?
        if "SPLIT" in tag and word == "and":  # sometimes the model decides other words are split words
            break
        elif "SPLIT" in tag:
            tag = "O"
        if tag != "O":
            tag_suffix = tag.split("-")[1]

            begin = tag.startswith("B") or (previous_tag_suffix != tag_suffix)

            if word != ",":
                word_to_append = word + " "
            else:
                word_to_append = ""
            if "MAIN" in tag or "CHILD" in tag:
                main_child += word_to_append
                if "MAIN" in tag:
                    main += ("BEGIN" if begin else "") + word_to_append
                else:
                    child += ("BEGIN" if begin else "") + word_to_append
            elif "ADDRESS" in tag:
                address += ("BEGIN" if begin else "") + word_to_append

            previous_tag_suffix = tag_suffix
        else:
            previous_tag_suffix = "O"  # this is how we break up multiple continuous children

    address = [i.strip() for i in address.split("BEGIN") if len(i) > 1]

    if len(main) > 0:
        main = [i.strip() for i in main.split("BEGIN") if len(i) > 1]
        child = [i.strip() for i in child.split("BEGIN") if len(i) > 1]
    elif len(main_child) > 0:
        main = [main_child.strip()]
        child = []
    elif len(address) > 0:
        main = address
        child = []
        address = []

    # check if any of the children have a "main" word and move to main
    child_new = []
    for i in child:
        child_split = set(i.split())
        if len(child_split.intersection(CERTAINLY_MAIN)) > 0:
            main.append(i)
        else:
            child_new.append(i)

    # we have direct look-up tables for both grid names and addresses
    # here we: check the children. if any appear in grid names lookup
    # we can add to early grid candidates
    early_candidates = []
    for c in child_new:
        c = c.lower()
        if c in ror_index.ror_name_direct_lookup:
            early_candidates.extend(list(ror_index.ror_name_direct_lookup[c]))

    # if something that is a main appears in grid addresses lookup
    # but not in the direct name look up
    # it is a sign that main was possibly incorrectly extracted.
    # To mitigate, we (a) remove these from the mains and
    # (b) add the children into mains to supplement
    main_in_address = False
    mains_to_remove = set()
    for m in main:
        if m.lower() in ror_index.ror_address_counter:
            main_in_address = True
            if m.lower() not in ror_index.ror_name_direct_lookup:
                mains_to_remove.add(m)

    if main_in_address:
        main.extend(child_new)

    # if a main is in the address lookup AND not in name lookup
    # we remove it and swap into address and children. it's probably just a mislabeled address
    for m in mains_to_remove:
        main.remove(m)
        if m not in child_new:
            child_new.append(m)
        if m not in address:
            address.append(m)

    # we might not have a main still. it just becomes child or address
    if len(main) == 0:
        if len(child_new) > 0:
            main = child_new
            child_new = []
        elif len(address) > 0:
            main = address
            address = []

    # join it all together
    return main, child_new, address, early_candidates


class NERPredictor:
    """Named Entity Recognition for affiliation strings.
    Uses SimpleTransformers under the hood.
    """

    def __init__(self, model_path=PATHS["ner_model"], model_type="roberta", use_cuda=True, model=None):
        self.model_path = model_path
        self.model_type = model_type
        self.use_cuda = use_cuda
        if model is not None:
            self.model = model
        elif self.model_path is not None:
            self.load_model(self.model_path, self.model_type)
        else:
            self.model = None

    def load_model(self, model_path=PATHS["ner_model"], model_type="roberta"):
        """Load a model from disk.

        model_path (str, optional): Location of the saved NER model.
            Should be what is saved by SimpleTransformers NERModel. Defaults to PATHS["ner_model"].
        model_type (str, optional): Model type such as roberta or bert.
            If you don't know, check the config.json in the model directory. Defaults to "roberta".
        """
        self.model = NERModel(
            model_type,
            model_path,
            use_cuda=self.use_cuda,
            args={
                "use_multiprocessing": False,
                "use_multiprocessing_for_evaluation": False,
                "process_count": 1,
                "eval_batch_size": 8,
            },
        )

    def save_model(self, model_path=PATHS["ner_model"]):
        """Save model to disk

        Args:
            model_output_path (str, optional): Where to save the model.
                Defaults to PATHS["ner_model"].
        """
        self.model.save_model(output_dir=model_path, model=self.model)

    def delete_model(self):
        """Clears the model from GPU memory."""
        del self.model
        gc.collect()
        torch.cuda.empty_cache()

    def fit(
        self,
        df_train,
        df_validation=None,
        model_type="roberta",
        model_name="roberta-large",
        num_train_epochs=3,
        learning_rate=1e-5,
    ):
        """Fit the NER model.
        Uses SimpleTransformers under the hood.

        Args:
            df_train (pd.DataFrame): Training data. Assumes data has already been preprocessed
                with `fix_text` and tokenized with  `blingfire.text_to_words`, and then shaped
                into a dataframe with 3 columns: ['sentence_id', 'words', 'labels'].
            df_validation (pd.DataFrame): Same as above but for validation. None by default.
            model_type (str, optional): Model type such as roberta or bert.. Defaults to "roberta".
            model_name (str, optional): Specific model name of the model_class. Defaults to "roberta-large".
            num_train_epochs (int, optional): Number of training epochs. Defaults to 3.
            learning_rate (float, optional): Learning rate. Defaults to 1e-5.

        Returns:
            model: SimpleTransformers NERModel.
            result_vl: Validation metrics provided by SimpleTransformers.
        """
        # most of these are not part of the input params, but they could be if we wanted
        model_args = NERArgs()
        model_args.scheduler = "linear_schedule_with_warmup"
        model_args.num_train_epochs = num_train_epochs
        model_args.learning_rate = learning_rate
        model_args.hidden_dropout_prob = 0.3
        model_args.attention_probs_dropout_prob = 0.3
        model_args.evaluate_during_training = False
        model_args.reprocess_input_data = True
        model_args.overwrite_output_dir = True
        model_args.manual_seed = 4
        model_args.train_batch_size = 32
        model_args.eval_batch_size = 128
        model_args.use_multiprocessing = False
        model_args.use_multiprocessing_for_evaluation = False
        model_args.process_count = 1

        # train and save model
        custom_labels = list(set(df_train.labels))
        model = NERModel(model_type, model_name, labels=custom_labels, args=model_args, use_cuda=self.use_cuda)
        model.train_model(df_train)
        self.model = model.model

        if df_validation is not None:
            result_vl, _ = self.eval(df_validation)
        else:
            result_vl = None

        return result_vl

    def eval(self, df):
        result, model_outputs, _ = self.model.eval_model(df)
        return result, model_outputs

    def predict(self, texts):
        """Do NER on input affiliation string(s).

        Args:
            texts (str or list[str]): Affiliation string. Can be a list of strings, in which case
                the output will be a list of predictions.

        Returns:
            predictions: List of list of dicts. Looks like this if you pass in string.
                [{'Chinese': 'I-MAIN'},
                {'Academy': 'I-MAIN'},
                {'of': 'I-MAIN'},
                {'Sciences': 'I-MAIN'},
                {'PRC': 'I-ADDRESS'}]
            and will be a list of predictions if the input is a list of strings.
        """
        if type(texts) == str:
            texts = [texts]
            input_was_single_text = True
        else:
            input_was_single_text = False

        texts = [text_to_words(fix_text(text)) for text in texts]

        if self.model is None:
            raise ValueError("NERPredictor has no underlying model; load or provide one before predicting.")

        predictions = self.model.predict(texts)[0]  # [1] are the scores

        if input_was_single_text:
            return predictions[0]
        else:
            return predictions


class PairwiseRORLightGBMReranker:
    def __init__(
        self,
        ror_index,
        model_path=PATHS["lightgbm_model"],
        kenlm_model_path=PATHS["kenlm_model"],
        num_threads=0,
        booster=None,
        language_model=None,
    ):
        self.ror_index = ror_index
        self.model_path = model_path
        self.kenlm_model_path = kenlm_model_path
        if booster is not None:
            self.model = booster
        else:
            self.load_model(model_path)
        if language_model is not None:
            self.lm = language_model
        else:
            self.lm = kenlm.LanguageModel(kenlm_model_path)
        self.num_threads = num_threads

        self.inds_to_check = [
            FEATURE_NAMES.index("names_frac_of_query_matched_in_text"),
            FEATURE_NAMES.index("acronyms_frac_of_query_matched_in_text"),
        ]
        self.city_ind = FEATURE_NAMES.index("city_frac_of_query_matched_in_text")
        self.num_threads = num_threads

    def load_model(self, model_path=PATHS["lightgbm_model"]):
        """Load a model from disk.

        model_path (str, optional): Location of the saved parwise classifier.
            Should be what is saved by SimpleTransformers ClassificationModel.
            Defaults to PATHS["reranker_model"].
        model_type (str, optional): Model type such as roberta or bert.
            If you don't know, check the config.json in the model directory. Defaults to "roberta".
        """
        self.model = lgb.Booster(model_file=model_path)

    def save_model(self, model_path=PATHS["lightgbm_model"]):
        """Save model to disk

        Args:
            model_output_path (str, optional): Where to save the model.
                Defaults to PATHS["lightgbm_model"].
        """
        self.model.save_model(output_dir=model_path, model=self.model)

    def delete_model(self):
        """Clears the model from GPU memory."""
        del self.model
        gc.collect()

    def predict(self, raw_affiliation, candidates, scores):
        """
        Given a list of candidates that are ROR ids, re-rank them using the trained model.

        Args:
            raw_affiliation (str): Raw affiliation string.
            candidates (list[str]): List of candidate ROR ids.
            scores (list[float]): List of candidate scores from first stage.

        Returns:
            reranked_candidates (np.array[str]): Array of candidate ROR ids
            reranked_scores (np.array[float]): Array of candidate scores
        """
        fixed_affiliation_string = fix_text(raw_affiliation).lower().replace(",", "")
        X = []
        for i, s in zip(candidates, scores):
            ror_entry = parse_ror_entry_into_single_string_lightgbm(i, self.ror_index)
            x = make_lightgbm_features(fixed_affiliation_string, ror_entry, self.lm)
            x[-3:] = [s, int(s == -0.15), int(s == -0.1)]
            X.append(x)
        X = np.array(X)
        scores = self.model.predict(X, num_threads=self.num_threads)
        # penalty when no match across fields
        has_no_match = X[:, self.inds_to_check].sum(1) == 0
        scores -= 0.05 * has_no_match  # magic number!
        scores += 0.05 * X[:, self.city_ind]
        scores_argsort = np.argsort(scores)[::-1]
        reranked = np.vstack([np.array(candidates), scores]).T[scores_argsort]
        return reranked[:, 0], reranked[:, 1].astype(float)

    def batch_predict(self, raw_affiliations, candidates_list, scores_list):
        """
        Batch version of predict that processes multiple affiliations at once.

        This is significantly faster than calling predict() in a loop because:
        1. LightGBM can batch predict all features at once
        2. Feature extraction can be more efficient

        Args:
            raw_affiliations (list[str]): List of raw affiliation strings
            candidates_list (list[list[str]]): List of candidate lists (one per affiliation)
            scores_list (list[list[float]]): List of score lists (one per affiliation)

        Returns:
            reranked_candidates_list (list[np.array[str]]): List of reranked candidate arrays
            reranked_scores_list (list[np.array[float]]): List of reranked score arrays
        """
        # Build feature matrix for all affiliation-candidate pairs
        X_all = []
        split_indices = [0]  # Track where each affiliation's candidates start/end

        for raw_affiliation, candidates, scores in zip(raw_affiliations, candidates_list, scores_list):
            fixed_affiliation_string = fix_text(raw_affiliation).lower().replace(",", "")

            # Build features for this affiliation's candidates
            for candidate_id, score in zip(candidates, scores):
                ror_entry = parse_ror_entry_into_single_string_lightgbm(candidate_id, self.ror_index)
                x = make_lightgbm_features(fixed_affiliation_string, ror_entry, self.lm)
                x[-3:] = [score, int(score == -0.15), int(score == -0.1)]
                X_all.append(x)

            split_indices.append(len(X_all))

        # If no candidates for any affiliation, return empty lists
        if len(X_all) == 0:
            return [np.array([]) for _ in raw_affiliations], [np.array([]) for _ in raw_affiliations]

        # Batch predict all at once
        X_all = np.array(X_all)
        all_scores = self.model.predict(X_all, num_threads=self.num_threads)

        # Apply penalties
        has_no_match = X_all[:, self.inds_to_check].sum(1) == 0
        all_scores -= 0.05 * has_no_match
        all_scores += 0.05 * X_all[:, self.city_ind]

        # Split results back to each affiliation and re-rank
        reranked_candidates_list = []
        reranked_scores_list = []

        for i in range(len(raw_affiliations)):
            start_idx = split_indices[i]
            end_idx = split_indices[i + 1]

            if start_idx == end_idx:  # No candidates for this affiliation
                reranked_candidates_list.append(np.array([]))
                reranked_scores_list.append(np.array([]))
                continue

            # Get scores and candidates for this affiliation
            affiliation_scores = all_scores[start_idx:end_idx]
            affiliation_candidates = candidates_list[i]

            # Re-rank
            scores_argsort = np.argsort(affiliation_scores)[::-1]
            reranked = np.vstack([np.array(affiliation_candidates), affiliation_scores]).T[scores_argsort]

            reranked_candidates_list.append(reranked[:, 0])
            reranked_scores_list.append(reranked[:, 1].astype(float))

        return reranked_candidates_list, reranked_scores_list
