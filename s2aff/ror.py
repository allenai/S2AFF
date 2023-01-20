import json
from collections import Counter, defaultdict
from operator import itemgetter
from random import shuffle

import numpy as np
import pandas as pd
from random import shuffle
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict, Counter
from s2aff.file_cache import cached_path
from operator import itemgetter
from s2aff.text import STOPWORDS, fix_text, INVERTED_ABBREVIATION_DICTIONARY
from s2aff.model import parse_ner_prediction
from s2aff.consts import (
    INSERT_EARLY_CANDIDATES_IND,
    MAX_INTERSECTION_DENOMINATOR,
    NS,
    PATHS,
    USE_PROB_WEIGHTS,
    WORD_MULTIPLIER,
    REINSERT_CUTOFF_FRAC,
    SCORE_BASED_EARLY_CUTOFF,
)
from s2aff.file_cache import cached_path


def get_special_tokens_dict():
    special_tokens_dict = {
        "bos_token": "<s>",
        "eos_token": "</s>",
        "unk_token": "<unk>",
        "sep_token": "</s>",
        "pad_token": "<pad>",
        "cls_token": "<s>",
        "mask_token": "<mask>",
    }
    special_tokens_dict.update({f"additional_special_tokens": [f"<reserved-{i}>" for i in range(10)]})
    special_tokens_dict.update(
        {
            "[NAME]": special_tokens_dict["additional_special_tokens"][0],
            "[ACRONYM]": special_tokens_dict["additional_special_tokens"][1],
            "[ADDRESS]": special_tokens_dict["additional_special_tokens"][2],
            "[WIKI]": special_tokens_dict["additional_special_tokens"][3],
        }
    )
    return special_tokens_dict


def index_min(l, j_set):
    found_indices = []
    l_set = set(l)
    for j in j_set:
        if j in l_set:
            found_indices.append(l.index(j))
    if len(found_indices) > 0:
        return min(found_indices)
    else:
        return 100000  # approximate # of ROR elements


def sum_list_of_list_of_tuples(dicts, use_log=False):
    ret = defaultdict(float)
    for d in dicts:
        for k, v in d:
            if use_log:
                ret[k] += np.log1p(v)
            else:
                ret[k] += v
    return dict(ret)


class RORIndex:
    """An index for ROR (Research Organization Registry) entities.
    This can be used to find the most likely ROR entity for a given affiliation,
    as well as parse the predictions of the NER model.

    See https://ror.org/ for more details.

    Args:
        ror_data_path (str, optional): Location of the ROR data dump.
            Defaults to PATHS["ror_data"].
        country_info_path (str, optional): Location of additional data about countries
            that ROR does not have. Defaults to PATHS["country_info"].
        use_prob_weights (str, optional): Whether to use inverse word probability weights
            when using the Jaccard retrieval. Will effectively down-weight common affiliation words.
            Defaults to USE_PROB_WEIGHTS.
        word_multiplier (str, optional): The weight in
            word_weight * word_jaccard_score + ngram_jaccard_score. Defaults to WORD_MULTIPLIER.
        max_intersection_denominator (str, optional): Whether to modify how the denominator is
            calculated for Jaccard similarity. If true, only restrics the denominator to
            the set of words/ngrams found across all candidates. Defaults to MAX_INTERSECTION_DENOMINATOR.
        ns (set, optional): The set of ngram sizes to use when generating ngrams. Can be any combo of 3, 4, 5
            Defaults to {3}
    """

    def __init__(
        self,
        ror_data_path=PATHS["ror_data"],
        country_info_path=PATHS["country_info"],
        works_counts_path=PATHS["openalex_works_counts"],
        use_prob_weights=USE_PROB_WEIGHTS,
        word_multiplier=WORD_MULTIPLIER,
        max_intersection_denominator=MAX_INTERSECTION_DENOMINATOR,
        ns=NS,
        insert_early_candidates_ind=INSERT_EARLY_CANDIDATES_IND,
        reinsert_cutoff_frac=REINSERT_CUTOFF_FRAC,
        score_based_early_cutoff=SCORE_BASED_EARLY_CUTOFF,
    ):

        self.ror_data_path = cached_path(ror_data_path)
        self.country_info_path = cached_path(country_info_path)
        works_counts = pd.read_csv(works_counts_path)
        self.works_counts = {i.ror: i.works_count for i in works_counts.itertuples()}
        self.use_prob_weights = use_prob_weights
        self.word_multiplier = word_multiplier
        self.max_intersection_denominator = max_intersection_denominator
        self.ns = ns
        self.insert_early_candidates_ind = insert_early_candidates_ind
        self.reinsert_cutoff_frac = reinsert_cutoff_frac
        self.score_based_early_cutoff = score_based_early_cutoff

        self.country_codes = pd.read_csv(self.country_info_path, sep="\t")
        self.country_codes_dict = {}
        for _, r in self.country_codes.iterrows():
            self.country_codes_dict[r["geonameid"]] = [r["Country"], r["ISO"], r["ISO3"], r["fips"]]

        with open(self.ror_data_path, "r") as f:
            ror = json.load(f)
            self.ror_dict = {i["id"]: i for i in ror if "name" in i}
            for key in self.ror_dict.keys():
                self.ror_dict[key]["works_count"] = self.works_counts.get(key, 0)

        # we need some indices for fetching first order candidates
        ror_ngrams_inverted_index = defaultdict(set)
        ror_ngrams_lengths_index = {}

        ror_words_inverted_index = defaultdict(set)
        ror_words_lengths_index = {}

        ror_address_inverted_index = defaultdict(set)
        ror_city_inverted_index = defaultdict(set)
        ror_address_counter = Counter()

        inverse_dict_fixed = {}
        ror_name_direct_lookup = defaultdict(set)

        texts = []

        for r in ror:
            if "name" not in r:
                continue

            ror_id = r["id"]
            official_name = fix_text(r["name"]).lower().replace(",", "")
            aliases = [fix_text(i).lower().replace(",", "") for i in r["aliases"]]
            labels = [
                fix_text(i["label"]).lower().replace(",", "") for i in r["labels"]
            ]  # e.g. Sorbonne University has a label of Sorbonne Universités
            acronyms = [fix_text(i).lower().replace(",", "") for i in r["acronyms"]]
            all_names = [official_name] + aliases + labels + acronyms

            for name in all_names:
                if len(name) > 0:
                    ror_name_direct_lookup[name].add(ror_id)

            types_of_names = (
                ["official_name"]
                + [f"alias_{i}" for i in range(len(aliases))]
                + [f"label_{i}" for i in range(len(labels))]
                + [f"acronym_{i}" for i in range(len(acronyms))]
            )

            # this is the text we'll use later for the IDF weights
            text = " ".join(all_names)
            texts.append(text)

            for name, type_of_name in zip(all_names, types_of_names):
                ror_id_with_type = ror_id + "__" + type_of_name

                # get rid of stopwords
                name = " ".join([i for i in name.split(" ") if i not in STOPWORDS])
                inverse_dict_fixed[ror_id_with_type] = name

            # address
            if "addresses" in r and len(r["addresses"]) > 0:
                city = fix_text(r["addresses"][0]["city"]).lower().replace(",", "").split() or []
                state = fix_text(r["addresses"][0]["state"]).lower().replace(",", "").split() or []
                if r["addresses"][0]["state_code"] is not None:
                    state_code = [
                        fix_text(i).lower().replace(",", "") for i in r["addresses"][0]["state_code"].split("-")
                    ]
                else:
                    state_code = []

                # sometimes the state code is just a number - remove it if so
                state_code = [i for i in state_code if i.isalpha()]

                # additional country names and codes
                country_and_codes = self.country_codes_dict[r["addresses"][0]["country_geonames_id"]]
                if country_and_codes[0] == "China":
                    extras = ["pr", "prc"]
                else:
                    extras = []

                country_and_codes = [fix_text(i).lower().replace(",", "") for i in country_and_codes]

                # fix them up
                fixed_address_elements = set(city + state + country_and_codes + extras + state_code)
                fixed_address_elements = [i for i in fixed_address_elements if len(i) > 1 and i not in STOPWORDS]

                ror_address_counter.update([i for i in fixed_address_elements if len(i) > 3])

                for i in fixed_address_elements:
                    ror_address_inverted_index[i].add(ror_id)

                for i in city:
                    ror_city_inverted_index[i].add(ror_id)

        if self.use_prob_weights:
            vectorizer = TfidfVectorizer(
                min_df=1, analyzer="word", tokenizer=None, preprocessor=None, lowercase=False
            ).fit(texts)
            self.idf_lookup = {
                term: vectorizer.idf_[ind] for i, (term, ind) in enumerate(vectorizer.vocabulary_.items())
            }
            self.idf_lookup_min = min(self.idf_lookup.values())

            self.idf_weight = lambda i: self.idf_lookup.get(i, self.idf_lookup_min)
        else:
            self.idf_weight = None

        # make the indices
        for ror_id_with_type, name in inverse_dict_fixed.items():
            # ngrams
            name_ngrams, name_ngrams_weights = get_text_ngrams(name, weights_lookup_f=self.idf_weight, ns=self.ns)

            # one issue is when the same ngram appears in a long affiliation string
            # many times, it ends up being overweighted
            # hack mitigation: we keep each ngram only once, but the weight
            # is its max weight
            candidate_ngrams_unique = set()
            ngram_largest_weight = {}
            for ngram, weight in zip(name_ngrams, name_ngrams_weights):
                if ngram not in candidate_ngrams_unique:
                    candidate_ngrams_unique.add(ngram)
                    ngram_largest_weight[ngram] = np.maximum(weight, ngram_largest_weight.get(ngram, 0))

            if self.use_prob_weights:
                ror_ngrams_lengths_index[ror_id_with_type] = np.sum(
                    [ngram_largest_weight[i] for i in candidate_ngrams_unique]
                )
            else:
                ror_ngrams_lengths_index[ror_id_with_type] = len(candidate_ngrams_unique)

            for i in name_ngrams:
                ror_ngrams_inverted_index[i].add(ror_id_with_type)

            # words
            name_unigrams = set(name.split(" "))
            if self.use_prob_weights:
                ror_words_lengths_index[ror_id_with_type] = np.sum([self.idf_weight(i) for i in name_unigrams])
            else:
                ror_words_lengths_index[ror_id_with_type] = len(name_unigrams)
            for i in name_unigrams:
                ror_words_inverted_index[i].add(ror_id_with_type)

        self.word_index = {"inverted_index": dict(ror_words_inverted_index), "lengths_index": ror_words_lengths_index}
        self.ngram_index = {
            "inverted_index": dict(ror_ngrams_inverted_index),
            "lengths_index": ror_ngrams_lengths_index,
        }
        self.address_index = {
            "full_index": dict(ror_address_inverted_index),
            "city_index": dict(ror_city_inverted_index),
        }
        self.ror_address_counter = ror_address_counter
        self.ror_name_direct_lookup = ror_name_direct_lookup
        self.inverse_dict_fixed = inverse_dict_fixed

    def get_candidates_from_raw_affiliation(self, raw_affiliation, ner_predictor):
        """A wrapper function that puts the raw affiliation string through the
        NER predictor, parses the predicted output, and fetches the ROR candidates.

        Args:
            raw_affiliation (str): the raw affiliation string
            ner_predictor (NERPredictor): a model that can predict named affiliation entities

        Returns:
            candidates (list strings) - list of ROR inds in decreasing order of likelihood
            scores (list of flaots) - list of scores for each candidate

            Note that the candidates are sometimes *not* fully sorted by score because arbitrarily
            scored candidates are inserted heuristically into the list to increase recall.
        """
        ner_prediction = ner_predictor.predict([raw_affiliation])
        main, child, address, early_candidates = parse_ner_prediction(ner_prediction[0], self)
        candidates, scores = self.get_candidates_from_main_affiliation(main, address, early_candidates)
        return candidates, scores

    def get_candidates_from_main_affiliation(self, main, address="", early_candidates=[]):
        """Get ROR candidates via ngram and token Jaccard similarity.
        If you have a raw affiliation string, this is how to use this function:

        ```
        # instantiate NERPredictor and RORIndex here
        raw_affiliation = "Department of Philosophy, University of California, Berkeley, CA, USA"
        ner_prediction = ner_predictor.predict(raw_affiliation)
        main, child, address, early_candidates = parse_ner_prediction(ner_prediction, ror_index)
        candidates, scores = ror_index.get_candidates_from_main_affiliation(main, address, early_candidates)
        ```
        You can also use a convenience function as follows:
        ```
        candidates, scores = ror_index.get_candidates_from_raw_affiliation(raw_affiliation, ner_predictor)
        ```

        Args:
            main (str or list[str]): The string for the "main" part of the affiliation.
                For example, if the raw affiliation string is as above,
                then `main` should be "University of California".
                Note that this function expects that main and address have gone
                through `fix_text` but have not had stopwords removed nor are lowercased
            address (str, optional): The address part of the raw affiliation string.
                From above example, it would be "Berkeley, CA, USA". Defaults to "".
            early_candidates (list, optional): Some early candidate from parsing the
                raw affiliation string. Defaults to [].

        Returns:
            candidates (list strings) - list of ROR inds in decreasing order of likelihood
            scores (list of flaots) - list of scores for each candidate

            Note that the candidates are sometimes *not* fully sorted by score because arbitrarily
            scored candidates are inserted heuristically into the list to increase recall.
        """
        if type(main) is str:
            main = [main]
        ranked_before_dfs = []
        main_fixed_all = []
        for m in main:
            main_fixed = [i for i in m.lower().replace(",", "").split(" ") if i not in STOPWORDS]
            main_fixed = " ".join([INVERTED_ABBREVIATION_DICTIONARY.get(word, word) for word in main_fixed])
            main_fixed_all.append(main_fixed)
            if len(main_fixed) > 0:
                ranked_words_before = jaccard_word_nns(
                    main_fixed,
                    self.word_index,
                    idf_weight=self.idf_weight,
                    max_intersection_denominator=self.max_intersection_denominator,
                    word_multiplier=self.word_multiplier,
                    sort=False,
                )
                ranked_ngrams_before = jaccard_ngram_nns(
                    main_fixed,
                    self.ngram_index,
                    ns=self.ns,
                    idf_weight=self.idf_weight,
                    max_intersection_denominator=self.max_intersection_denominator,
                    sort=False,
                )
                ranked_before = sum_list_of_list_of_tuples(
                    [ranked_words_before, ranked_ngrams_before], use_log=self.word_multiplier == "log"
                )
                if len(ranked_before) > 0:
                    ranked_before_df = pd.DataFrame(ranked_before.items()).set_index(0)
                    if self.word_multiplier == "log":
                        ranked_before_df = ranked_before_df[
                            ranked_before_df[1] > np.log1p(self.score_based_early_cutoff)
                        ]
                    else:
                        ranked_before_df = ranked_before_df[ranked_before_df[1] > self.score_based_early_cutoff]
                    ranked_before_dfs.append(ranked_before_df)
        if len(ranked_before_dfs) > 0:
            x = pd.concat(ranked_before_dfs, axis=1).fillna(0).max(1)
        else:
            return [], []
        ranked_before = list(zip(x.index, x))
        ranked_before = sorted(ranked_before, key=itemgetter(1), reverse=True)

        # we have multiple appearances of the same grid id
        # keep only the first appearance in the ranked list
        ranked_unique = []
        seen_rors = set()
        for ror_id, score in ranked_before:
            ror_id = ror_id.split("__")[0]
            if ror_id not in seen_rors:
                ranked_unique.append((ror_id, score))
                seen_rors.add(ror_id)

        # insert early candidates in position self.insert_early_candidates_ind if not seen already
        if self.insert_early_candidates_ind is not None:
            early_candidates_tuples = [(i, -0.1) for i in early_candidates if i not in seen_rors]
            ranked_unique = (
                ranked_unique[: self.insert_early_candidates_ind]
                + early_candidates_tuples
                + ranked_unique[self.insert_early_candidates_ind :]
            )

        if len(ranked_unique) == 0:
            return [], []

        # if we have address text, we can find the RORs that have at least one word in that text
        if address != "" and len(address) > 0:
            if type(address) is list:
                address = " ".join(address)
            address_fixed_tokens = set([i for i in address.lower().replace(",", "").split(" ") if i not in STOPWORDS])

            inverted = self.address_index["full_index"]
            acceptable_rors = set()
            for unigram in address_fixed_tokens:
                if unigram in inverted:
                    acceptable_rors.update(inverted[unigram])

            ranked_after_address_filter = [i for i in ranked_unique if i[0] in acceptable_rors]

            if len(ranked_after_address_filter) == 0:
                ranked_after_address_filter = ranked_unique
            else:
                # reinsert top from ranked_unique that were removed, just in case the address was wrong
                ranked_removed = [i for i in ranked_unique if i[0] not in acceptable_rors]
                if len(ranked_removed) > 0:
                    # reinsert the ones that all are close to top score
                    ranked_removed_top_score = self.reinsert_cutoff_frac * ranked_after_address_filter[0][1]
                    # artificially setting the score to distinguish from "real" top
                    ranked_removed_to_reinsert = [
                        (i[0], -0.15) for i in ranked_removed if i[1] >= ranked_removed_top_score
                    ]
                    if self.insert_early_candidates_ind is not None:
                        insert_ind = len(early_candidates_tuples) + self.insert_early_candidates_ind + 10
                    else:
                        insert_ind = len(early_candidates_tuples) + 10
                    ranked_after_address_filter = (
                        ranked_after_address_filter[:insert_ind]
                        + ranked_removed_to_reinsert
                        + ranked_after_address_filter[insert_ind:]
                    )
        else:
            ranked_after_address_filter = ranked_unique

        candidates, scores = list(zip(*ranked_after_address_filter))

        return candidates, scores

    def parse_ror_entry_into_single_string(
        self, ror_id, use_separator_tokens=True, shuffle_names=False, special_tokens_dict=get_special_tokens_dict()
    ):
        if ror_id not in self.ror_dict:
            # assuming that this is just not a ror_id and instead a string of some affiliation
            return "[NAME] " + ror_id
        ror_entry = self.ror_dict[ror_id]
        return parse_ror_entry_into_single_string(
            ror_entry, self.country_codes_dict, special_tokens_dict, use_separator_tokens, shuffle_names, use_wiki=False
        )


def parse_ror_entry_into_single_string(
    ror_entry,
    country_codes_dict,
    special_tokens_dict,
    use_separator_tokens=True,
    shuffle_names=False,
    use_wiki=True,
):

    official_name = fix_text(ror_entry["name"])
    aliases = [fix_text(i) for i in ror_entry["aliases"]]
    labels = [
        fix_text(i["label"]) for i in ror_entry["labels"]
    ]  # e.g. Sorbonne University has a label of Sorbonne Universités
    acronyms = [fix_text(i) for i in ror_entry["acronyms"]]
    all_names = [official_name] + aliases + labels + acronyms

    if use_separator_tokens:
        types_of_names = (
            [f"{special_tokens_dict.get('[NAME]')}"]
            + [f"{special_tokens_dict.get('[NAME]')}" for i in range(len(aliases))]
            + [f"{special_tokens_dict.get('[NAME]')}" for i in range(len(labels))]
            + [f"{special_tokens_dict.get('[ACRONYM]')}" for i in range(len(acronyms))]
        )

        names_and_types = list(zip(all_names, types_of_names))
        if shuffle_names:
            shuffle(names_and_types)
        output = " ".join(f"{b} {a}" for a, b in names_and_types)
    else:
        if shuffle_names:
            shuffle(all_names)
        output = " ; ".join(all_names)

    if "addresses" in ror_entry and len(ror_entry["addresses"]) > 0:
        city = ror_entry["addresses"][0]["city"] or ""
        state = ror_entry["addresses"][0]["state"] or ""
        if ror_entry["addresses"][0]["state_code"] is not None:
            state_code = ror_entry["addresses"][0]["state_code"].split("-")
        else:
            state_code = []
        state_code = [i for i in state_code if i.isalpha()]
        country_and_codes = country_codes_dict[ror_entry["addresses"][0]["country_geonames_id"]]
        if country_and_codes[0] == "China":
            extras = ["PR", "PRC"]
        else:
            extras = []
        fixed_address_elements = []
        seen = set()
        for i in [city, state] + state_code + country_and_codes + extras:
            i_fix = fix_text(i)
            if i_fix not in seen and len(i_fix) > 0:
                fixed_address_elements.append(i_fix)
                seen.add(i_fix)

        address = " ".join(fixed_address_elements)
        if use_separator_tokens:
            output += f" {special_tokens_dict.get('[ADDRESS]')} {address}"
        else:
            output += f" ; {address}"

    if "wikipedia_page" in ror_entry and use_wiki:
        output += f" {special_tokens_dict.get('[WIKI]')} {' '.join(ror_entry['wikipedia_page'])}"

    return output


def jaccard_ngram_nns(
    candidate,
    ngram_index,
    ns=NS,
    idf_weight=None,
    max_intersection_denominator=MAX_INTERSECTION_DENOMINATOR,
    sort=False,
):

    lengths = ngram_index["lengths_index"]
    inverted = ngram_index["inverted_index"]

    intersections = Counter()
    candidate_ngrams, candidate_ngram_weights = get_text_ngrams(candidate, weights_lookup_f=idf_weight, ns=ns)

    # one issue is when the same ngram appears in a long affiliation string
    # many times, it ends up being overweighted
    # hack mitigation: we keep each ngram only once, but the weight
    # is its max weight
    candidate_ngrams_unique = set()
    ngram_largest_weight = {}
    for ngram, weight in zip(candidate_ngrams, candidate_ngram_weights):
        if ngram not in candidate_ngrams_unique:
            candidate_ngrams_unique.add(ngram)
            ngram_largest_weight[ngram] = np.maximum(weight, ngram_largest_weight.get(ngram, 0))

    intersections = defaultdict(float)
    candidate_ngrams_in_inverted = []
    weights_in_inverted = []
    for ng in candidate_ngrams_unique:
        if ng in inverted:
            weight = ngram_largest_weight[ng]
            for ror_id in inverted[ng]:
                intersections[ror_id] += weight
            weights_in_inverted.append(weight)
            candidate_ngrams_in_inverted.append(ng)

    if idf_weight is not None:
        num_candidate_ngrams = np.sum(candidate_ngram_weights)
        num_relevant_candidate_ngrams = np.sum(weights_in_inverted)
    else:
        num_candidate_ngrams = len(candidate_ngrams)
        num_relevant_candidate_ngrams = len(candidate_ngrams_in_inverted)

    if max_intersection_denominator:
        jaccards = [
            (grid, intersection / (num_relevant_candidate_ngrams + lengths[grid] - intersection))
            for grid, intersection in intersections.items()
        ]
    else:
        jaccards = [
            (grid, intersection / (num_candidate_ngrams + lengths[grid] - intersection))
            for grid, intersection in intersections.items()
        ]

    if sort:
        jaccards = sorted(jaccards, key=itemgetter(1), reverse=True)

    return jaccards


def jaccard_word_nns(
    candidate,
    word_index,
    idf_weight=None,
    max_intersection_denominator=MAX_INTERSECTION_DENOMINATOR,
    word_multiplier=WORD_MULTIPLIER,
    sort=False,
):
    if word_multiplier == "log":
        word_multiplier = 1

    lengths = word_index["lengths_index"]
    inverted = word_index["inverted_index"]

    candidate_unigrams = set(candidate.split())

    use_prob_weights = idf_weight is not None

    if use_prob_weights:
        unigram_inv_probs = {i: idf_weight(i) for i in candidate_unigrams}

    intersections = defaultdict(float)
    all_matched_unigrams = set()
    for unigram in candidate_unigrams:
        if unigram in inverted:
            if use_prob_weights:
                weight = unigram_inv_probs[unigram]
            else:
                weight = 1
            for ror_id in inverted[unigram]:
                intersections[ror_id] += weight
            all_matched_unigrams.add(unigram)

    if use_prob_weights:
        num_candidate_unigrams = np.sum(list(unigram_inv_probs.values()))
        num_relevant_candidate_unigrams = np.sum([unigram_inv_probs[i] for i in all_matched_unigrams])
    else:
        num_candidate_unigrams = len(candidate_unigrams)
        num_relevant_candidate_unigrams = len(all_matched_unigrams)

    if max_intersection_denominator:
        jaccards = [
            (
                ror_id,
                word_multiplier * intersection / (num_relevant_candidate_unigrams + lengths[ror_id] - intersection),
            )
            for ror_id, intersection in intersections.items()
        ]
    else:
        jaccards = [
            (ror_id, word_multiplier * intersection / (num_candidate_unigrams + lengths[ror_id] - intersection))
            for ror_id, intersection in intersections.items()
        ]

    if sort:
        jaccards = sorted(jaccards, key=itemgetter(1), reverse=True)

    return jaccards


def get_text_ngrams(text, weights_lookup_f=None, ns={3, 4, 5}):
    """
    Get character bigrams, trigrams, quadgrams, and optionally unigrams for a piece of text.
    Note: respects word boundaries

    Parameters
    ----------
    text: string
        the text to get ngrams for
    use_unigrams: bool
        whether or not to include unigrams
    stopwords: Set
        The set of stopwords to filter out before computing character ngrams

    Returns
    -------
    Counter: the ngrams present in the text
    """
    if text is None or len(text) == 0:
        return [], []

    words = text.split(" ")

    if weights_lookup_f is not None:
        unigram_inv_probs = [weights_lookup_f(i) for i in words]
    else:
        unigram_inv_probs = [1] * len(words)

    text = " ".join(words)

    ngrams = []
    weights = []

    if 3 in ns:
        trigrams = map(
            lambda x: "".join(x),
            filter(lambda x: " " not in x, zip(text, text[1:], text[2:])),
        )
        trigram_weights = sum(
            [[j / max(len(i) - 2, 1e-6)] * (len(i) - 2) for i, j in zip(words, unigram_inv_probs)], []
        )
        ngrams += list(trigrams)
        weights += trigram_weights

    if 4 in ns:
        quadgrams = map(
            lambda x: "".join(x),
            filter(lambda x: " " not in x, zip(text, text[1:], text[2:], text[3:])),
        )
        quadgram_weights = sum(
            [[j / max(len(i) - 3, 1e-6)] * (len(i) - 3) for i, j in zip(words, unigram_inv_probs)], []
        )
        ngrams += list(quadgrams)
        weights += quadgram_weights

    if 5 in ns:
        fivegrams = map(
            lambda x: "".join(x),
            filter(lambda x: " " not in x, zip(text, text[1:], text[2:], text[3:], text[4:])),
        )
        fivegram_weights = sum(
            [[j / max(len(i) - 4, 1e-6)] * (len(i) - 4) for i, j in zip(words, unigram_inv_probs)], []
        )
        ngrams += list(fivegrams)
        weights += fivegram_weights

    return ngrams, weights
