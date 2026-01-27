import json
import math
import re
from collections import Counter, defaultdict
from operator import itemgetter
from random import shuffle

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

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
from s2aff.model import parse_ner_prediction
from s2aff.text import STOPWORDS, fix_text, INVERTED_ABBREVIATION_DICTIONARY, normalize_geoname_id

ror_extractor = re.compile(r"(?:https?://)?ror\.org/(0[a-z0-9]{8})", re.I)
grid_extractor = re.compile(r"(grid\.\d{4,6}\.[0-9a-f]{1,2})")
isni_extractor = re.compile(r"(?=(\d{4}\s{0,1}\d{4}\s{0,1}\d{4}\s{0,1}[xX\d]{4}))")


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


def sum_list_of_list_of_tuples(dicts, use_log=False):
    ret = defaultdict(float)
    for d in dicts:
        for k, v in d:
            if use_log:
                ret[k] += np.log1p(v)
            else:
                ret[k] += v
    return dict(ret)




def v2_display_name(rec):
    """Extract display name from ROR v2 names array."""
    names = rec.get("names") or []
    for n in names:
        if not isinstance(n, dict):
            continue
        if "ror_display" in (n.get("types") or []):
            value = n.get("value")
            if value:
                return value
    for n in names:
        if not isinstance(n, dict):
            continue
        value = n.get("value")
        if value:
            return value
    return None


def coerce_v2_to_v1like(rec):
    """Convert ROR v2 record to a v1-shaped dict that existing code understands."""
    if "names" not in rec:  # already v1
        return rec

    rec = dict(rec)  # shallow copy

    # ---- Names
    names = rec.get("names") or []
    rec["name"] = v2_display_name(rec)
    rec["aliases"] = [n.get("value") for n in names if isinstance(n, dict) and n.get("value") and "alias" in (n.get("types") or [])]
    rec["acronyms"] = [n.get("value") for n in names if isinstance(n, dict) and n.get("value") and "acronym" in (n.get("types") or [])]
    rec["labels"] = [{"label": n.get("value")} for n in names if isinstance(n, dict) and n.get("value") and "label" in (n.get("types") or [])]

    # ---- Locations → addresses (city/state/state_code, + country_code for fallback)
    # v2 has locations[0].geonames_details.{name (locality), country_name, country_code, country_subdivision_*}
    addr = {}
    if rec.get("locations"):
        loc0 = rec["locations"][0] or {}
        g = (loc0.get("geonames_details") or {}) if isinstance(loc0, dict) else {}
        addr = {
            "city": g.get("name"),  # locality (city/town)
            "state": g.get("country_subdivision_name"),
            "state_code": g.get("country_subdivision_code"),
            # keep a v1-like field, but we can't guarantee a country geonames id from v2
            "country_geonames_id": None,
            # add for later fallback in __init__
            "country_code": g.get("country_code"),
            "country_name": g.get("country_name"),
        }
    rec["addresses"] = [addr] if addr else []

    # ---- External IDs array → v1-style dict (upper keys; robust preferred fallback)
    ed = {}
    for e in rec.get("external_ids", []):
        if not isinstance(e, dict):
            continue
        key = (e.get("type") or "").upper()
        if not key:
            continue
        all_ids = e.get("all", []) or []
        ed[key] = {
            "all": all_ids,
            "preferred": e.get("preferred", (all_ids[0] if all_ids else None)),
        }
    rec["external_ids"] = ed

    # ---- Links → wikipedia fields
    wiki_links = [l.get("value") for l in rec.get("links", []) if isinstance(l, dict) and l.get("type") == "wikipedia" and l.get("value")]
    # keep current code working:
    rec["wikipedia_page"] = wiki_links                          # list for parse_ror_entry_into_single_string
    # also provide the canonical v1-ish field if ever switching:
    rec["wikipedia_url"] = wiki_links[0] if wiki_links else None

    # ---- Relationships: title-case (so 'Child' check keeps working)
    relationships = []
    for rel in rec.get("relationships", []) or []:
        if not isinstance(rel, dict):
            continue
        rel_copy = dict(rel)
        t = rel_copy.get("type")
        if isinstance(t, str) and t:
            rel_copy["type"] = t[0].upper() + t[1:] if len(t) > 1 else t.upper()
        relationships.append(rel_copy)
    rec["relationships"] = relationships

    return rec


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
        ror_edits_path=PATHS["ror_edits"],
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
        self.ror_edits_path = cached_path(ror_edits_path)
        self.works_counts_path = cached_path(works_counts_path)
        works_counts = pd.read_csv(self.works_counts_path)
        self.works_counts = {i.ror: i.works_count for i in works_counts.itertuples()}
        self.use_prob_weights = use_prob_weights
        self.word_multiplier = word_multiplier
        self.max_intersection_denominator = max_intersection_denominator
        self.ns = ns
        self.insert_early_candidates_ind = insert_early_candidates_ind
        self.reinsert_cutoff_frac = reinsert_cutoff_frac
        self.score_based_early_cutoff = score_based_early_cutoff
        self._rust_backend = None

        self.country_codes = pd.read_csv(self.country_info_path, sep="\t")
        self.country_codes_dict = {}
        self.country_by_iso2 = {}  # NEW: for v2 fallback
        for _, r in self.country_codes.iterrows():
            geoid = normalize_geoname_id(r["geonameid"])
            if geoid:
                self.country_codes_dict[geoid] = [r["Country"], r["ISO"], r["ISO3"], r["fips"]]
            iso2 = (r["ISO"] or "").upper() if pd.notna(r["ISO"]) else ""
            if iso2:
                self.country_by_iso2[iso2] = [r["Country"], iso2, r["ISO3"], r["fips"]]

        with open(self.ror_data_path, "r") as f:
            raw = json.load(f)
            ror = [coerce_v2_to_v1like(r) for r in raw]  # normalize v2 → v1-like
            self.ror_dict = {i["id"]: i for i in ror if i.get("name")}
            for key in self.ror_dict.keys():
                self.ror_dict[key]["works_count"] = self.works_counts.get(key, 0)

        self.grid_to_ror = {}
        self.isni_to_ror = {}
        for r in ror:
            external_ids = r.get("external_ids") or {}
            grid = external_ids.get("GRID", {})
            pref = grid.get("preferred")
            if pref:
                self.grid_to_ror[pref] = r["id"]
            isni = external_ids.get("ISNI")
            if isni:
                for value in isni.get("all", []):
                    self.isni_to_ror[value.replace(' ', '')] = r["id"]

        # ROR database has some issues so we'll edit it directly
        with open(self.ror_edits_path, "r") as f:
            for line in f:
                line_json = json.loads(line)
                if line_json["ror_id"] in self.ror_dict:
                    print("Editing ROR database", line_json)
                    if line_json["action"] == "append":
                        if line_json["value"] not in self.ror_dict[line_json["ror_id"]][line_json["key"]]:
                            self.ror_dict[line_json["ror_id"]][line_json["key"]].append(line_json["value"])
                    elif line_json["action"] == "remove":
                        if line_json["value"] in self.ror_dict[line_json["ror_id"]][line_json["key"]]:
                            self.ror_dict[line_json["ror_id"]][line_json["key"]].remove(line_json["value"])

        # works_count came from OpenAlex and it has some errors
        # fixing them here by swapping works_count between parent ROR and child ROR
        # when the child ROR has suspiciously way more works_count AND the parent and child names are
        # name (location_1) and name (location_2)
        for i in range(2):  # have to do this twice because there are multi-level hierarchies
            for ror_id in self.ror_dict.keys():
                parent_works_count = self.ror_dict[ror_id]["works_count"]
                parent_name = self.ror_dict[ror_id]["name"]
                relationships = self.ror_dict[ror_id]["relationships"]
                relationships = sorted(relationships, key=lambda x: -self.ror_dict[x["id"]]["works_count"])
                for rel in relationships:
                    if rel["type"] == "Child":
                        child_id = rel["id"]
                        child_entry = self.ror_dict[child_id]
                        child_name = child_entry["name"]
                        child_works_count = child_entry["works_count"]

                        # we only care about the ones where the names are the same EXCEPT anything in parentheses
                        name_no_parens = parent_name.split("(")[0].strip()
                        child_name_no_parens = child_name.split("(")[0].strip()
                        if name_no_parens == child_name_no_parens:
                            if child_works_count > parent_works_count:
                                self.ror_dict[ror_id]["works_count"] = child_works_count
                                self.ror_dict[child_id]["works_count"] = parent_works_count
                                break  # we sorted so this swap is only necessary once

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
            if not r.get("name"):
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
                addr0 = r["addresses"][0]
                city = fix_text(addr0["city"]).lower().replace(",", "").split() or [] if addr0.get("city") else []
                state = fix_text(addr0["state"]).lower().replace(",", "").split() or [] if addr0.get("state") else []
                if addr0.get("state_code") is not None:
                    state_code = [
                        fix_text(i).lower().replace(",", "") for i in addr0["state_code"].split("-")
                    ]
                else:
                    state_code = []

                # sometimes the state code is just a number - remove it if so
                state_code = [i for i in state_code if i.isalpha()]

                # additional country names and codes (with ISO2 fallback for v2 records)
                country_and_codes = None
                geoid = addr0.get("country_geonames_id")
                geoid = normalize_geoname_id(geoid)
                if geoid and geoid in self.country_codes_dict:
                    country_and_codes = self.country_codes_dict[geoid]
                else:
                    iso2 = (addr0.get("country_code") or "").upper()
                    if iso2:
                        country_and_codes = self.country_by_iso2.get(iso2)

                if country_and_codes is None:
                    country_and_codes = ["", "", "", ""]

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

        if self.use_prob_weights and texts:
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

        # integer-indexed variants for faster retrieval
        self._typed_keys = list(inverse_dict_fixed.keys())
        self._typed_key_to_int = {k: i for i, k in enumerate(self._typed_keys)}
        self._int_to_typed_key = list(self._typed_keys)
        self._ror_id_list = list(self.ror_dict.keys())
        self._ror_id_to_int = {k: i for i, k in enumerate(self._ror_id_list)}
        self._int_to_ror_id = list(self._ror_id_list)
        self._typed_id_to_ror_id_int = np.fromiter(
            (self._ror_id_to_int[k.split("__")[0]] for k in self._typed_keys),
            dtype=np.int32,
            count=len(self._typed_keys),
        )

        self.word_index_int = {
            "inverted_index": {
                token: np.array(
                    sorted(self._typed_key_to_int[k] for k in key_set),
                    dtype=np.int32,
                )
                for token, key_set in ror_words_inverted_index.items()
            },
            "lengths_index": np.array(
                [ror_words_lengths_index.get(k, 0) for k in self._typed_keys], dtype=np.float64
            ),
        }
        self.ngram_index_int = {
            "inverted_index": {
                token: np.array(
                    sorted(self._typed_key_to_int[k] for k in key_set),
                    dtype=np.int32,
                )
                for token, key_set in ror_ngrams_inverted_index.items()
            },
            "lengths_index": np.array(
                [ror_ngrams_lengths_index.get(k, 0) for k in self._typed_keys], dtype=np.float64
            ),
        }


    def get_candidates_from_raw_affiliation(self, raw_affiliation, ner_predictor, look_for_extractable_ids=True):
        """A wrapper function that puts the raw affiliation string through the
        NER predictor, parses the predicted output, and fetches the ROR candidates.

        Args:
            raw_affiliation (str): the raw affiliation string
            ner_predictor (NERPredictor): a model that can predict named affiliation entities
            look_for_extractable_ids (bool): whether to look for GRID and ISNI IDs in the raw affiliation string
                Defalts to True. If found, just returns that candidate as the only one.

        Returns:
            candidates (list strings) - list of ROR inds in decreasing order of likelihood
            scores (list of flaots) - list of scores for each candidate

            Note that the candidates are sometimes *not* fully sorted by score because arbitrarily
            scored candidates are inserted heuristically into the list to increase recall.
        """
        if look_for_extractable_ids:
            extracted_ror = self.extract_ror(raw_affiliation)
            ror_from_grid = self.extract_grid_and_map_to_ror(raw_affiliation)
            ror_from_isni = self.extract_isni_and_map_to_ror(raw_affiliation)
            ror_from_extracted_id = extracted_ror or ror_from_grid or ror_from_isni
            if ror_from_extracted_id is not None:
                return [ror_from_extracted_id], [1.0]
            
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
            x = pd.concat(ranked_before_dfs, axis=1).fillna(0).max(axis=1)
        else:
            return [], []
        ranked_before = list(zip(x.index, x))
        tie_scale = 1e12

        def _tie_key(score):
            scaled = score * tie_scale
            if scaled >= 0:
                return int(math.floor(scaled + 0.5))
            return int(math.ceil(scaled - 0.5))

        typed_key_to_int = getattr(self, "_typed_key_to_int", None)
        ranked_before = sorted(
            ranked_before,
            key=lambda pair: (
                -_tie_key(pair[1]),
                pair[0].split("__")[0],
                typed_key_to_int.get(pair[0], 0) if typed_key_to_int is not None else 0,
            ),
        )

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
        early_candidates_tuples = []
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
                    ec_len = len(early_candidates_tuples) if self.insert_early_candidates_ind is not None else 0
                    base = self.insert_early_candidates_ind or 0
                    insert_ind = ec_len + base + 10
                    ranked_after_address_filter = (
                        ranked_after_address_filter[:insert_ind]
                        + ranked_removed_to_reinsert
                        + ranked_after_address_filter[insert_ind:]
                    )
        else:
            ranked_after_address_filter = ranked_unique

        candidates, scores = list(zip(*ranked_after_address_filter))

        return candidates, scores

    def _get_rust_backend(self):
        if self._rust_backend is not None:
            return self._rust_backend
        from s2aff.rust_backend import get_rust_backend

        self._rust_backend = get_rust_backend(self)
        return self._rust_backend

    def get_candidates_from_main_affiliation_v1(self, main, address="", early_candidates=[]):
        return self.get_candidates_from_main_affiliation(main, address, early_candidates)

    def get_candidates_from_main_affiliation_v7(self, main, address="", early_candidates=[]):
        rust_backend = self._get_rust_backend()
        if rust_backend is not None:
            address_arg = address
            if isinstance(address_arg, list):
                address_arg = " ".join(address_arg)
            return rust_backend.get_candidates_from_main_affiliation_v7(
                main, address_arg, early_candidates
            )
        address_arg = address
        if isinstance(address_arg, list):
            address_arg = " ".join(address_arg)
        return self.get_candidates_from_main_affiliation_v1(main, address_arg, early_candidates)

    def get_candidates_from_main_affiliation_v7_batch(
        self, mains, addresses=None, early_candidates_list=None
    ):
        rust_backend = self._get_rust_backend()
        if addresses is None:
            addresses = [""] * len(mains)
        if early_candidates_list is None:
            early_candidates_list = [[] for _ in mains]
        if len(addresses) != len(mains):
            raise ValueError("addresses length mismatch")
        if len(early_candidates_list) != len(mains):
            raise ValueError("early_candidates_list length mismatch")
        if rust_backend is None:
            candidates_list = []
            scores_list = []
            for main, address, early in zip(mains, addresses, early_candidates_list):
                address_arg = address
                if isinstance(address_arg, list):
                    address_arg = " ".join(address_arg)
                candidates, scores = self.get_candidates_from_main_affiliation_v1(
                    main, address_arg, early
                )
                candidates_list.append(list(candidates))
                scores_list.append(list(scores))
            return candidates_list, scores_list
        return rust_backend.get_candidates_from_main_affiliation_v7_batch(
            mains, addresses, early_candidates_list
        )

    def extract_ror(self, s):
        match = ror_extractor.search(s)
        if not match:
            return None
        ror_suffix = match.group(1)
        if not ror_suffix:
            return None
        return f"https://ror.org/{ror_suffix.lower()}"

    def extract_grid_and_map_to_ror(self, s):
        extracted_grids = grid_extractor.findall(s)
        if len(extracted_grids) == 0:
            return None
        else:
            extracted_grid_1 = extracted_grids[0]
            extracted_ror_1 = self.grid_to_ror.get(extracted_grid_1, None)
            if len(extracted_grid_1.split('.')[-1]) == 2:
                extracted_grid_2 = extracted_grid_1[:-1]
                extracted_ror_2 = self.grid_to_ror.get(extracted_grid_2, None)
                if extracted_ror_1 and extracted_ror_2:
                    return None
                else:
                    return extracted_ror_1 or extracted_ror_2
            else:
                return extracted_ror_1

    def extract_isni_and_map_to_ror(self, s):
        extracted_isnis = isni_extractor.findall(s)
        # look up all extracted ISNIs and return the ror if exactly one ROR is found
        found_rors = [self.isni_to_ror.get(isni.replace(' ', ''), None) for isni in extracted_isnis]
        filtered_rors = [x for x in found_rors if x is not None]
        if len(filtered_rors) == 1:
            return filtered_rors[0]
        else:
            return None



def parse_ror_entry_into_single_string(
    ror_entry,
    country_codes_dict,
    country_by_iso2,
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
        addr0 = ror_entry["addresses"][0]
        city = addr0.get("city") or ""
        state = addr0.get("state") or ""
        if addr0.get("state_code") is not None:
            state_code = addr0["state_code"].split("-")
        else:
            state_code = []
        state_code = [i for i in state_code if i.isalpha()]

        # Country lookup with ISO2 fallback for v2 records
        country_and_codes = None
        geoid = normalize_geoname_id(addr0.get("country_geonames_id"))
        if geoid and geoid in country_codes_dict:
            country_and_codes = country_codes_dict[geoid]
        else:
            iso2 = (addr0.get("country_code") or "").upper()
            if iso2:
                country_and_codes = country_by_iso2.get(iso2)

        if country_and_codes is None:
            country_and_codes = ["", "", "", ""]

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
