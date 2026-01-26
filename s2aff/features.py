import re
import numpy as np
from nltk.util import ngrams
from s2aff.text import fix_text, STOPWORDS, normalize_geoname_id


def parse_ror_entry_into_single_string_lightgbm(ror_id_or_other_affiliation, ror_index):
    if ror_id_or_other_affiliation not in ror_index.ror_dict:
        # assuming that this is just not a ror_id and instead a string of some affiliation
        return {
            "names": [ror_id_or_other_affiliation],
            "acronyms": [],
            "city": [],
            "state": [],
            "country": [],
            "works_count": 0,
        }
    else:
        # otherwise it's a ROR id
        ror_id = ror_id_or_other_affiliation

    ror_entry = ror_index.ror_dict[ror_id]
    official_name = fix_text(ror_entry["name"]).lower().replace(",", "")
    aliases = [fix_text(i).lower().replace(",", "") for i in ror_entry["aliases"]]
    labels = [
        fix_text(i["label"]).lower().replace(",", "") for i in ror_entry["labels"]
    ]  # e.g. Sorbonne University has a label of Sorbonne Universités
    acronyms = [fix_text(i).lower().replace(",", "") for i in ror_entry["acronyms"]]

    output = {"names": list(set([official_name] + aliases + labels)), "acronyms": acronyms}

    if "addresses" in ror_entry and len(ror_entry["addresses"]) > 0:
        city = fix_text(ror_entry["addresses"][0]["city"] or "").lower().replace(",", "")

        #
        state = fix_text(ror_entry["addresses"][0]["state"] or "").lower().replace(",", "")

        if ror_entry["addresses"][0]["state_code"] is not None:
            state_code = ror_entry["addresses"][0]["state_code"].split("-")
        else:
            state_code = []

        # sometimes the state code is just a number - remove it if so
        state_code = [fix_text(i).lower().replace(",", "") for i in state_code if i.isalpha()]

        # country (ensure geonames id fallback to ISO2 for ROR v2 records)
        addr0 = ror_entry["addresses"][0]
        country_and_codes = None
        geoid = normalize_geoname_id(addr0.get("country_geonames_id"))
        if geoid and geoid in ror_index.country_codes_dict:
            country_and_codes = ror_index.country_codes_dict[geoid]
        else:
            iso2 = (addr0.get("country_code") or "").upper()
            if iso2:
                country_and_codes = ror_index.country_by_iso2.get(iso2)

        if country_and_codes is None:
            country_and_codes = ["", "", "", ""]

        if country_and_codes[0] == "China":
            extras = ["PR", "PRC"]
        else:
            extras = []

        seen = set()
        country_elements = []
        for i in country_and_codes + extras:
            i_fix = fix_text(i).lower().replace(",", "")
            if i_fix not in seen and len(i_fix) > 0:
                country_elements.append(i_fix)
                seen.add(i_fix)

        state_info = [state] + state_code
        # sometimes the country code ends up being a state code
        state_info = [i for i in state_info if i not in seen]
        # sometimes the state is also a city
        state_info = [i for i in state_info if i != city]

        output["city"] = [city]
        output["state"] = state_info
        output["country"] = country_elements
    else:
        output["city"] = []
        output["state"] = []
        output["country"] = []

    output["works_count"] = ror_entry["works_count"]

    return output


def split(delimiters, string, maxsplit=0):
    regexPattern = "|".join(map(re.escape, delimiters))
    return re.split(regexPattern, string, maxsplit)


def build_query_ngrams(q, max_ngram_len=7):
    q_split = q.split()
    n_grams = []
    longest_ngram = np.minimum(max_ngram_len, len(q_split))
    for i in range(int(longest_ngram), 0, -1):
        n_grams += [" ".join(ngram).replace("|", "\|") for ngram in ngrams(q_split, i)]
    return n_grams


def build_query_context(query, max_q_len=256, use_word_boundaries=False, max_ngram_len=7):
    q = str(query)[:max_q_len]
    if len(q) == 0:
        return {"q": "", "q_split_set": set(), "q_set_len": 1, "regex": None, "use_word_boundaries": use_word_boundaries}
    q_split_set = set(q.split()) - STOPWORDS
    q_set_len = len(" ".join(q_split_set))
    q_set_len = np.maximum(q_set_len, 1)

    n_grams = build_query_ngrams(q, max_ngram_len=max_ngram_len)
    if len(n_grams) == 0:
        regex = None
    else:
        if use_word_boundaries:
            pattern = "|".join(["\\b" + i + "\\b" for i in n_grams])
        else:
            pattern = "|".join(n_grams)
        regex = re.compile(pattern)

    return {
        "q": q,
        "q_split_set": q_split_set,
        "q_set_len": q_set_len,
        "regex": regex,
        "use_word_boundaries": use_word_boundaries,
    }


def find_query_ngrams_in_text_precompiled(t, regex, len_filter=1, remove_stopwords=True):
    if regex is None or len(t) == 0:
        return [], []
    if type(t) is not str:
        return [], []

    matches = list(regex.finditer(t))
    match_spans = [i.span() for i in matches if i.span()[1] - i.span()[0] > len_filter]
    match_text_tokenized = [i.group() for i in matches if i.span()[1] - i.span()[0] > len_filter]

    if remove_stopwords:
        match_spans = [span for i, span in enumerate(match_spans) if match_text_tokenized[i] not in STOPWORDS]
        match_text_tokenized = [text for text in match_text_tokenized if text not in STOPWORDS]

    return match_spans, match_text_tokenized


def find_query_ngrams_in_text(q, t, len_filter=1, remove_stopwords=True, use_word_boundaries=False, max_ngram_len=7):
    """A function to find instances of ngrams of query q
    inside text t. Finds all possible ngrams and returns their
    character-level span.

    Note: because of the greedy match this function can miss
    some matches when there's repetition in the query, but
    this is likely rare enough that we can ignore it

    Arguments:
        q {str} -- query
        t {str} -- text to find the query within
        len_filter {int} -- shortest allowable matches in characters
        remove_stopwords {bool} -- whether to remove stopwords-only matches
        use_word_boundaries {bool} -- whether to care about word boundaries
                                      when finding matches
        max_ngram_len {int} -- longest allowable derived word n-grams

    Returns:
        match_spans -- a list of span tuples
        match_text_tokenized -- a list of matched tokens
    """

    if len(q) == 0 or len(t) == 0:
        return [], []
    if type(q[0]) is not str or type(t) is not str:
        return [], []

    n_grams = build_query_ngrams(q, max_ngram_len=max_ngram_len)

    if use_word_boundaries:
        matches = list(re.finditer("|".join(["\\b" + i + "\\b" for i in n_grams]), t))
    else:
        matches = list(re.finditer("|".join(n_grams), t))
    match_spans = [i.span() for i in matches if i.span()[1] - i.span()[0] > len_filter]
    match_text_tokenized = [i.group() for i in matches if i.span()[1] - i.span()[0] > len_filter]

    # now we remove any of the results if the entire matched ngram is just a stopword
    if remove_stopwords:
        match_spans = [span for i, span in enumerate(match_spans) if match_text_tokenized[i] not in STOPWORDS]
        match_text_tokenized = [text for text in match_text_tokenized if text not in STOPWORDS]

    return match_spans, match_text_tokenized


def make_lightgbm_features_with_query_context(query_context, ror_entry, lm):
    def lm_score(s):
        return lm.score(s, eos=False, bos=False)

    log_prob_nonsense = lm_score("qwertyuiop")

    q = query_context.get("q", "")
    if len(q) == 0:
        return [np.nan] * len(FEATURE_NAMES)

    q_split_set = query_context["q_split_set"]
    q_set_len = query_context["q_set_len"]
    regex = query_context["regex"]

    matched_across_fields = []
    feats = [np.round(np.log1p(ror_entry["works_count"]))]  # first feature
    names_and_acronyms_matches = set()
    for field in ["names", "acronyms", "city", "state", "country"]:
        match_spans = []
        match_text = []

        text_in_query = []
        for text in ror_entry[field]:
            # forward search (precompiled query pattern)
            ms, mt = find_query_ngrams_in_text_precompiled(text, regex)
            match_spans.extend(ms)
            match_text.extend(mt)
            matched_across_fields.extend(mt)
            text_in_query.append(int(text in q))

            # reverse search (text against query)
            ms, mt = find_query_ngrams_in_text(text, q)
            match_spans.extend(ms)
            match_text.extend(mt)
            matched_across_fields.extend(mt)

        match_spans_set = []
        match_text_set = []
        for t, s in sorted(zip(match_text, match_spans), key=lambda s: len(s[0]))[::-1]:
            if t not in match_text_set and ~np.any([t in i for i in match_text_set]):
                match_spans_set.append(s)
                match_text_set.append(t)

        matched_text_unigrams = set()
        for i in match_text_set:
            i_split = i.split()
            matched_text_unigrams.update(i_split)
            if field in {"names", "acronyms"}:
                names_and_acronyms_matches.update(i_split)

        if len(match_text_set) > 0:
            lm_probs = [lm_score(match) for match in match_text_set]
            match_word_lens = [len(i.split()) for i in match_text_set]

            matched_text_unigrams -= STOPWORDS
            matched_text_len = len(" ".join(matched_text_unigrams))

            feats.extend(
                [
                    matched_text_len / q_set_len,
                    np.nanmean(lm_probs),
                    np.nansum(np.array(lm_probs) * np.array(match_word_lens)),
                    np.any(text_in_query),
                ]
            )
        else:
            feats.extend([0, 0, 0, 0])

    if len(q_split_set) > 0:
        matched_split_set = set()
        for i in matched_across_fields:
            matched_split_set.update(i.split())
        matched_split_set -= STOPWORDS
        matched_len = len(" ".join(matched_split_set))
        feats.append(matched_len / q_set_len)
        unmatched = q_split_set - matched_split_set
        log_probs_unmatched_unquoted = [lm_score(i) for i in unmatched]
        feats.append(np.nansum([i for i in log_probs_unmatched_unquoted if i > log_prob_nonsense]))
        feats.extend([np.nan] * 3)
    else:
        feats.extend([np.nan] * 5)

    return feats


def make_feature_names_and_constraints():
    # we get these counts from openalex and unclear how good they are
    # it works better to just take a ballpark value here
    # instead of the exact value.
    feats = ["works_count"]

    # for lightgbm, 1 means positively monotonic, -1 means negatively monotonic and 0 means non-constraint
    constraints = ["1"]

    # features for each field in ROR
    for field in ["names", "acronyms", "city", "state", "country"]:

        feats.extend(
            [
                f"{field}_frac_of_query_matched_in_text",  # total fraction of the query that was matched in text
                f"{field}_mean_of_log_probs",  # statistics of the log-probs
                f"{field}_sum_of_log_probs*match_lens",
                f"{field}_any_text_matched_in_query",
            ]
        )

        constraints.extend(
            [
                "1",
                "-1",
                "-1",
                "1",
            ]
        )

    feats.extend(
        [
            "fraction_of_query_matched_across_all_fields",
            "sum_log_prob_of_unmatched_unigrams",
            "stage_1_score",
            "stage_1_score_is_-0.15",
            "stage_1_score_is_-0.10",
        ]
    )

    constraints.extend(["1", "1", "0", "0", "0"])

    return np.array(feats), ",".join(constraints)


FEATURE_NAMES, FEATURE_CONSTRAINTS = make_feature_names_and_constraints()


def make_lightgbm_features(query, ror_entry, lm, max_q_len=256):
    def lm_score(s):
        return lm.score(s, eos=False, bos=False)

    # later we will filter some features based on nonsensical unigrams in the query
    # this is the log probability lower-bound for sensible unigrams
    log_prob_nonsense = lm_score("qwertyuiop")

    # fix and lowercase the text
    q = str(query)[:max_q_len]

    # if there's no query left at this point, we return NaNs
    # which the model natively supports
    if len(q) == 0:
        return [np.nan] * len(FEATURE_NAMES)

    q_split_set = set(q.split()) - STOPWORDS
    q_set_len = len(" ".join(q_split_set))
    q_set_len = np.maximum(q_set_len, 1)  # to avoid division by zero later

    matched_across_fields = []
    feats = [np.round(np.log1p(ror_entry["works_count"]))]
    names_and_acronyms_matches = set()
    for field in ["names", "acronyms", "city", "state", "country"]:

        match_spans = []
        match_text = []

        text_in_query = []
        for text in ror_entry[field]:
            # forward search
            ms, mt = find_query_ngrams_in_text(q, text)
            match_spans.extend(ms)
            match_text.extend(mt)
            matched_across_fields.extend(mt)
            text_in_query.append(int(text in q))

            # reverse search
            ms, mt = find_query_ngrams_in_text(text, q)
            match_spans.extend(ms)
            match_text.extend(mt)
            matched_across_fields.extend(mt)

        # take the set of the results while excluding sub-ngrams if longer ngrams are found
        # e.g. if we already matched 'sentiment analysis', then 'sentiment' is excluded
        match_spans_set = []
        match_text_set = []
        for t, s in sorted(zip(match_text, match_spans), key=lambda s: len(s[0]))[::-1]:
            if t not in match_text_set and ~np.any([t in i for i in match_text_set]):
                match_spans_set.append(s)
                match_text_set.append(t)

        # match_text_set but unigrams
        matched_text_unigrams = set()
        for i in match_text_set:
            i_split = i.split()
            matched_text_unigrams.update(i_split)
            if field in {"names", "acronyms"}:
                names_and_acronyms_matches.update(i_split)

        if len(match_text_set) > 0:  # if any matches
            # log probabilities of the scores
            lm_probs = [lm_score(match) for match in match_text_set]

            match_word_lens = [len(i.split()) for i in match_text_set]

            # remove stopwords from unigrams
            matched_text_unigrams -= STOPWORDS
            matched_text_len = len(" ".join(matched_text_unigrams))

            feats.extend(
                [
                    # total fraction of the query that was matched in text
                    matched_text_len / q_set_len,
                    np.nanmean(lm_probs),  # average log-prob of the matches
                    # sum of log-prob of matches times word-lengths
                    np.nansum(np.array(lm_probs) * np.array(match_word_lens)),
                    np.any(text_in_query),
                ]
            )
        else:
            # if we have no matches, then the features are deterministically 0
            feats.extend([0, 0, 0, 0])

    # special features for how much of the query was matched/unmatched across all fields
    if len(q_split_set) > 0:
        matched_split_set = set()
        for i in matched_across_fields:
            matched_split_set.update(i.split())
        # making sure stopwords aren't an issue
        matched_split_set -= STOPWORDS
        # fraction of the query matched
        matched_len = len(" ".join(matched_split_set))
        feats.append(matched_len / q_set_len)
        # the log-prob of the unmatched quotes
        unmatched = q_split_set - matched_split_set
        log_probs_unmatched_unquoted = [lm_score(i) for i in unmatched]
        feats.append(np.nansum([i for i in log_probs_unmatched_unquoted if i > log_prob_nonsense]))
        feats.extend([np.nan] * 3)  # this will later be populated by the first stage ranker score
    else:
        feats.extend([np.nan] * 5)

    return feats
