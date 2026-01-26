import json
import os
import sys
import time
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from s2aff.features import FEATURE_NAMES

from s2aff.consts import CACHE_ROOT
from s2aff.file_cache import cached_path

try:
    import s2aff_rust  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    s2aff_rust = None

RUST_INDEX_CACHE_VERSION = 8
FEATURE_LEN = len(FEATURE_NAMES)


def rust_enabled() -> bool:
    pipeline = os.getenv("S2AFF_PIPELINE")
    if pipeline is not None:
        pipeline = str(pipeline).strip().lower()
        if pipeline in {"python", "py", "legacy"}:
            return False
        if pipeline in {"rust", "rs", "fast"}:
            return True
    return True


def _rust_module_available() -> bool:
    return s2aff_rust is not None and hasattr(s2aff_rust, "RorIndex")


def rust_available() -> bool:
    return rust_enabled() and _rust_module_available()


def _rust_log_enabled() -> bool:
    flag = os.getenv("S2AFF_RUST_LOG", "")
    return flag.lower() in {"1", "true", "yes", "on"}


def _rust_log(message: str) -> None:
    if not _rust_log_enabled():
        return
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[s2aff_rust] {ts} {message}", file=sys.stderr, flush=True)


_RUST_BACKEND_CACHE: Dict[Tuple[Any, ...], "RustBackend"] = {}


class RustBackend:
    def __init__(self, ror_index):
        word_multiplier = ror_index.word_multiplier
        word_multiplier_is_log = isinstance(word_multiplier, str) and word_multiplier == "log"
        word_multiplier_value = 1.0 if word_multiplier_is_log else float(word_multiplier)
        cache_path = _rust_cache_path(ror_index)

        start = time.perf_counter()
        _rust_log(
            "RorIndex init start "
            f"(cache_path={cache_path}, use_prob_weights={bool(ror_index.use_prob_weights)}, "
            f"ns={sorted(list(ror_index.ns))})"
        )
        self._inner = s2aff_rust.RorIndex(
            ror_data_path=cached_path(ror_index.ror_data_path),
            ror_edits_path=cached_path(ror_index.ror_edits_path),
            country_info_path=cached_path(ror_index.country_info_path),
            works_counts_path=cached_path(ror_index.works_counts_path),
            use_prob_weights=bool(ror_index.use_prob_weights),
            word_multiplier=word_multiplier_value,
            word_multiplier_is_log=bool(word_multiplier_is_log),
            max_intersection_denominator=bool(ror_index.max_intersection_denominator),
            ns=sorted(list(ror_index.ns)),
            insert_early_candidates_ind=ror_index.insert_early_candidates_ind,
            reinsert_cutoff_frac=float(ror_index.reinsert_cutoff_frac),
            score_based_early_cutoff=float(ror_index.score_based_early_cutoff),
            cache_path=cache_path,
        )
        elapsed = time.perf_counter() - start
        _rust_log(f"RorIndex init done in {elapsed:.2f}s")

    def match_query_context_batch(self, queries: List[str], candidates_list: List[List[str]]):
        return self._inner.match_query_context_batch(queries, candidates_list)

    def get_candidates_from_main_affiliation_v7(
        self, main: Any, address: str = "", early_candidates: Optional[List[str]] = None
    ) -> Tuple[List[str], List[float]]:
        if early_candidates is None:
            early_candidates = []
        if isinstance(main, str):
            main = [main]
        address_arg = address if address else None
        return self._inner.get_candidates_v7(main, address_arg, early_candidates)

    def get_candidates_from_main_affiliation_v7_batch(
        self,
        mains: List[Any],
        addresses: Optional[List[Any]] = None,
        early_candidates_list: Optional[List[List[str]]] = None,
    ) -> Tuple[List[List[str]], List[List[float]]]:
        if addresses is None:
            addresses = [None] * len(mains)
        if early_candidates_list is None:
            early_candidates_list = [[] for _ in mains]
        if len(addresses) != len(mains):
            raise ValueError("addresses length mismatch")
        if len(early_candidates_list) != len(mains):
            raise ValueError("early_candidates_list length mismatch")

        normalized_mains: List[List[str]] = []
        normalized_addresses: List[Optional[str]] = []
        for main, address in zip(mains, addresses):
            if isinstance(main, str):
                normalized_mains.append([main])
            else:
                normalized_mains.append(list(main))
            addr_value = address
            if isinstance(addr_value, list):
                addr_value = " ".join(addr_value)
            if isinstance(addr_value, str) and addr_value:
                normalized_addresses.append(addr_value)
            else:
                normalized_addresses.append(None)

        normalized_early: List[List[str]] = []
        for early in early_candidates_list:
            normalized_early.append(list(early) if early else [])

        return self._inner.get_candidates_v7_batch(
            normalized_mains, normalized_addresses, normalized_early
        )

    def build_lightgbm_features_with_query_context_batch(
        self,
        lm,
        query_contexts: List[Dict[str, Any]],
        candidates_list: List[List[str]],
        get_ror_entry: Callable[[str], Dict[str, Any]],
    ) -> Tuple[np.ndarray, List[int]]:
        queries = [qc["q"] for qc in query_contexts]
        total_candidates = sum(len(cands) for cands in candidates_list)
        start = time.perf_counter()
        _rust_log(
            f"match_query_context_batch start (queries={len(queries)}, total_candidates={total_candidates})"
        )
        rust_matches = self.match_query_context_batch(queries, candidates_list)
        elapsed = time.perf_counter() - start
        _rust_log(f"match_query_context_batch done in {elapsed:.2f}s")

        def lm_score(s: str) -> float:
            return lm.score(s, eos=False, bos=False)

        log_prob_nonsense = lm_score("qwertyuiop")

        X_all: List[List[float]] = []
        split_indices: List[int] = [0]

        for qc, candidates, matches in zip(query_contexts, candidates_list, rust_matches):
            if not qc.get("q"):
                nan_row = [np.nan] * FEATURE_LEN
                for _ in candidates:
                    X_all.append(list(nan_row))
                split_indices.append(len(X_all))
                continue

            q_set_len = qc["q_set_len"]
            q_split_set = qc["q_split_set"]
            for candidate_id, match in zip(candidates, matches):
                ror_entry = get_ror_entry(candidate_id)
                feats: List[float] = [float(np.round(np.log1p(ror_entry["works_count"])))]

                for field_idx in range(5):
                    field_matches = match.field_matches[field_idx]
                    if field_matches:
                        lm_probs = [lm_score(m) for m in field_matches]
                        match_word_lens = [len(m.split()) for m in field_matches]
                        matched_text_len = match.field_matched_text_lens[field_idx]
                        feats.extend(
                            [
                                matched_text_len / q_set_len,
                                float(np.nanmean(lm_probs)) if lm_probs else 0.0,
                                float(np.nansum(np.array(lm_probs) * np.array(match_word_lens)))
                                if lm_probs
                                else 0.0,
                                bool(match.field_any_text_in_query[field_idx]),
                            ]
                        )
                    else:
                        feats.extend([0.0, 0.0, 0.0, 0.0])

                if q_split_set:
                    matched_split_set = set(match.matched_split_set)
                    matched_len = len(" ".join(matched_split_set))
                    feats.append(matched_len / q_set_len)
                    unmatched = q_split_set - matched_split_set
                    log_probs_unmatched = [lm_score(i) for i in unmatched]
                    feats.append(
                        float(
                            np.nansum([i for i in log_probs_unmatched if i > log_prob_nonsense])
                        )
                    )
                    feats.extend([np.nan] * 3)
                else:
                    feats.extend([np.nan] * 5)

                X_all.append(feats)
            split_indices.append(len(X_all))

        if len(X_all) == 0:
            return np.array([]), split_indices
        return np.array(X_all), split_indices


def get_rust_backend(ror_index) -> Optional[RustBackend]:
    if not rust_available():
        return None
    key = (
        ror_index.ror_data_path,
        ror_index.ror_edits_path,
        ror_index.country_info_path,
        getattr(ror_index, "works_counts_path", None),
        ror_index.use_prob_weights,
        ror_index.word_multiplier,
        ror_index.max_intersection_denominator,
        tuple(sorted(list(ror_index.ns))),
        ror_index.insert_early_candidates_ind,
        ror_index.reinsert_cutoff_frac,
        ror_index.score_based_early_cutoff,
    )
    cached = _RUST_BACKEND_CACHE.get(key)
    if cached is not None:
        return cached
    backend = RustBackend(ror_index)
    _RUST_BACKEND_CACHE[key] = backend
    return backend


def _rust_cache_path(ror_index) -> str:
    def _stat(path: str) -> Dict[str, Any]:
        try:
            st = os.stat(path)
            return {"size": st.st_size, "mtime": st.st_mtime}
        except FileNotFoundError:
            return {"size": None, "mtime": None}

    payload = {
        "rust_cache_version": RUST_INDEX_CACHE_VERSION,
        "ror_data_path": str(ror_index.ror_data_path),
        "ror_edits_path": str(ror_index.ror_edits_path),
        "country_info_path": str(ror_index.country_info_path),
        "works_counts_path": str(ror_index.works_counts_path),
        "use_prob_weights": bool(ror_index.use_prob_weights),
        "word_multiplier": ror_index.word_multiplier,
        "max_intersection_denominator": bool(ror_index.max_intersection_denominator),
        "ns": sorted(list(ror_index.ns)),
        "insert_early_candidates_ind": ror_index.insert_early_candidates_ind,
        "reinsert_cutoff_frac": float(ror_index.reinsert_cutoff_frac),
        "score_based_early_cutoff": float(ror_index.score_based_early_cutoff),
        "ror_stat": _stat(str(ror_index.ror_data_path)),
        "edits_stat": _stat(str(ror_index.ror_edits_path)),
        "country_stat": _stat(str(ror_index.country_info_path)),
        "works_stat": _stat(str(ror_index.works_counts_path)),
    }
    digest = sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    cache_dir = Path(CACHE_ROOT) / "indices" / "rust"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return str(cache_dir / f"ror_index_{digest}.bin")
