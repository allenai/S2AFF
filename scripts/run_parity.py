"""
Parity harness for S2AFF stage comparisons.

Runs old vs new variants on identical inputs and checks equality with
floating-point tolerances where needed.
"""

import argparse
import sys

import pandas as pd
import numpy as np

from s2aff import S2AFF
from s2aff.consts import PATHS
from s2aff.model import NERPredictor, PairwiseRORLightGBMReranker, parse_ner_prediction
from s2aff.ror import RORIndex


EDGE_CASES = [
    "AI",
    "Dept. of Physics, MIT",
    "Department of Horse Racing, University of Washington, Seattle, WA 98115 USA",
    "1234",
    "A",
    "University of California, Berkeley, CA, USA",
]


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def _normalize_pipeline_arg(value: str) -> str:
    normalized = str(value).strip().lower()
    if normalized in {"python", "py", "legacy"}:
        return "python"
    if normalized in {"rust", "rs", "fast"}:
        return "rust"
    return normalized


def load_texts(limit=None, include_edge_cases=True):
    df = pd.read_csv(PATHS["gold_affiliation_annotations"])
    texts = df["original_affiliation"].astype(str).tolist()
    if include_edge_cases:
        texts = texts + EDGE_CASES
    if limit is not None:
        texts = texts[:limit]
    return texts


def _is_candidate_score_pair(item):
    return (
        isinstance(item, (tuple, list))
        and len(item) == 2
        and isinstance(item[0], (list, tuple))
        and isinstance(item[1], (list, tuple))
    )


def _compare_candidate_scores(left_item, right_item, rtol=1e-6, atol=1e-9):
    left_candidates, left_scores = left_item
    right_candidates, right_scores = right_item

    if list(left_candidates) != list(right_candidates):
        return False, "candidate order mismatch"

    if len(left_scores) != len(right_scores):
        return False, "score length mismatch"

    left_scores_arr = np.array(left_scores, dtype=float)
    right_scores_arr = np.array(right_scores, dtype=float)
    if not np.allclose(left_scores_arr, right_scores_arr, rtol=rtol, atol=atol, equal_nan=True):
        diffs = np.abs(left_scores_arr - right_scores_arr)
        max_idx = int(np.argmax(diffs))
        return (
            False,
            f"score mismatch at index {max_idx} (left={left_scores_arr[max_idx]}, right={right_scores_arr[max_idx]})",
        )
    return True, ""


def _compare_score_list(left_scores, right_scores, rtol=1e-6, atol=1e-9):
    if len(left_scores) != len(right_scores):
        return False, "score length mismatch"
    left_scores_arr = np.array(left_scores, dtype=float)
    right_scores_arr = np.array(right_scores, dtype=float)
    if not np.allclose(left_scores_arr, right_scores_arr, rtol=rtol, atol=atol, equal_nan=True):
        diffs = np.abs(left_scores_arr - right_scores_arr)
        max_idx = int(np.argmax(diffs))
        return (
            False,
            f"score mismatch at index {max_idx} (left={left_scores_arr[max_idx]}, right={right_scores_arr[max_idx]})",
        )
    return True, ""


def compare_stage_outputs(label, left, right, texts, max_mismatches=5, score_rtol=1e-6, score_atol=1e-9):
    if len(left) != len(right):
        print(f"{label}: length mismatch {len(left)} vs {len(right)}")
        return False

    mismatches = []
    for i, (l, r) in enumerate(zip(left, right)):
        if _is_candidate_score_pair(l) and _is_candidate_score_pair(r):
            ok, reason = _compare_candidate_scores(l, r, rtol=score_rtol, atol=score_atol)
            if ok:
                continue
            mismatches.append(i)
            if len(mismatches) <= max_mismatches:
                print(f"{label} mismatch at index {i}")
                print("text:", texts[i])
                print("reason:", reason)
                print("left candidates (head):", list(l[0])[:5])
                print("right candidates (head):", list(r[0])[:5])
                print("---")
            continue
        if l != r:
            mismatches.append(i)
            if len(mismatches) <= max_mismatches:
                print(f"{label} mismatch at index {i}")
                print("text:", texts[i])
                print("left:", l)
                print("right:", r)
                print("---")

    if mismatches:
        print(f"{label}: {len(mismatches)} mismatches")
        return False

    print(f"{label}: parity OK ({len(left)} items)")
    return True


def compare_end2end_outputs(label, left, right, texts, max_mismatches=5, score_rtol=1e-6, score_atol=1e-9):
    if len(left) != len(right):
        print(f"{label}: length mismatch {len(left)} vs {len(right)}")
        return False

    mismatches = []
    for i, (l, r) in enumerate(zip(left, right)):
        if not isinstance(l, dict) or not isinstance(r, dict):
            if l != r:
                mismatches.append(i)
                if len(mismatches) <= max_mismatches:
                    print(f"{label} mismatch at index {i}")
                    print("text:", texts[i])
                    print("left:", l)
                    print("right:", r)
                    print("---")
            continue

        if l.keys() != r.keys():
            mismatches.append(i)
            if len(mismatches) <= max_mismatches:
                print(f"{label} mismatch at index {i}")
                print("text:", texts[i])
                print("left keys:", sorted(l.keys()))
                print("right keys:", sorted(r.keys()))
                print("---")
            continue

        for key in l.keys():
            lv = l[key]
            rv = r[key]
            if key in {"stage1_scores", "stage2_scores"}:
                ok, reason = _compare_score_list(lv, rv, rtol=score_rtol, atol=score_atol)
                if ok:
                    continue
                mismatches.append(i)
                if len(mismatches) <= max_mismatches:
                    print(f"{label} mismatch at index {i}")
                    print("text:", texts[i])
                    print("field:", key)
                    print("reason:", reason)
                    print("---")
                break
            elif key in {"stage1_candidates", "stage2_candidates"}:
                if list(lv) != list(rv):
                    mismatches.append(i)
                    if len(mismatches) <= max_mismatches:
                        print(f"{label} mismatch at index {i}")
                        print("text:", texts[i])
                        print("field:", key)
                        print("left:", lv)
                        print("right:", rv)
                        print("---")
                    break
            else:
                if lv != rv:
                    mismatches.append(i)
                    if len(mismatches) <= max_mismatches:
                        print(f"{label} mismatch at index {i}")
                        print("text:", texts[i])
                        print("field:", key)
                        print("left:", lv)
                        print("right:", rv)
                        print("---")
                    break

    if mismatches:
        print(f"{label}: {len(mismatches)} mismatches")
        return False

    print(f"{label}: parity OK ({len(left)} items)")
    return True


def stage1_results(texts, ner_predictions, ror_index, stage1_pipeline):
    outputs = []
    for raw_affiliation, ner_prediction in zip(texts, ner_predictions):
        main, child, address, early_candidates = parse_ner_prediction(ner_prediction, ror_index)
        if stage1_pipeline == "rust" and hasattr(ror_index, "get_candidates_from_main_affiliation_rust"):
            candidates, scores = ror_index.get_candidates_from_main_affiliation_rust(
                main, address, early_candidates
            )
        else:
            if hasattr(ror_index, "get_candidates_from_main_affiliation_python"):
                candidates, scores = ror_index.get_candidates_from_main_affiliation_python(
                    main, address, early_candidates
                )
            else:
                candidates, scores = ror_index.get_candidates_from_main_affiliation(
                    main, address, early_candidates
                )
        outputs.append((list(candidates), [float(s) for s in scores]))
    return outputs


def stage2_results(texts, stage1_outputs, pairwise_model, top_k_first_stage, stage2_pipeline):
    outputs = []
    for raw_affiliation, (candidates, scores) in zip(texts, stage1_outputs):
        if len(candidates) > 1:
            candidates_in = candidates[:top_k_first_stage]
            scores_in = scores[:top_k_first_stage]
            if stage2_pipeline == "rust":
                if hasattr(pairwise_model, "predict_with_rust_backend"):
                    reranked_candidates, reranked_scores = pairwise_model.predict_with_rust_backend(
                        raw_affiliation, candidates_in, scores_in
                    )
                else:
                    reranked_candidates, reranked_scores = pairwise_model.predict(
                        raw_affiliation, candidates_in, scores_in
                    )
            else:
                if hasattr(pairwise_model, "predict_python"):
                    reranked_candidates, reranked_scores = pairwise_model.predict_python(
                        raw_affiliation, candidates_in, scores_in
                    )
                else:
                    reranked_candidates, reranked_scores = pairwise_model.predict(
                        raw_affiliation, candidates_in, scores_in
                    )
            outputs.append((list(reranked_candidates), [float(s) for s in reranked_scores]))
        else:
            outputs.append((list(candidates), [float(s) for s in scores]))
    return outputs


def end2end_outputs(
    texts, ner_predictor, ror_index, pairwise_model, stage1_pipeline, stage2_pipeline
):
    model = S2AFF(
        ner_predictor,
        ror_index,
        pairwise_model,
        stage1_pipeline=stage1_pipeline,
        stage2_pipeline=stage2_pipeline,
    )
    outputs = model.predict(texts)
    normalized = []
    for output in outputs:
        normalized_output = dict(output)
        for key in ("stage1_scores", "stage2_scores"):
            if key in normalized_output:
                normalized_output[key] = [float(s) for s in normalized_output[key]]
        normalized.append(normalized_output)
    return normalized


def main():
    parser = argparse.ArgumentParser(description="Run parity checks between S2AFF variants.")
    parser.add_argument(
        "--mode",
        choices=["stage1", "stage2", "end2end", "all"],
        default="all",
        help="Which parity checks to run.",
    )
    parser.add_argument(
        "--left-stage1",
        default="python",
        help="Stage1 pipeline for left side comparisons (python/rust).",
    )
    parser.add_argument(
        "--right-stage1",
        default="rust",
        help="Stage1 pipeline for right side comparisons (python/rust).",
    )
    parser.add_argument(
        "--left-stage2",
        default="python",
        help="Stage2 pipeline for left side comparisons (python/rust).",
    )
    parser.add_argument(
        "--right-stage2",
        default="rust",
        help="Stage2 pipeline for right side comparisons (python/rust).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of affiliations to test.",
    )
    parser.add_argument(
        "--no-edge-cases",
        action="store_true",
        help="Skip additional edge-case inputs.",
    )
    parser.add_argument(
        "--use-cuda",
        action="store_true",
        help="Use CUDA for NER predictor.",
    )
    parser.add_argument(
        "--top-k-first-stage",
        type=int,
        default=100,
        help="Top-k candidates to rerank during stage2 parity.",
    )
    args = parser.parse_args()
    left_stage1_pipeline = _normalize_pipeline_arg(args.left_stage1)
    right_stage1_pipeline = _normalize_pipeline_arg(args.right_stage1)
    left_stage2_pipeline = _normalize_pipeline_arg(args.left_stage2)
    right_stage2_pipeline = _normalize_pipeline_arg(args.right_stage2)

    texts = load_texts(limit=args.limit, include_edge_cases=not args.no_edge_cases)

    ner_predictor = NERPredictor(use_cuda=args.use_cuda)
    ror_index = RORIndex()
    pairwise_model = PairwiseRORLightGBMReranker(ror_index)

    ner_predictions = ner_predictor.predict(texts)

    ok = True
    if args.mode in {"stage1", "all"}:
        left = stage1_results(texts, ner_predictions, ror_index, left_stage1_pipeline)
        right = stage1_results(texts, ner_predictions, ror_index, right_stage1_pipeline)
        ok = compare_stage_outputs("stage1", left, right, texts) and ok

    if args.mode in {"stage2", "all"}:
        stage1_left = stage1_results(texts, ner_predictions, ror_index, left_stage1_pipeline)
        left = stage2_results(
            texts,
            stage1_left,
            pairwise_model,
            args.top_k_first_stage,
            left_stage2_pipeline,
        )
        right = stage2_results(
            texts,
            stage1_left,
            pairwise_model,
            args.top_k_first_stage,
            right_stage2_pipeline,
        )
        ok = compare_stage_outputs("stage2", left, right, texts) and ok

    if args.mode in {"end2end", "all"}:
        left = end2end_outputs(
            texts,
            ner_predictor,
            ror_index,
            pairwise_model,
            left_stage1_pipeline,
            left_stage2_pipeline,
        )
        right = end2end_outputs(
            texts,
            ner_predictor,
            ror_index,
            pairwise_model,
            right_stage1_pipeline,
            right_stage2_pipeline,
        )
        ok = compare_end2end_outputs(
            "end2end",
            left,
            right,
            texts,
            score_rtol=1e-6,
            score_atol=1e-9,
        ) and ok

    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
