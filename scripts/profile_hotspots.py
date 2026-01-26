"""
Profile S2AFF hotspots (stage1/stage2) with cProfile.
Focuses on Python-level bottlenecks while preserving exact behavior.
"""
import argparse
import time
import pstats
import cProfile
from typing import List, Tuple

from s2aff.consts import PATHS
from s2aff.data import load_gold_affiliation_annotations
from s2aff.model import NERPredictor, PairwiseRORLightGBMReranker, parse_ner_prediction
from s2aff.ror import RORIndex


def _normalize_pipeline_variant(value: str, stage: int) -> str:
    normalized = str(value).strip().lower()
    if normalized in {"python", "py", "legacy"}:
        return "v1"
    if normalized in {"rust", "rs", "fast"}:
        return "v7" if stage == 1 else "v3"
    return normalized


def _profile_block(fn, *args, **kwargs):
    profiler = cProfile.Profile()
    profiler.enable()
    result = fn(*args, **kwargs)
    profiler.disable()
    return result, profiler


def _print_top_stats(profiler, title, limit=40, filename_filters=None, sort_key="cumulative"):
    print(f"\n=== {title} (top {limit}, sort={sort_key}) ===")
    stats = pstats.Stats(profiler)
    if filename_filters:
        entries = []
        for (filename, line, func), (cc, nc, tt, ct, callers) in stats.stats.items():
            if not any(f in filename for f in filename_filters):
                continue
            entries.append((ct, tt, nc, cc, filename, line, func))
        entries.sort(reverse=True, key=lambda x: x[0])
        print(f"{'cum':>9} {'self':>9} {'calls':>9} {'ccalls':>9}  location")
        for ct, tt, nc, cc, filename, line, func in entries[:limit]:
            print(f"{ct:9.3f} {tt:9.3f} {nc:9d} {cc:9d}  {filename}:{line}({func})")
        return
    stats.sort_stats(sort_key).print_stats(limit)


def _stage1_variants(ror_index, variant, main, address, early_candidates):
    if variant == "v7" and hasattr(ror_index, "get_candidates_from_main_affiliation_v7"):
        return ror_index.get_candidates_from_main_affiliation_v7(main, address, early_candidates)
    if variant == "v1" and hasattr(ror_index, "get_candidates_from_main_affiliation_v1"):
        return ror_index.get_candidates_from_main_affiliation_v1(main, address, early_candidates)
    return ror_index.get_candidates_from_main_affiliation(main, address, early_candidates)


def run_stage1(parsed, ror_index, stage1_variant):
    candidates_and_scores = []
    if stage1_variant == "v7" and hasattr(ror_index, "get_candidates_from_main_affiliation_v7_batch"):
        mains = [main for main, _, _, _ in parsed]
        addresses = [address for _, _, address, _ in parsed]
        early_candidates_list = [early_candidates for _, _, _, early_candidates in parsed]
        candidates_list, scores_list = ror_index.get_candidates_from_main_affiliation_v7_batch(
            mains, addresses, early_candidates_list
        )
        candidates_and_scores = list(zip(candidates_list, scores_list))
        return candidates_and_scores
    for main, child, address, early_candidates in parsed:
        candidates, scores = _stage1_variants(ror_index, stage1_variant, main, address, early_candidates)
        candidates_and_scores.append((candidates, scores))
    return candidates_and_scores


def run_stage2(raw_affiliations, candidates_and_scores, pairwise_model, stage2_variant, top_k_first_stage):
    if stage2_variant == "v3" and hasattr(pairwise_model, "batch_predict_v3"):
        batch_candidates = [c[:top_k_first_stage] for c, _ in candidates_and_scores]
        batch_scores = [s[:top_k_first_stage] for _, s in candidates_and_scores]
        return pairwise_model.batch_predict_v3(raw_affiliations, batch_candidates, batch_scores)

    reranked_candidates_list = []
    reranked_scores_list = []
    for raw_affiliation, (candidates, scores) in zip(raw_affiliations, candidates_and_scores):
        candidates_in = candidates[:top_k_first_stage]
        scores_in = scores[:top_k_first_stage]
        if stage2_variant == "v3" and hasattr(pairwise_model, "predict_v3"):
            reranked_candidates, reranked_scores = pairwise_model.predict_v3(
                raw_affiliation, candidates_in, scores_in
            )
        else:
            reranked_candidates, reranked_scores = pairwise_model.predict(
                raw_affiliation, candidates_in, scores_in
            )
        reranked_candidates_list.append(reranked_candidates)
        reranked_scores_list.append(reranked_scores)
    return reranked_candidates_list, reranked_scores_list


def main():
    parser = argparse.ArgumentParser(description="Profile S2AFF hotspots for stage1/stage2.")
    parser.add_argument("--limit", type=int, default=20, help="Number of affiliations to profile.")
    parser.add_argument("--use-cuda", action="store_true", help="Use CUDA for NER predictor.")
    parser.add_argument("--stage1-variant", default="rust", help="Stage1 pipeline (python/rust).")
    parser.add_argument("--stage2-variant", default="rust", help="Stage2 pipeline (python/rust).")
    parser.add_argument("--top-k-first-stage", type=int, default=100, help="Top-k to rerank for stage2.")
    parser.add_argument(
        "--profile",
        choices=["stage1", "stage2", "all"],
        default="all",
        help="Which sections to profile.",
    )
    parser.add_argument("--top", type=int, default=40, help="Top functions to report.")
    parser.add_argument("--sort", default="cumulative", help="Sort key for stats (cumulative, time).")
    args = parser.parse_args()

    df = load_gold_affiliation_annotations()
    texts = df.original_affiliation.values[: args.limit].tolist()

    print(f"Loaded {len(texts)} affiliations for profiling.")
    ner_predictor = NERPredictor(use_cuda=args.use_cuda)
    ror_index = RORIndex()
    pairwise_model = PairwiseRORLightGBMReranker(ror_index)

    t0 = time.perf_counter()
    ner_predictions = ner_predictor.predict(texts)
    t1 = time.perf_counter()
    print(f"NER time (not profiled): {t1 - t0:.3f}s")

    parsed = [parse_ner_prediction(pred, ror_index) for pred in ner_predictions]
    stage1_variant = _normalize_pipeline_variant(args.stage1_variant, stage=1)
    stage2_variant = _normalize_pipeline_variant(args.stage2_variant, stage=2)

    candidates_and_scores = None
    stage1_prof = None
    stage2_prof = None

    if args.profile in {"stage1", "all"}:
        t_start = time.perf_counter()
        (candidates_and_scores, stage1_prof) = _profile_block(
            run_stage1, parsed, ror_index, args.stage1_variant
        )
        t_end = time.perf_counter()
        print(f"Stage 1 total: {t_end - t_start:.3f}s")
    else:
        candidates_and_scores = run_stage1(parsed, ror_index, stage1_variant)

    if args.profile in {"stage2", "all"}:
        t_start = time.perf_counter()
        (_, stage2_prof) = _profile_block(
            run_stage2,
            texts,
            candidates_and_scores,
            pairwise_model,
            stage2_variant,
            args.top_k_first_stage,
        )
        t_end = time.perf_counter()
        print(f"Stage 2 total: {t_end - t_start:.3f}s")

    filename_filters = ["s2aff"]
    if stage1_prof is not None:
        _print_top_stats(stage1_prof, "Stage 1 Hotspots", args.top, filename_filters, args.sort)
    if stage2_prof is not None:
        _print_top_stats(stage2_prof, "Stage 2 Hotspots", args.top, filename_filters, args.sort)


if __name__ == "__main__":
    main()
