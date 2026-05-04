"""GPTCache QQP Benchmark

Evaluates cache accuracy and performance using the Quora Question Pairs dataset.
Runs two configurations for comparison:
  1. Baseline: ONNX (768d) + Flat FAISS index
  2. Optimized: MRL (256d) + HNSW+SQ8 FAISS index

Metrics: True Positive rate (should hit), False Positive rate (should NOT hit),
         latency percentiles, throughput, storage size.

Usage:
  python benchmark_qqp.py --mode baseline
  python benchmark_qqp.py --mode optimized
"""

import argparse
import os
import shutil
import time

import numpy as np
import psutil
from datasets import load_dataset

from gptcache import cache, Config
from gptcache.manager import get_data_manager, CacheBase, VectorBase
from gptcache.similarity_evaluation.distance import SearchDistanceEvaluation
from gptcache.embedding.sbert_mrl import SBERTMRL

# --- Configuration ---
NUM_INGEST = 10000       # Number of duplicate pairs to build the database from
NUM_TP_TEST = 2000       # True-duplicate queries (goal: high hit rate)
NUM_FP_TEST = 2000       # Non-duplicate queries (goal: low hit rate)
SIMILARITY_THRESH = 0.90


def create_encoder(mode):
    """Create the embedding encoder for the given mode."""
    if mode == "baseline":
        # Baseline uses the full 768 dimensions
        return SBERTMRL(target_dim=768)
    else:
        # Optimized uses MRL 256 dimensions
        return SBERTMRL(target_dim=256)


def setup_cache(mode, work_dir):
    """Initialize GPTCache with the appropriate configuration.

    Returns (encoder, data_manager, faiss_path, sqlite_path).
    """
    os.makedirs(work_dir, exist_ok=True)
    encoder = create_encoder(mode)
    dim = encoder.dimension

    sqlite_path = os.path.join(work_dir, "sqlite.db")
    faiss_path = os.path.join(work_dir, "faiss.index")

    cache_base = CacheBase("sqlite", sql_url=f"sqlite:///{sqlite_path}")

    if mode == "baseline":
        vector_base = VectorBase("faiss", dimension=dim, index_path=faiss_path)
        config_label = f"Flat index, {dim}d float32"
    else:
        vector_base = VectorBase(
            "faiss", dimension=dim, index_path=faiss_path, index_type="hnsw_sq8"
        )
        config_label = f"HNSW+SQ8 index, {dim}d uint8"

    data_manager = get_data_manager(cache_base, vector_base, max_size=200000)

    cache.init(
        embedding_func=encoder.to_embeddings,
        data_manager=data_manager,
        similarity_evaluation=SearchDistanceEvaluation(),
        config=Config(similarity_threshold=SIMILARITY_THRESH),
    )

    print(f"  Encoder   : {encoder.__class__.__name__} (dim={dim})")
    print(f"  Index     : {config_label}")
    print(f"  Threshold : {SIMILARITY_THRESH}")

    return encoder, data_manager, faiss_path, sqlite_path


def query_cache_direct(query_text, encoder, data_manager):
    """Perform a single cache lookup using the internal pipeline.

    GPTCache has no public `cache.get()` API — queries normally go through
    the OpenAI adapter. For benchmarking we call the pipeline directly:
      embed → search → evaluate threshold

    Returns (is_hit: bool, latency_seconds: float).
    """
    start = time.time()

    # 1. Embed the query
    embedding = encoder.to_embeddings(query_text)

    # 2. Search (data_manager.search normalizes the vector internally)
    search_results = data_manager.search(embedding)

    is_hit = False
    if search_results:
        distance, cache_id = search_results[0]  # best match

        # 3. Evaluate: SearchDistanceEvaluation computes score = max_distance - L2_distance
        evaluator = cache.similarity_evaluation
        score = evaluator.evaluation({}, {"search_result": (distance, cache_id)})
        min_r, max_r = evaluator.range()
        rank_threshold = (max_r - min_r) * SIMILARITY_THRESH

        if score >= rank_threshold:
            is_hit = True

    latency = time.time() - start
    return is_hit, latency


def run_test(test_name, queries, encoder, data_manager):
    """Run a set of queries sequentially and collect metrics."""
    print(f"\n--- {test_name} ({len(queries)} queries) ---")

    hits = 0
    latencies = []
    process = psutil.Process(os.getpid())
    peak_ram = 0

    for i, q in enumerate(queries):
        is_hit, latency = query_cache_direct(q, encoder, data_manager)
        if is_hit:
            hits += 1
        latencies.append(latency)
        ram = process.memory_info().rss / (1024 * 1024)
        peak_ram = max(peak_ram, ram)

        if (i + 1) % 500 == 0:
            print(f"  Progress: {i+1}/{len(queries)} "
                  f"(hits so far: {hits}, avg latency: {np.mean(latencies)*1000:.1f}ms)")

    latencies_ms = np.array(latencies) * 1000
    total_time = sum(latencies)
    n = len(queries)

    print(f"\n  Results for: {test_name}")
    print(f"  Total Time  : {total_time:.2f}s ({n / total_time:.1f} QPS)")
    print(f"  Cache Hits  : {hits}/{n} ({hits/n*100:.2f}%)")
    print(f"  Cache Misses: {n-hits}/{n} ({(n-hits)/n*100:.2f}%)")
    print(f"  Avg Latency : {np.mean(latencies_ms):.2f} ms")
    print(f"  P50 Latency : {np.percentile(latencies_ms, 50):.2f} ms")
    print(f"  P90 Latency : {np.percentile(latencies_ms, 90):.2f} ms")
    print(f"  P99 Latency : {np.percentile(latencies_ms, 99):.2f} ms")
    print(f"  Peak RAM    : {peak_ram:.1f} MB")

    return {"hits": hits, "total": n, "hit_rate": hits/n,
            "avg_latency_ms": np.mean(latencies_ms),
            "p99_latency_ms": np.percentile(latencies_ms, 99),
            "peak_ram_mb": peak_ram}


def run(mode):
    # 1. Load Dataset
    print("=" * 60)
    print(f"GPTCache QQP Benchmark — Mode: {mode.upper()}")
    print("=" * 60)

    print("\nLoading Quora Question Pairs dataset...")
    dataset = load_dataset("glue", "qqp", split="train")

    # Split into duplicates and non-duplicates
    duplicates = dataset.filter(lambda x: x["label"] == 1)
    non_duplicates = dataset.filter(lambda x: x["label"] == 0)
    print(f"  Total pairs: {len(dataset)}")
    print(f"  Duplicates: {len(duplicates)}, Non-duplicates: {len(non_duplicates)}")

    # 2. Setup cache
    work_dir = f"bench_{mode}"
    # Clean previous run
    if os.path.isdir(work_dir):
        shutil.rmtree(work_dir)

    print(f"\nInitializing GPTCache ({mode})...")
    encoder, data_manager, faiss_path, sqlite_path = setup_cache(mode, work_dir)

    # 3. Prepare data
    dup_pairs = list(duplicates.select(range(NUM_INGEST)))
    db_questions = [pair["question1"] for pair in dup_pairs]

    print(f"\nIngesting {len(db_questions)} questions...")
    start_insert = time.time()
    dummy_answers = [f"Answer_{i}" for i in range(len(db_questions))]
    cache.import_data(questions=db_questions, answers=dummy_answers)
    insert_time = time.time() - start_insert
    print(f"Ingestion complete in {insert_time:.2f}s "
          f"({len(db_questions)/insert_time:.0f} vectors/sec)")

    # TP queries: question2 from the same duplicate pairs we ingested
    tp_queries = [pair["question2"] for pair in dup_pairs[:NUM_TP_TEST]]

    # FP queries: question2 from non-duplicate pairs
    fp_pairs = list(non_duplicates.select(range(NUM_FP_TEST)))
    fp_queries = [pair["question2"] for pair in fp_pairs]

    # 4. Run tests
    tp_results = run_test("True Positive (should HIT)", tp_queries, encoder, data_manager)
    fp_results = run_test("False Positive (should MISS)", fp_queries, encoder, data_manager)

    # 5. Storage telemetry
    data_manager.close()
    print("\n" + "=" * 60)
    print(f"FINAL SUMMARY — {mode.upper()}")
    print("=" * 60)
    print(f"  TP Hit Rate      : {tp_results['hit_rate']*100:.2f}%  (goal: high)")
    print(f"  FP Hit Rate      : {fp_results['hit_rate']*100:.2f}%  (goal: low)")
    print(f"  Avg Latency (TP) : {tp_results['avg_latency_ms']:.2f} ms")
    print(f"  P99 Latency (TP) : {tp_results['p99_latency_ms']:.2f} ms")

    for filepath in [faiss_path, sqlite_path]:
        if os.path.isfile(filepath):
            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            print(f"  {os.path.basename(filepath):15s}: {size_mb:.2f} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPTCache QQP Benchmark")
    parser.add_argument(
        "--mode",
        choices=["baseline", "optimized"],
        required=True,
        help="baseline = ONNX+Flat (768d), optimized = MRL+HNSW+SQ8 (256d)",
    )
    args = parser.parse_args()
    run(args.mode)