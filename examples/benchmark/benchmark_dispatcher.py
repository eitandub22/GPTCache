"""Concurrency benchmark for EmbeddingDispatcher (report contribution #3).

Measures embedding *throughput* under concurrent load: a single-process encoder
(the GPTCache default -- concurrent callers serialize on one model) versus an
EmbeddingDispatcher that fans calls across worker processes. This targets
throughput at concurrency, NOT single-call latency: a lone caller sees only IPC
overhead, so the dispatcher is expected to LOSE at low concurrency and WIN once
enough callers overlap. The output is that crossover.

Schema matches the reference logs in bench_embedding_dispatcher/results_run*.json.

Usage:
  # offline plumbing check -- real worker processes, fake (no-download) encoder
  python examples/benchmark/benchmark_dispatcher.py --self-check

  # paper run: real SBERT, UltraChat prompts, the recorded crossover
  python examples/benchmark/benchmark_dispatcher.py --dataset ultrachat \
      --n-prompts 200 --concurrency-levels 1,10,50,100 --out bench_embedding_dispatcher/results_run1.json
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List

import numpy as np

# ---------------------------------------------------------------------------
# Ensure repo root is importable when run directly
# ---------------------------------------------------------------------------
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from gptcache.embedding.dispatcher import EmbeddingDispatcher

try:
    import psutil
    _HAVE_PSUTIL = True
except ImportError:                       # ponytail: soft dep; rss_mb -> 0.0 without it
    _HAVE_PSUTIL = False


# ---------------------------------------------------------------------------
# Picklable, module-level factories. On Windows the pool uses "spawn", which
# pickles the factory to each worker -- so it must be a top-level callable, not
# a lambda or closure.
# ---------------------------------------------------------------------------
class _SBERTFactory:
    def __init__(self, model_name):
        self.model_name = model_name

    def __call__(self):
        from gptcache.embedding import SBERT
        return SBERT(self.model_name)


class _FakeEmbedding:
    """Zero-download encoder for --self-check: exercises the real process
    fan-out and the JSON schema without pulling a model."""

    def to_embeddings(self, data, **_):
        return np.zeros(8, dtype=np.float32)

    @property
    def dimension(self):
        return 8


class _FakeFactory:
    def __call__(self):
        return _FakeEmbedding()


# ---------------------------------------------------------------------------
# Workload
# ---------------------------------------------------------------------------
def load_prompts(dataset: str, n: int, seed: int) -> List[str]:
    if dataset == "synthetic":
        # Encode cost is what we measure, so exact repeats are irrelevant here;
        # distinct prompts keep every call doing real work.
        return [f"benchmark dispatcher prompt {i} about topic {i % 97} with filler"
                for i in range(n)]
    from benchmark_lmsys import load_lmsys, load_wildchat, load_ultrachat
    loader = {"lmsys": load_lmsys, "wildchat": load_wildchat}.get(dataset, load_ultrachat)
    entries = loader(n, seed)
    if not entries:
        raise SystemExit("No entries loaded -- check dataset access or --n-prompts.")
    return [e.prompt for e in entries]


def _rss_mb() -> float:
    """RSS of this process plus all worker children (MB). The dispatcher's cost
    is exactly the per-worker model copies, so children must be summed in."""
    if not _HAVE_PSUTIL:
        return 0.0
    proc = psutil.Process()
    total = proc.memory_info().rss
    for child in proc.children(recursive=True):
        try:
            total += child.memory_info().rss
        except psutil.Error:
            pass
    return total / (1024 * 1024)


def run_concurrent(encode_fn, prompts: List[str], concurrency: int) -> dict:
    """Push every prompt through encode_fn using `concurrency` client threads,
    timing each call. Wall-clock elapsed is the throughput number; per-call
    latencies give p50/p99."""
    latencies = [0.0] * len(prompts)

    def task(i):
        t0 = time.perf_counter()
        encode_fn(prompts[i])
        latencies[i] = (time.perf_counter() - t0) * 1000.0

    t_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        list(ex.map(task, range(len(prompts))))
    elapsed = time.perf_counter() - t_start

    lat = np.asarray(latencies)
    return {
        "elapsed_s": elapsed,
        "throughput_qps": (len(prompts) / elapsed) if elapsed > 0 else float("inf"),
        "p50_ms": float(np.percentile(lat, 50)),
        "p99_ms": float(np.percentile(lat, 99)),
    }


# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="EmbeddingDispatcher concurrency benchmark")
    p.add_argument("--dataset", default="ultrachat",
                   choices=["synthetic", "ultrachat", "lmsys", "wildchat"],
                   help="synthetic needs no HF access but still downloads the SBERT "
                        "model; use --self-check for a fully offline plumbing test")
    p.add_argument("--n-prompts", type=int, default=200)
    p.add_argument("--concurrency-levels", default="1,10,50,100",
                   help="Comma-separated in-flight-request counts to sweep")
    p.add_argument("--num-workers", type=int, default=None,
                   help="Dispatcher worker processes (default: auto = min(cores-1, 8))")
    p.add_argument("--embed-model", default="all-MiniLM-L6-v2")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workdir", default="bench_embedding_dispatcher")
    p.add_argument("--out", default=None)
    p.add_argument("--self-check", action="store_true",
                   help="Offline: fake encoder + small workload, exercises the real "
                        "process fan-out and schema without downloading a model")
    args = p.parse_args()

    if args.self_check:
        args.dataset = "synthetic"
        args.n_prompts = min(args.n_prompts, 40)
        levels = [1, 4]
        factory = _FakeFactory()

        def make_seq():
            return _FakeEmbedding()
    else:
        levels = [int(x) for x in args.concurrency_levels.split(",") if x.strip()]
        factory = _SBERTFactory(args.embed_model)

        def make_seq():
            from gptcache.embedding import SBERT
            return SBERT(args.embed_model)

    prompts = load_prompts(args.dataset, args.n_prompts, args.seed)
    n = len(prompts)
    print(f"Dispatcher benchmark: dataset={args.dataset} n_prompts={n} "
          f"levels={levels}{' [self-check]' if args.self_check else ''}")
    if not _HAVE_PSUTIL:
        print("  (psutil not installed -- rss_mb will be 0.0; `pip install psutil` for memory numbers)")

    results = []
    for c in levels:
        print(f"\n=== concurrency {c} ===")

        # --- sequential: one model instance, callers serialize on it ---
        seq_model = make_seq()
        seq_model.to_embeddings("warmup")                # exclude cold-start from timing
        seq = run_concurrent(seq_model.to_embeddings, prompts, c)
        seq["rss_mb"] = _rss_mb()
        del seq_model

        # --- dispatcher: fan across worker processes ---
        disp = EmbeddingDispatcher(factory, num_workers=args.num_workers)
        num_workers = disp.num_workers
        disp.to_embeddings("warmup")                     # build the per-worker models first
        d = run_concurrent(disp.to_embeddings, prompts, c)
        d["rss_mb"] = _rss_mb()                           # measured while the pool is alive
        disp.shutdown()

        speedup = seq["elapsed_s"] / d["elapsed_s"] if d["elapsed_s"] > 0 else float("inf")
        print(f"  sequential  {seq['elapsed_s']:.2f}s  {seq['throughput_qps']:6.1f} qps  "
              f"p50 {seq['p50_ms']:.1f}ms  rss {seq['rss_mb']:.0f}MB")
        print(f"  dispatcher  {d['elapsed_s']:.2f}s  {d['throughput_qps']:6.1f} qps  "
              f"p50 {d['p50_ms']:.1f}ms  rss {d['rss_mb']:.0f}MB  ({num_workers} workers)")
        verdict = "WINS" if speedup > 1 else "loses"
        print(f"  speedup {speedup:.3f}x  ({verdict})")

        results.append({
            "concurrency": c,
            "n_prompts": n,
            "num_workers": num_workers,
            "sequential": seq,
            "dispatcher": d,
            "speedup": speedup,
        })

    out_path = args.out or os.path.join(args.workdir, "results.json")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"args": vars(args), "results": results}, f, indent=2)
    print(f"\nResults -> {out_path}")

    if args.self_check:
        # Fail loudly if the plumbing broke: every cell must be fully populated.
        for r in results:
            for mode in ("sequential", "dispatcher"):
                assert r[mode]["elapsed_s"] > 0, f"{mode} produced no timing"
                for k in ("throughput_qps", "p50_ms", "p99_ms", "rss_mb"):
                    assert k in r[mode], f"missing {k} in {mode}"
            assert r["speedup"] > 0
        print("self-check OK: schema populated, fan-out ran across processes.")


if __name__ == "__main__":
    main()
