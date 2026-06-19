"""Cost-Aware W-TinyLFU eviction policy with EWMA time-decay frequency.

Combines the W-TinyLFU structure (Window-LRU + SLRU Probation/Protected +
Count-Min Sketch + Doorkeeper) with a lexicographic admission score that
prefers items with higher regeneration cost when frequency is equal.

Admission score (lexicographic — frequency always dominates):
    freq_score  = min(ewma_freq, 15.0)           per-item, time-decayed
    cost_score  = EWMACostTracker.score(cost)     log-cost z-score in [0, 15]
    score       = freq_score * 16.0 + cost_score

    1 unit of frequency (= 16 in score) beats the full cost range (max 15),
    so a twice-accessed cheap item always wins over a once-accessed expensive
    item. Cost only breaks ties within the same frequency level.

Frequency decay (our unique contribution vs. related work):
    f' = min(f * exp(-λ * dt) + 1, 15)
where dt = seconds since last access (monotonic clock). Every item decays
independently — no global reset schedule. A once-hot item that has not been
touched for 19 hours (default λ = 1e-5) loses frequency organically.

Cost normalization (prevents 600× raw-cost spread from overwhelming frequency):
    log(cost) → EWMA(mean, variance) → z-score → clamp[-1,1] → [0, 15]
    Returns 8.0 (neutral) during the first `ewma_warmup` observations.

Doorkeeper (Bloom filter, from TinyLFU paper):
    First access to a key: register in filter, skip sketch increment.
    Second+ access: pass through to sketch.
    This suppresses one-hit wonders — items seen once and never again —
    from inflating sketch counters and polluting the admission decision.
    Cleared whenever the sketch resets to prevent stale false positives.
"""

import math
import random
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional


DEFAULT_GENERATION_LATENCY_MS = 1000.0
DEFAULT_TOKEN_COUNT = 100
DEFAULT_MODEL_TIER = 1.0
DEFAULT_RESPONSE_SIZE_BYTES = 500

EWMA_FREQ_CAP = 15.0  # aligns freq_score with the [0, 15] cost_score range

# Caffeine's ADMIT_HASHDOS_THRESHOLD adapted for EWMA freq (cap=15, half≈7).
# A candidate with ewma_freq >= this gets a 1/128 random admission chance
# even if it loses the score contest, preventing frequency-flooding attacks.
_HASHDOS_THRESHOLD = 7.0


@dataclass
class LLMCost:
    """Per-item regeneration cost signal.

    `cost` = latency_ms × model_tier × (1 + tokens/1000), capturing both
    wall-clock delay and model pricing tier in a single scalar.

    Defaults give every item an identical cost, collapsing the policy to
    W-TinyLFU + EWMA decay — still better than LRU before Phase 4 plumbs
    real latency/token counts through the adapter.
    """

    generation_latency_ms: float = DEFAULT_GENERATION_LATENCY_MS
    token_count: int = DEFAULT_TOKEN_COUNT
    model_tier: float = DEFAULT_MODEL_TIER
    response_size_bytes: int = DEFAULT_RESPONSE_SIZE_BYTES

    @property
    def cost(self) -> float:
        return (
            self.generation_latency_ms
            * self.model_tier
            * (1.0 + self.token_count / 1000.0)
        )


@dataclass
class ItemMeta:
    """Per-item state for the lexicographic scoring function."""

    key: Any
    ewma_freq: float = 0.0   # time-decayed access frequency, capped at EWMA_FREQ_CAP
    cost: float = 0.0        # LLMCost.cost snapshot at last insert/update
    size: int = 1            # response_size_bytes (kept for potential future use)
    last_access: float = 0.0 # time.monotonic() of last touch


class Doorkeeper:
    """Bloom filter that suppresses one-hit wonders from the sketch.

    A key must be seen at least twice before its Count-Min Sketch counters
    are incremented. Items accessed only once — the long tail of one-off
    queries — never pollute the frequency estimator.

    Cleared on every sketch reset so stale membership does not persist
    across aging cycles (per the TinyLFU paper, Einziger et al. 2017).

    Implementation uses a bytearray bit-array with double-hashing to avoid
    any numpy dependency in this module.
    """

    def __init__(self, capacity: int, fp_rate: float = 0.01):
        capacity = max(capacity, 16)
        # Optimal Bloom sizing: m = -n·ln(p) / (ln2)², k = (m/n)·ln2
        m = int(-capacity * math.log(fp_rate) / (math.log(2) ** 2))
        self._num_bits = max(m, 64)
        self._num_hashes = max(int((self._num_bits / capacity) * math.log(2)), 1)
        self._bits = bytearray((self._num_bits + 7) // 8)

    def allow_and_add(self, key_hash: int) -> bool:
        """Return True if key was already present; always add key to filter."""
        present = self._contains(key_hash)
        self._add(key_hash)
        return present

    def clear(self) -> None:
        for i in range(len(self._bits)):
            self._bits[i] = 0

    def _contains(self, h: int) -> bool:
        for i in range(self._num_hashes):
            pos = self._hash_pos(h, i)
            if not (self._bits[pos >> 3] & (1 << (pos & 7))):
                return False
        return True

    def _add(self, h: int) -> None:
        for i in range(self._num_hashes):
            pos = self._hash_pos(h, i)
            self._bits[pos >> 3] |= 1 << (pos & 7)

    def _hash_pos(self, h: int, i: int) -> int:
        h1 = h & 0xFFFFFFFF
        h2 = (h >> 32) & 0xFFFFFFFF
        return ((h1 + i * h2) & 0xFFFFFFFFFFFFFFFF) % self._num_bits


class EWMACostTracker:
    """Maps LLM regeneration costs to [0, 15] via EWMA z-score normalization.

    Pipeline per access:
        cost  →  log(cost)  →  EWMA(mean, variance)  →  z-score  →
        clamp[-1, 1]  →  [0, 15]

    Log-transform compresses the heavy-tailed cost distribution (token counts
    span 10–4000+, latency spans 200ms–15s, producing raw costs with a ~600×
    spread). Without this, raw cost would dominate the score regardless of
    the lexicographic structure.

    EWMA alpha=0.05 gives an effective window of ~20 samples — slow enough
    to be stable, fast enough to track workload shifts over hours.

    During the first `warmup` observations the distribution is unreliable,
    so score() returns 8.0 (the neutral midpoint of [0, 15]).
    """

    def __init__(self, alpha: float = 0.05, warmup: int = 20):
        self._alpha = alpha
        self._warmup = warmup
        self._count = 0
        self._mean = 0.0
        self._variance = 1.0

    def update(self, cost: float) -> None:
        log_c = math.log(max(cost, 1.0))
        if self._count == 0:
            self._mean = log_c
        else:
            delta = log_c - self._mean
            self._mean += self._alpha * delta
            self._variance = (1.0 - self._alpha) * (
                self._variance + self._alpha * delta * delta
            )
        self._count += 1

    def score(self, cost: float) -> float:
        """Return cost score in [0.0, 15.0]. Returns 8.0 during warmup."""
        if self._count < self._warmup:
            return 8.0
        log_c = math.log(max(cost, 1.0))
        std = math.sqrt(max(self._variance, 1e-10))
        z = (log_c - self._mean) / std
        z = max(-1.0, min(1.0, z))
        return (z + 1.0) * 7.5  # [-1, 1] → [0, 15]


class CountMinSketch:
    """Count-Min Sketch with periodic halving (W-TinyLFU frequency filter).

    increment() returns True when a halve occurred so the caller can clear
    the Doorkeeper in sync with the aging cycle.
    """

    def __init__(self, width: int = 2048, depth: int = 4, seed: int = 0):
        if width <= 0 or depth <= 0:
            raise ValueError("width and depth must be positive")
        self._width = width
        self._depth = depth
        self._table = [[0] * width for _ in range(depth)]
        self._seeds = [seed + i * 0x9E3779B1 for i in range(depth)]
        self._reset_threshold = 10 * width
        self._total = 0

    def _index(self, row: int, key: Any) -> int:
        return (hash((self._seeds[row], key)) & 0x7FFFFFFF) % self._width

    def increment(self, key: Any) -> bool:
        """Increment counters for key. Returns True if a halve occurred."""
        for r in range(self._depth):
            self._table[r][self._index(r, key)] += 1
        self._total += 1
        if self._total >= self._reset_threshold:
            self._halve()
            return True
        return False

    def estimate(self, key: Any) -> int:
        return min(self._table[r][self._index(r, key)] for r in range(self._depth))

    def _halve(self) -> None:
        for r in range(self._depth):
            row = self._table[r]
            for i in range(self._width):
                row[i] >>= 1
        self._total >>= 1


class _LRUSegment:
    """OrderedDict-backed LRU with peek_victim / evict_victim helpers."""

    def __init__(self, maxsize: int):
        self.maxsize = max(1, maxsize)
        self._od: "OrderedDict[Any, bool]" = OrderedDict()

    def __len__(self) -> int:
        return len(self._od)

    def __contains__(self, key: Any) -> bool:
        return key in self._od

    def is_full(self) -> bool:
        return len(self._od) >= self.maxsize

    def add_mru(self, key: Any) -> None:
        self._od[key] = True
        self._od.move_to_end(key, last=True)

    def touch(self, key: Any) -> None:
        if key in self._od:
            self._od.move_to_end(key, last=True)

    def peek_victim(self) -> Optional[Any]:
        if not self._od:
            return None
        return next(iter(self._od))

    def evict_victim(self) -> Optional[Any]:
        if not self._od:
            return None
        key, _ = self._od.popitem(last=False)
        return key

    def remove(self, key: Any) -> bool:
        return self._od.pop(key, None) is not None


class CostAwareWTinyLFU:
    """Cost-Aware W-TinyLFU with EWMA time-decay and lexicographic scoring.

    Structure:
        window LRU    (~1% of maxsize)   — new items land here unconditionally
        probation LRU (~20% of main)     — admitted but not yet proven
        protected LRU (~80% of main)     — items that earned a second access

    Insert flow:
        1. Register cost in EWMACostTracker; register key in Doorkeeper.
        2. Add to window.
        3. If window was full: pop its LRU victim → admission contest.

    Admission contest (window victim vs. probation LRU end):
        winner = argmax(score); loser → on_evict callback + meta removed.
        score = freq_score * 16 + cost_score  (lexicographic)

    Get flow:
        1. Doorkeeper-gated sketch increment + EWMA freq update.
        2. Segment routing: protected → touch; probation → promote; window → touch.

    Public API matches MemoryCacheEviction (put / get / policy) for Phase 2 routing.
    """

    def __init__(
        self,
        maxsize: int,
        clean_size: int = 1,
        on_evict: Optional[Callable[[List[Any]], None]] = None,
        decay_rate: float = 1e-5,
        sketch_width: int = 2048,
        sketch_depth: int = 4,
        window_ratio: float = 0.01,
        protected_ratio: float = 0.8,
        ewma_alpha: float = 0.05,
        ewma_warmup: int = 20,
        time_fn: Callable[[], float] = time.monotonic,
        default_cost: Optional[LLMCost] = None,
        cost_aware: bool = True,
        freq_weight: float = 16.0,
        **_unused,
    ):
        if maxsize < 4:
            raise ValueError(
                "maxsize must be >= 4 for W-TinyLFU to allocate all three segments"
            )
        self._maxsize = maxsize
        self._clean_size = max(1, clean_size if clean_size is not None else 1)
        self._on_evict = on_evict or (lambda keys: None)
        self._decay_rate = decay_rate
        self._time = time_fn
        self._default_cost = default_cost or LLMCost()
        # cost_aware=False collapses the score to frequency-only, reproducing a
        # plain W-TinyLFU (the prior-art baseline). freq_weight controls how
        # strongly frequency dominates cost: 16.0 (>= cost range of 15) keeps the
        # original lexicographic behaviour; lower values blend cost into the
        # decision so it influences eviction beyond mere tie-breaking.
        self._cost_aware = cost_aware
        self._freq_weight = freq_weight

        window_size = max(1, int(maxsize * window_ratio))
        main_size = maxsize - window_size
        protected_size = max(1, int(main_size * protected_ratio))
        probation_size = max(1, main_size - protected_size)

        self._window = _LRUSegment(window_size)
        self._probation = _LRUSegment(probation_size)
        self._protected = _LRUSegment(protected_size)

        self._meta: Dict[Any, ItemMeta] = {}
        self._sketch = CountMinSketch(sketch_width, sketch_depth)
        self._doorkeeper = Doorkeeper(capacity=maxsize)
        self._cost_tracker = EWMACostTracker(alpha=ewma_alpha, warmup=ewma_warmup)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def policy(self) -> str:
        return "CA_W_TINYLFU"

    def __len__(self) -> int:
        return len(self._window) + len(self._probation) + len(self._protected)

    def __contains__(self, key: Any) -> bool:
        return key in self._meta

    def put(self, objs: List[Any], costs: Optional[List[LLMCost]] = None) -> None:
        if costs is None:
            costs = [self._default_cost] * len(objs)
        elif len(costs) != len(objs):
            raise ValueError("costs length must match objs length")
        for obj, cost in zip(objs, costs):
            self._insert(obj, cost)

    def get(self, obj: Any) -> Optional[bool]:
        if obj not in self._meta:
            return None
        self._record_access(obj)
        self._touch_segments(obj)
        return True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _record_access(self, key: Any) -> None:
        """Doorkeeper-gated sketch increment + per-item EWMA freq decay."""
        k_hash = hash(key)
        if self._doorkeeper.allow_and_add(k_hash):
            reset = self._sketch.increment(key)
            if reset:
                self._doorkeeper.clear()

        m = self._meta[key]
        now = self._time()
        dt = max(0.0, now - m.last_access)
        decay = math.exp(-self._decay_rate * dt) if dt > 0 else 1.0
        m.ewma_freq = min(EWMA_FREQ_CAP, m.ewma_freq * decay + 1.0)
        m.last_access = now

    def _score(self, key: Any) -> float:
        """Admission score: frequency weighted by ``freq_weight``, plus cost.

        Both components are in [0, 15]. With ``freq_weight = 16`` (default) one
        unit of freq_score outweighs the full cost range (max 15), so the score
        is lexicographic — a twice-accessed cheap item always beats a
        once-accessed expensive one and cost only breaks exact ties. Lower
        ``freq_weight`` blends cost into the ordering so it can override small
        frequency differences. ``cost_aware=False`` drops the cost term
        entirely, reproducing a plain frequency-only W-TinyLFU.
        """
        m = self._meta[key]
        freq_score = m.ewma_freq  # in [0, EWMA_FREQ_CAP] = [0, 15]
        score = freq_score * self._freq_weight
        if self._cost_aware:
            score += self._cost_tracker.score(m.cost)  # cost_score in [0, 15]
        return score

    def _touch_segments(self, key: Any) -> None:
        if key in self._protected:
            self._protected.touch(key)
        elif key in self._probation:
            self._probation.remove(key)
            self._promote_to_protected(key)
        elif key in self._window:
            self._window.touch(key)

    def _insert(self, key: Any, cost: LLMCost) -> None:
        self._cost_tracker.update(cost.cost)

        if key in self._meta:
            m = self._meta[key]
            m.cost = cost.cost
            m.size = max(1, cost.response_size_bytes)
            self._record_access(key)
            self._touch_segments(key)
            return

        now = self._time()
        self._meta[key] = ItemMeta(
            key=key,
            ewma_freq=0.0,
            cost=cost.cost,
            size=max(1, cost.response_size_bytes),
            last_access=now,
        )
        self._record_access(key)  # sets ewma_freq → 1.0, registers in doorkeeper

        if self._window.is_full():
            window_victim = self._window.evict_victim()
            if window_victim is not None:
                self._admit_or_reject(window_victim)
        self._window.add_mru(key)

    def _admit_or_reject(self, candidate: Any) -> None:
        if not self._probation.is_full():
            self._probation.add_mru(candidate)
            return

        victim = self._probation.peek_victim()
        if victim is None:
            self._probation.add_mru(candidate)
            return

        # Strict greater-than: on a tie the victim (proven, in main) wins over
        # the candidate (unproven, from window). Matches Caffeine's admit().
        candidate_score = self._score(candidate)
        victim_score = self._score(victim)
        if candidate_score > victim_score:
            self._probation.evict_victim()
            self._emit_evict([victim])
            self._probation.add_mru(candidate)
        elif self._meta[candidate].ewma_freq >= _HASHDOS_THRESHOLD:
            # Hash-DoS defence: a moderately warm candidate that loses on score
            # gets a 1/128 random admission chance. Prevents an attacker from
            # pinning the victim by artificially inflating its frequency.
            # Matches Caffeine's ADMIT_HASHDOS_THRESHOLD logic.
            if random.randint(0, 127) == 0:
                self._probation.evict_victim()
                self._emit_evict([victim])
                self._probation.add_mru(candidate)
            else:
                self._emit_evict([candidate])
        else:
            self._emit_evict([candidate])

    def _promote_to_protected(self, key: Any) -> None:
        if self._protected.is_full():
            demoted = self._protected.evict_victim()
            if demoted is not None:
                self._admit_or_reject(demoted)
        self._protected.add_mru(key)

    def _emit_evict(self, keys: List[Any]) -> None:
        for k in keys:
            self._meta.pop(k, None)
        if keys:
            self._on_evict(keys)
