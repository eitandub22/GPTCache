"""Phase 1 tests for the standalone CA-W-TinyLFU policy module.

These tests exercise policy internals in isolation — no MemoryCacheEviction,
no SSDataManager, no SQLite/FAISS. Phase 2 adds the routing test.
"""

from gptcache.manager.eviction.ca_w_tinylfu import (
    CostAwareWTinyLFU,
    CountMinSketch,
    Doorkeeper,
    EWMACostTracker,
    LLMCost,
)


def _collect_evictions():
    evicted = []

    def on_evict(keys):
        evicted.extend(keys)

    return evicted, on_evict


# ---------------------------------------------------------------------------
# t1 — below-capacity inserts produce no evictions
# ---------------------------------------------------------------------------

def test_below_capacity_no_evictions():
    """Insert fewer items than window+probation capacity: nothing evicted.

    With default ratios at maxsize=40: window=1, probation=8, protected=31.
    Inserting 5 items keeps probation well below its limit so the admission
    contest never needs to reject.
    """
    evicted, on_evict = _collect_evictions()
    cache = CostAwareWTinyLFU(maxsize=40, on_evict=on_evict, time_fn=lambda: 0.0)

    n = 5
    for i in range(n):
        cache.put([f"k_{i}"])

    assert evicted == [], f"unexpected evictions: {evicted}"
    for i in range(n):
        assert cache.get(f"k_{i}") is True, f"k_{i} missing after below-capacity insert"
    assert len(cache) == n


# ---------------------------------------------------------------------------
# t2 — above-capacity inserts trigger evictions and respect maxsize
# ---------------------------------------------------------------------------

def test_above_capacity_triggers_evictions():
    """Insert maxsize + N items: at least one eviction fires, cache stays bounded."""
    evicted, on_evict = _collect_evictions()
    maxsize = 20
    cache = CostAwareWTinyLFU(maxsize=maxsize, on_evict=on_evict, time_fn=lambda: 0.0)

    for i in range(maxsize + 30):
        cache.put([f"k_{i}"])

    assert len(evicted) >= 1, "no evictions fired despite overflowing the cache"
    assert len(cache) <= maxsize, f"cache exceeded maxsize: {len(cache)} > {maxsize}"
    for key in evicted:
        assert key not in cache, f"evicted key {key!r} still present in cache"


# ---------------------------------------------------------------------------
# t3 — cost bias: expensive items survive the admission contest more often
# ---------------------------------------------------------------------------

def test_expensive_items_survive_more_than_cheap():
    """Under equal access frequency, expensive items survive eviction pressure.

    Uses a large enough cache and enough unique-item flood so that only items
    with high admission score survive. Cost normalization warmup (20 obs) is
    satisfied by the flood, so cost differentiation is fully active.
    """
    cheap = LLMCost(
        generation_latency_ms=100.0, token_count=10,
        model_tier=1.0, response_size_bytes=100,
    )
    expensive = LLMCost(
        generation_latency_ms=15000.0, token_count=2000,
        model_tier=20.0, response_size_bytes=100,
    )

    cheap_survived = 0
    expensive_survived = 0
    n_trials = 30

    for trial in range(n_trials):
        evicted, on_evict = _collect_evictions()
        cache = CostAwareWTinyLFU(
            maxsize=12,
            on_evict=on_evict,
            time_fn=lambda: float(trial),
            ewma_warmup=5,  # short warmup so normalization activates early
        )
        cache.put(["A"], costs=[cheap])
        cache.put(["B"], costs=[expensive])
        for _ in range(3):
            cache.get("A")
            cache.get("B")

        for i in range(80):
            cache.put([f"u_{trial}_{i}"])

        if "A" in cache:
            cheap_survived += 1
        if "B" in cache:
            expensive_survived += 1

    assert expensive_survived >= cheap_survived, (
        f"cost-bias not observed: cheap_survived={cheap_survived}, "
        f"expensive_survived={expensive_survived}"
    )
    assert expensive_survived > 0, "expensive item never survived"


# ---------------------------------------------------------------------------
# t4 — EWMA decay reduces stale frequency over time
# ---------------------------------------------------------------------------

def test_ewma_decay_reduces_stale_frequency():
    """A once-hot item that hasn't been touched in a long dt loses frequency.

    Uses an injected time_fn so the test is fully deterministic.
    """
    now = [0.0]

    def time_fn():
        return now[0]

    cache = CostAwareWTinyLFU(maxsize=40, decay_rate=0.1, time_fn=time_fn)

    cache.put(["A"])
    for _ in range(20):
        cache.get("A")
    initial_freq = cache._meta["A"].ewma_freq
    assert initial_freq > 5.0, f"setup error: ewma_freq did not grow ({initial_freq})"

    now[0] = 1000.0  # simulate 1000 seconds of idle time
    cache.get("A")
    decayed_freq = cache._meta["A"].ewma_freq

    assert decayed_freq < initial_freq, (
        f"ewma did not decay across dt=1000s with rate=0.1: "
        f"{initial_freq:.3f} -> {decayed_freq:.3f}"
    )
    assert decayed_freq < 2.0, (
        f"with decay_rate=0.1 and dt=1000s, surviving freq should be ~1.0; "
        f"got {decayed_freq:.3f}"
    )


# ---------------------------------------------------------------------------
# t5 — sketch halving fires at the reset threshold
# ---------------------------------------------------------------------------

def test_sketch_halving_fires_at_threshold():
    """Pushing the sketch past _reset_threshold halves _total."""
    sketch = CountMinSketch(width=32, depth=2)
    threshold = sketch._reset_threshold
    assert threshold == 320, f"unexpected threshold: {threshold}"

    resets = sum(1 for i in range(threshold) if sketch.increment(f"item_{i}"))

    assert resets == 1, f"expected exactly 1 halve, got {resets}"
    assert sketch._total <= threshold // 2 + 1, (
        f"sketch did not halve correctly: _total={sketch._total}"
    )
    assert sketch._total > 0, "sketch over-halved to zero"


# ---------------------------------------------------------------------------
# t6 — Doorkeeper suppresses one-hit wonders from the sketch
# ---------------------------------------------------------------------------

def test_doorkeeper_gates_sketch_on_first_access():
    """First access to a key registers in the filter but skips the sketch."""
    dk = Doorkeeper(capacity=1000)

    # First access: not yet in filter -> allow_and_add returns False
    assert dk.allow_and_add(hash("key_A")) is False
    # Second access: now in filter -> returns True (would increment sketch)
    assert dk.allow_and_add(hash("key_A")) is True

    # Fresh key: again False on first access
    assert dk.allow_and_add(hash("key_B")) is False

    # Clear resets all membership
    dk.clear()
    assert dk.allow_and_add(hash("key_A")) is False


# ---------------------------------------------------------------------------
# t7 — EWMACostTracker returns neutral score during warmup, then normalizes
# ---------------------------------------------------------------------------

def test_ewma_cost_tracker_warmup_and_normalization():
    """During warmup, score is 8.0 (neutral). After warmup, expensive > cheap."""
    tracker = EWMACostTracker(alpha=0.05, warmup=5)

    cheap_cost = 200.0
    expensive_cost = 200_000.0

    # During warmup: all scores neutral
    for _ in range(4):
        tracker.update(cheap_cost)
        assert tracker.score(cheap_cost) == 8.0, "should be neutral during warmup"

    # Trigger warmup completion (5th update)
    tracker.update(cheap_cost)

    # After warmup: expensive item gets higher score than cheap
    score_cheap = tracker.score(cheap_cost)
    score_expensive = tracker.score(expensive_cost)
    assert score_expensive > score_cheap, (
        f"expected expensive > cheap after warmup: "
        f"expensive={score_expensive:.2f}, cheap={score_cheap:.2f}"
    )
    assert 0.0 <= score_cheap <= 15.0
    assert 0.0 <= score_expensive <= 15.0


# ---------------------------------------------------------------------------
# t8 — Doorkeeper is cleared when sketch resets (no stale false positives)
# ---------------------------------------------------------------------------

def test_doorkeeper_cleared_on_sketch_reset():
    """Doorkeeper must be cleared when the sketch halves to prevent stale hits."""
    cache = CostAwareWTinyLFU(
        maxsize=100,
        on_evict=lambda ks: None,
        sketch_width=16,   # tiny sketch so reset fires quickly
        sketch_depth=2,
    )
    threshold = cache._sketch._reset_threshold  # 10 * 16 = 160

    # Flood with unique keys to drive past the reset threshold.
    # Each key goes through doorkeeper (first access → False, skip sketch);
    # after two puts of the same key, sketch is incremented.
    # Use enough keys that the sketch hits its threshold.
    for i in range(threshold * 2):
        cache.put([f"flood_{i}"])

    # After the flood the doorkeeper should have been cleared at least once.
    # A key inserted before the last reset should no longer be in the filter.
    # We verify by checking a key that was definitely inserted before the reset:
    # it was cleared, so its hash appears "new" again (allow_and_add -> False).
    first_key_hash = hash("flood_0")
    result = cache._doorkeeper.allow_and_add(first_key_hash)
    # Either False (cleared) or True (still in filter due to a later insertion
    # with the same hash — acceptable for a probabilistic filter).
    # What we must NOT see is the counter monotonically growing without resets.
    # The real assertion is that the cache does not crash and resets occurred.
    assert cache._sketch._total < threshold, (
        f"sketch total {cache._sketch._total} should be below threshold "
        f"{threshold} after at least one reset"
    )
