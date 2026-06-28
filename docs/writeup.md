# Cost-Aware W-TinyLFU for Semantic LLM Caches — write-up

> **STATUS: skeleton for confirmation.** No prose yet. Each section lists the claim it
> makes and the *exact* evidence it cites (phase + file), so every empirical statement is
> traceable to a measured result before we write it. Confirm structure + scope, then we fill prose.

---

## 0. Open questions to settle before writing (see chat)
- **Format/length:** thesis chapter (deep, ~15–25pp) vs conference-paper style (tight, ~8pp)?
- **Storage scope:** ✅ IN SCOPE — second contribution (like the multi-aspect example write-up). §9
  is now a full section, not optional.
- **Format/voice:** to match last year's example write-up (user providing). Don't draft prose until seen.
- **Audience:** thesis committee vs workshop reviewers (changes how much GPTCache background to spell out).

---

## 1. Abstract
One paragraph. Claims, each gated on a measured number:
- Cost-aware value-weighted eviction (CA_W_TINYLFU) for semantic LLM caches.
- Beats LRU on cost-weighted hit rate in **both** regimes a cache faces: drift and stationary skew.
- Drift: paired ADAPT−LRU +3.6/+4.1pp cost_wt, p≈0.001 (n=7).  *[Phase 3.6]*
- Stationary: paired CA−LRU positive in 18/18 cells, +5…+18pp cost_wt (n=3).  *[Phase 0.6]*
- Honest scope: the *cost term itself* pays under sharp skew, is noisy under flat skew.

## 2. Introduction / motivation
- Semantic caching for LLMs (GPTCache): embed query → ANN search → return cached answer on hit.
- **Why eviction matters here and is different:** LLM responses have *heterogeneous regeneration cost*
  (latency, output tokens, model tier). A miss on an expensive answer costs more than a miss on a cheap
  one — so hit *rate* is the wrong objective; **cost-weighted** hit rate is.
- Gap: GPTCache shipped only recency/frequency eviction (cachetools LRU/LFU), cost-blind.
- Contribution bullets (forward-refs to §4, §6).

## 3. Background
- **3.1 GPTCache request flow** — pre-embed → embed → search → evaluate → post → save (1 fig).
- **3.2 The eviction layer** — key-set mirror, `on_evict` soft-deletes in scalar+vector. *[research.md §1.1]*
- **3.3 W-TinyLFU / Caffeine** — window-LRU + SLRU main + Count-Min sketch + admission filter + doorkeeper.
- **3.4 Cost-aware caching prior art** — GDSF / GreedyDual lineage; what's new is fusing it into TinyLFU
  admission for a *semantic* cache.

## 4. Design — CA_W_TINYLFU
- **4.1 Cost model** — `LLMCost(latency, tokens≈chars/4, model_tier)`; how a cost reaches the policy
  (`_build_llm_cost` → `save` → `eviction.put(costs=)`). *[research.md §1.2]*
- **4.2 Value-weighted admission/eviction** — frequency estimate × cost → the admission contest;
  one item evicted per admission (why `clean_size` is not forwarded). *[ca_w_tinylfu.py; Phase 4]*
- **4.3 Adaptive window (ADAPT)** — window grows/shrinks to chase drift; the drift-regime variant.
- **4.4 Time-decay** — read-time EWMA discount on freq score, dormant at `virtual-clock-sec=0`;
  the knob that attacks stale-frequency pollution under drift. *[Phase 3]*
- **4.5 Hash-DoS defense** — seeded sketch/doorkeeper hashing. *[commit 34507bc]*

## 5. Implementation notes
- Plumbing is optional/degrades cleanly (`_build_llm_cost` returns None if module absent). *[research.md §1.2]*
- Only CA reads `costs`; cachetools policies ignore them — zero overhead when unused.
- Test surface + the PYTHONHASHSEED flake note (statistical tests). *[test_ca_w_tinylfu.py]*

## 6. Experimental setup
- **6.1 Harness & data** — `benchmark_lmsys.py`, lmsys-chat replay; cost tiers by model name.
- **6.2 Metrics** — cost_weighted_hit_rate (primary), hit_rate, token_saving_ratio.
- **6.3 Two regimes** — drift (`--drift-rotate N>0`, rotating hot-set = LRU's bet) vs stationary
  (`--drift-rotate 0`, fixed Zipf, working-set > cache = TinyLFU's regime).
- **6.4 Policies** — LRU, LFU, WTINYLFU_FREQ (freq-only twin), CA_W_TINYLFU, CA_W_TINYLFU_ADAPT.
- **6.5 Statistics** — seeds are **paired** (same seed = same stream); we report paired deltas +
  sign-consistency, not unpaired ±std. Why (the z11 ±9–14pp variance is cross-seed difficulty). *[paired.py]*

## 7. Results
- **7.1 Drift regime — the headline LRU upset.** Decay tuned to drift rate → ADAPT beats LRU on all
  three metrics; paired cost_wt +3.63/+4.10pp, t≈5.9/6.3, 7/7 seeds, p≈0.001 fast+slow. *[Phase 3.5/3.6]*
  - Table: drift_e2/e3 aggregate; the decay sweep (vc0/vc30/vc120) showing decay is the lever. *[Phase 3]*
- **7.2 Stationary regime — frequency/CA crush LRU.** Paired CA−LRU positive 3/3 in all 18 cells,
  cost_wt +5…+18pp, hit% + tok% too. *[Phase 0.6, win_z{11,15}, paired.py]*
  - Table: the 6-cell paired CA−LRU grid (already in plan.md Phase 0.6).
- **7.3 Cost-isolation ablation (CA − WTINYLFU_FREQ) — the honest nuance.** Cost term pays + is
  sign-consistent under sharp skew (z15, 3/3 all cells), noisy under flat skew (z11, only cs100 clean);
  trades hit% for cost-weighted value by design (hit% flat/down, tok% up). *[Phase 0.6 Table 2 + paired]*
- **7.4 Crossover surface** — one figure: LRU's edge confined to churning hot-sets; decay claws back
  even that; everywhere else value-aware frequency wins.
- **7.5 The cost_priority dial — a statistically clean money↔hit Pareto knob (drift, cs200, paired
  n=7).** Per-seed paired runs (seeds 0–6) collapse the **±10–13pp cross-seed cost_wt spread** (the
  §6.5 difficulty variance) that made the single-seed sweep unreliable — paired means: LFU 61.1,
  cp0 67.4 … cp1 69.9. Headline: the dial **endpoints** are a clean Pareto trade — cp0→cp1 moves
  cost-weighted hit rate **+2.47pp (t=4.7, 7/7 seeds)** while spending raw hit% **−4.13pp (t=−10.1,
  0/7)**; token-saving flat (+0.45pp, 5/7, ns). "Use CA at all" is the dominant win: **every** dial
  point beats LFU on cost_wt **+6.3…+8.8pp (t=3.1–4.5, 6–7/7)** and on token-saving (7/7). Honest
  corrections to the single-seed read (both falsify the earlier "every point beats LFU on all three"):
  (i) cp1 does **not** beat LFU on raw hit% (paired −0.43pp, 2/7, ns) — at full cost-priority CA gives
  up its hit-rate edge; (ii) the dial's **interior** is within noise (adjacent steps 2–5/7
  sign-consistent), so `cost_priority` is a coarse high/low knob, **not** a fine monotone control —
  only the endpoints separate cleanly. p50/p95 and mem flat across the sweep — the knob is free.
  *[bench_cost_priority/cp_seed{0..6}.json]*

## 8. Discussion / threats to validity
- Where cost-awareness does **not** help (flat skew) and why — stated up front, not buried.
- Single-config recommendation: **CA_W_TINYLFU + adaptive-window on** (wins drift, strongest stationary).
- Limitations: cost ≈ chars/4 token proxy; n=3 stationary / n=7 drift; one dataset (lmsys);
  in-memory eviction only (Redis path delegates).

## 9. Storage co-contribution — SBERTMRL + HNSW/SQ8
- **9.1 SBERTMRL** — MRL-truncated embeddings: slice to `target_dim`, re-normalize. *[sbert_mrl.py]*
- **9.2 Faiss HNSW + SQ8** — approximate index + 8-bit quantization; tombstone deletion + rebuild. *[faiss.py]*
- **9.3 Result** — measured **5.7× (SQ8 M=32) → 7.5× (SQ8 M=16) → 9.8× (PQ)** index-RAM compression
  vs the ONNX/768/Flat baseline, with ~100× faster *search*; **Pareto-frontier framing** (E=max
  compression, G=recall ceiling). *[bench_real_100k/results.json, frontier.md]*
- **9.3.1 Accuracy is not the cost it first looks like (threshold sweep + isolation cell I).** The
  single-threshold FP gap (18.5%→24.8–40.9%) is **largely a threshold artifact**: the global 0.90 was
  tuned for the 768-d baseline. At a per-cell threshold matched to A's precision (~0.83) the MRL cells
  hold TP ≥ baseline (E @0.92: TP 0.882 @ prec 0.864; G @0.94: TP 0.900 @ prec 0.838) at 7.5–9.8× less
  RAM. Isolation cell I (MRL/768/Flat) attributes the FP rise to the **encoder swap (+14.4pp)**, not
  truncation (I→G +7.9pp; I→E −8.0pp via PQ). *[frontier.md §isolation, §sweep]*
- **9.3.2 The genuine costs.** (i) **e2e latency rises** (A 59 → E/G 97–98 → I 117 ms p95) — the
  ~100× search win is real but invisible at e2e because the MRL encoder is slower than ONNX and
  embedding dominates; quote search-p95 and e2e-p95 separately. (ii) **Tombstone steady-state** under
  sustained eviction (§9.2 / walkthrough §5). Both are fresh-index, no-eviction measurements.
  *[bench_real_100k/results.json e2e_latency_ms]*

## 10. Conclusion
- Two external LRU-beats on complementary regimes + an honest map of when the cost term pays.

---

## Evidence index (every number above traces here)
| Claim | Source |
|---|---|
| drift paired +3.6/+4.1pp p≈0.001 n=7 | plan.md Phase 3.6; bench_lmsys/drift_e{2,3}_seed*.json |
| stationary paired CA−LRU 18/18 +5…+18pp | plan.md Phase 0.6; bench_lmsys/win_z{11,15}_seed*.json; paired.py |
| cost-isolation regime-dependent | plan.md Phase 0.6 §Table 2; paired.py CA−FREQ |
| decay is the drift lever | plan.md Phase 3; decay_vc{0,30,120}_seed*.json |
| eviction plumbing / clean_size | research.md §1.2; memory_cache.py; ca_w_tinylfu.py |
| storage 5.7×/7.5×/9.8×, precision tradeoff | bench_real_100k/results.json; bench_real_100k/frontier.md |
| FP gap is threshold artifact; matched-precision TP ≥ baseline | bench_real_100k/results.json (threshold_sweep); frontier.md §sweep |
| FP rise = encoder swap (+14.4pp), not truncation | bench_real_100k/results.json (cell I); frontier.md §isolation |
| e2e latency rises (encoder-bound); search ~100× | bench_real_100k/results.json e2e_latency_ms |
| cost_priority dial: paired cp1−cp0 cost_wt +2.5pp (7/7) ↔ hit% −4.1pp (0/7); all cp beat LFU cost_wt +6.3…8.8pp (6–7/7) | bench_cost_priority/cp_seed{0..6}.json |
