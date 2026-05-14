# RingX Attention — Routing Map

_A pictorial walkthrough of how a ring attention call flows through the codebase,
from the user-facing API down to the Triton kernel (or portable fallback)._

---

## 1. Top-Level Flow

```
ringX{1,2,3}_attn_func(local_q, local_k, local_v, ...)
         │
         │  N ring steps: 1 symmetric local call + (world_size−1) cross-rank calls
         │  Each step calls local_attn_forward / local_attn_backward
         │
         ▼
  local_attn_forward(q, k, v, ...)       ┐
  local_attn_backward(dout, q, k, v, ...) ┘  [ringX_attn/backend.py]
         │
         ▼
  resolve_backend()            ← RINGX_ATTN_BACKEND (default: "auto")
         │
         ├── "auto"
         │      flash_attn available? ──YES──▶  flash_attn  (CUDA only)
         │           │ NO
         │      fused available? ──────YES──▶  fused  (Triton, all platforms)
         │           │ NO                        │ unsupported call shape?
         │           │                           └──▶  portable  (silent fallback)
         │      portable  (always available)
         │
         ├── "fused"     ──────────────────────▶  fused   (error if unsupported)
         ├── "portable"  ──────────────────────▶  portable
         └── "flash_attn" ─────────────────────▶  flash_attn  (error if unavailable)
```

---

## 2. Forward Path

```
local_attn_forward(q, k, v, ...)
         │
         ├── flash_attn ──▶  _flash_forward()              [backend.py]
         │                   wraps _flash_attn_forward from the flash_attn package
         │                   returns: (out, lse)
         │
         ├── fused ────────▶  _fused_forward()             [backend.py]
         │                    permutes q/k/v to head-first layout
         │                    ▼
         │              _attention.forward()                [fused_attention.py]
         │                    ▼
         │              _attn_fwd[grid](...)                Triton kernel
         │                    • supports N_CTX_Q ≠ N_CTX_K  (symmetric + asymmetric)
         │                    • grid: (cdiv(N_CTX_Q, BLOCK_M), 1, BATCH*N_HEAD)
         │                    • autotune cache: attn_fwd_autotune_rank{0,1}.json
         │                    • returns: out, lse  (log-sum-exp, needed by ring merge)
         │
         └── portable ─────▶  _portable_forward()          [backend.py]
                              chunked softmax attention (Q-tile loop, pure PyTorch)
                              • no Triton required — runs on any platform
                              • supports all shapes: symmetric and asymmetric, any seqlen
                              • Q-tile size: RINGX_ATTN_PORTABLE_Q_TILE (explicit)
                                         or auto from RINGX_ATTN_PORTABLE_SCORE_CHUNK_MB
                                         (default: 256 MB score-buffer budget)
                              • returns: out, lse
```

### Forward autotune cache

`_attn_fwd` saves one JSON per rank because different ranks see different
`(N_CTX_Q, N_CTX_K)` shapes in asymmetric algorithms (e.g. ringX3 cross-rank calls).

| File | Written by | Purpose |
|------|-----------|---------|
| `attn_fwd_autotune_rank0.json` | rank 0 | shapes that rank 0 calls |
| `attn_fwd_autotune_rank1.json` | rank 1 | shapes that rank 1 calls — always covers all 3 ringX3 shapes for any world_size ≥ 2 |

At load time, both files are merged and the config with the lower measured time wins
for any shape present in both.

---

## 3. Backward Path

```
local_attn_backward(dout, q, k, v, out, lse, ...)
         │
         ├── flash_attn ──▶  _flash_backward()             [backend.py]
         │                   wraps _flash_attn_backward from the flash_attn package
         │
         ├── fused ────────▶  _fused_backward()            [backend.py]
         │                    permutes tensors to head-first layout
         │                    ▼
         │              _attention.backward()               [fused_attention.py]
         │                    │
         │                    ├── always runs first:
         │                    │   _attn_bwd_preprocess      Triton kernel
         │                    │   • computes Delta = sum(O * dO, dim=-1)
         │                    │   • grid: (cdiv(N_CTX_Q, 128), BATCH*N_HEAD)
         │                    │
         │                    └── kernel dispatch  (see Section 3a below)
         │
         └── portable ─────▶  _portable_backward()         [backend.py]
                              Q-tile loop with full attention recompute (pure PyTorch)
                              • no Triton required — runs on any platform
                              • supports all shapes and causal masks
```

### 3a. Fused backward kernel dispatch

```
_attention.backward()
│
├── is_asymmetric (N_CTX_Q ≠ N_CTX_K)?
│   │
│   ├── YES
│   │   ├── is_xpu?
│   │   │   ├── YES → _attn_bwd_two_phase_asym        [bwd_ta_configs, attn_bwd_ta_autotune.json]
│   │   │   │         • max-grid layout, scalar phase guards
│   │   │   │         • dQ accumulated in registers, written to HBM once
│   │   │   │         • always CAUSAL=False
│   │   │   │
│   │   │   └── NO  → _attn_bwd_single_pass            [bwd_sp_configs, attn_bwd_sp_autotune.json]
│   │   │             • evict_last honored on A100 — dQ tiles stay in L2
│   │   │             • load-add-store dQ; pre_hook zeros DQ buffer before each trial
│   │   │             • CAUSAL=False for all ringX3 cross-rank calls
│   │
│   └── NO  (symmetric, N_CTX_Q == N_CTX_K)
│       │
│       └── RINGX_ATTN_BWD_KERNEL= ?
│           │
│           ├── "two_phase"  (default)
│           │   └── _attn_bwd                          [bwd_configs, attn_bwd_autotune_{xpu,cuda}.json]
│           │         • two-phase structure: Phase 1 computes dK/dV, Phase 2 computes dQ
│           │         • dQ accumulated in registers (Phase 2) — immune to evict_last
│           │         • correct and performant on both XPU and CUDA
│           │
│           └── "single_pass"
│               └── _attn_bwd_single_pass              [bwd_sp_configs, attn_bwd_sp_autotune.json]
│                   ├── XPU  ⚠️  evict_last is a no-op → 1.84× SLOWER than two_phase — avoid
│                   └── CUDA ✅  +20.7% vs two_phase @ seqlen=8192 on A100
│
│
Preprocessing (always): _attn_bwd_preprocess
  • computes Delta = sum(O * dO, dim=-1)
  • grid: (cdiv(N_CTX_Q, 128), BATCH*N_HEAD)
```

---

## 4. Asymmetric Call Shapes (ringX3, world_size=4, seqlen=8192)

ringX3 uses **zigzag sharding**: each rank holds an early block and a late block from
opposite ends of the global sequence. This produces asymmetric local attention calls for
every cross-rank step.

| Ring step | N_CTX_Q | N_CTX_K | Kernel (XPU) | Kernel (CUDA) |
|-----------|---------|---------|--------------|---------------|
| Local (own rank) | 2048 | 2048 | `_attn_bwd` | `_attn_bwd` |
| i < rank: full Q, half KV | 2048 | 1024 | `_attn_bwd_two_phase_asym` | `_attn_bwd_single_pass` |
| i > rank: half Q, full KV | 1024 | 2048 | `_attn_bwd_two_phase_asym` | `_attn_bwd_single_pass` |

All cross-rank calls use `causal=False` — the zigzag slice choice guarantees every Q token
is globally later than every K token in each cross-rank call, so the causal mask is
all-pass and can be skipped.

---

## 5. Autotune Cache Files

All files are written to `$TRITON_AUTOTUNE_CACHE_DIR`.

**Load pattern (all cache files):** rank 0 reads the JSON file(s) from disk, deserializes
the configs, and broadcasts them to all other ranks via `dist.broadcast_object_list`.
This means only one rank pays the disk-read cost, and every rank ends up with identical
configs — no per-rank re-autotuning on warm runs.

| File | Kernel | Notes |
|------|--------|-------|
| `attn_fwd_autotune_rank0.json` | `_attn_fwd` | rank 0 shapes; merged at load |
| `attn_fwd_autotune_rank1.json` | `_attn_fwd` | rank 1 shapes; always covers all ringX3 shapes |
| `attn_bwd_autotune_xpu.json` | `_attn_bwd` (two_phase) | XPU winning configs |
| `attn_bwd_autotune_cuda.json` | `_attn_bwd` (two_phase) | CUDA winning configs |
| `attn_bwd_sp_autotune.json` | `_attn_bwd_single_pass` | both platforms |
| `attn_bwd_ta_autotune.json` | `_attn_bwd_two_phase_asym` | both platforms |

---

## 6. Env Var Quick Reference

| Env var | Values | Default | Effect |
|---------|--------|---------|--------|
| `RINGX_ATTN_BACKEND` | `auto`, `fused`, `portable`, `flash_attn` | `auto` | Backend selection |
| `RINGX_ATTN_BWD_KERNEL` | `two_phase`, `single_pass` | `two_phase` | Symmetric backward kernel (fused only) |
| `DEVICE_TYPE` | `cuda`, `xpu` | `cuda` | Device type and dist backend (`nccl` vs `xccl`) |
| `TRITON_AUTOTUNE_CACHE_DIR` | path | unset | Persistent autotune cache directory |
| `NUM_WARMUP` | integer | `1` | Warmup iterations before timed region (use `8` on XPU) |
| `RINGX_ATTN_PORTABLE_Q_TILE` | integer | auto | Portable backend Q-tile size (explicit override) |
| `RINGX_ATTN_PORTABLE_SCORE_CHUNK_MB` | integer | `256` | Portable backend score-buffer memory budget |

---

## 7. Key Rules

| Rule | Detail |
|------|--------|
| Asymmetric dispatch is automatic | Shape-driven (`N_CTX_Q ≠ N_CTX_K`), no env var needed |
| `RINGX_ATTN_BWD_KERNEL` only controls the symmetric path | Values: `two_phase` (default) or `single_pass` |
| Never use `single_pass` on XPU for symmetric calls | `evict_last` is a no-op on XPU → 1.84× regression vs `two_phase` |
| `two_phase_asym` is always `CAUSAL=False` | ringX3 mixes causal=True (local symmetric call → `_attn_bwd`) and causal=False (cross-rank asymmetric calls → `two_phase_asym`); zigzag geometry guarantees all Q tokens are globally later than all K tokens in every cross-rank call, making the causal mask redundant |
| Portable never falls back further | Always available; all shapes, all platforms, no Triton needed |
| In `auto` mode, unsupported fused calls fall back to portable silently | No error raised; check `forward_backend`/`backward_backend` fields in benchmark CSV output |
