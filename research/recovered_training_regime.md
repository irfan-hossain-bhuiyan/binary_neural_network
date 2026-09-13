# Recovered Discretizing Training Regime — Forensic Report

**Date:** 2026-09-13  
**Branch:** `autoresearch-kaggle` @ `1855818` (verified HEAD)  
**Goal:** Reconstruct historical *training procedure* (not architecture) that produced Boolean-corner-converged, discretizable networks.

---

## 1. Current Architecture Fingerprint (verified at `1855818`)

```
CURRENT ARCHITECTURE
--------------------
outer discrete gate: OR                         — DiscreteOrNorGateLayer.forward: z.any(dim=-1) (no inversion)
edge selector: learned w                        — continuous: z = xor(x,b)*w ; discrete: AND with w
literal polarity: learned b via XOR             — continuous: xor(a,b)=a+b-2ab (layers.py:12,131); discrete: x^bias
continuous XOR: a+b-2ab                         — confirmed layers.py
continuous aggregation: softmax(z*tau) weighting or hard max (use_softmax flag) — layers.py:140-149
residuals: none                                 — no skip/XOR residual in layers.py / models.py / discrete_logic_net.py
discrete conversion: threshold w,b → Boolean    — to_discrete(threshold=0.5) uses actual_weight/bias >= threshold
weight/bias mapping: leaky_clamp(raw,0,1,0.1)  — actual_weight / actual_bias
temperature: per-layer or shared, clamp 1e-3..10 via constraint; tau=1/temperature
```

No NOR, no output inversion — matches intended `u_ij = w AND (x XOR b)` ; `n_i = OR_j u_ij`. **No architecture change required.**

---

## 2. Software Archaeology Summary

Sources inspected (without checkout):
- `git log --all --oneline --graph --max-count=100`, `git show <commit>:<file>` for `regularizers.py`, `train_xor.py`, `run_xor_anneal.py`, `models.py`, `report/note.md`, `report/research.md`
- Current files: `report/note.md` (196 lines, experiments 1–22), `report/research.md`, `regularizers.py` (97 lines), `train_xor.py` (96 lines), `run_xor_anneal.py` (110 lines), `todo.md`, `idea.md`
- No archived checkpoints/logs with numeric discrete accuracies found; evidence is commit history + markdown + code.

Key historical timeline:
- `065f64d` ("Made big modification") introduced `regularizers.py` (both factories), `train_xor.py`, `run_xor_anneal.py` with discretizing training, and current OR architecture.
- `train_xor.py` is the *discretizing* regime (never overwritten by benchmark harness). `run_xor_anneal.py` started discretizing (with `odd=0 even=1` variance path) then drifted at `1e87bb8` to `odd=0.5 even=0.5` + variance-only (fuzzy) — this is exactly the benchmark's failing config.
- `report/note.md` experiments 1–12 converge on: alternating init `odd=0 even=1` is best, `reg=0.5` best, higher layers + grad_scalar help; experiments 13–19 test plateau/isolation variants.

---

## 3. Forensic Table

| property | old evidence | confidence |
|---|---|---|
| **architecture** | OR + w selector + b polarity via XOR, no residuals — confirmed `layers.py`/`discrete_logic_net.py` unchanged since `065f64d` | **confirmed** |
| **odd-layer initialization** (code `odd_initialization`, used when `i%2==1`) | `NormalInitWrapper(0.0)` in `train_xor.py:54` and `run_xor_anneal.py@065f64d:odd=0.0`; `report/note.md #12` "odd=0, even=1 biased is the way to good" | **confirmed** — layer 1,3,5… mean 0.0 |
| **even-layer initialization** (used when `i%2==0`) | `NormalInitWrapper(1.0)` same sources | **confirmed** — layer 0,2,4… mean 1.0 |
| **bias initialization** | `NormalInitWrapper(1.0)` in both `train_xor.py:56` and all `run_xor_anneal` versions; `report/note.md #14` "bias don't have much effect" but historically 1.0 | **confirmed** |
| **layer_dims (historical success)** | `train_xor.py:48` — `(64, 32, 16)` final dim = `num_bits` via `layer_dims=(64,32,16)` where last maps to bits; `run_xor_anneal@065f64d` — `(64,32,num_bits)` then `(64,32,64,num_bits)`. Report #3/#6 used `(256,128,64,128,64,32)` but those were later ablated. Most reproducible is `(64,32,32,num_bits)` family. | **likely** — `(64,32,32,num_bits)` is closest to current harness; original best had extra depth but not essential |
| **optimizer** | `Adam` in every historical `train_xor.py`/`run_xor_anneal.py` | **confirmed** |
| **learning rate** | `train_xor.py:66` — `0.01` (explicit); `run_xor_anneal@065f64d` — commented out (default Adam 1e-3), then `0.05` at `1e87bb8`. Report #19 notes `lr:0.1 betas:(0.5,0.5)` was an *experimental* deviation that underperformed without noise. | **confirmed** for discretizing regime: `0.01` |
| **loss** | `MSELoss` in all historical files; report #5 tried Huber but MSE with proper reg won. | **confirmed** |
| **regularizer — type** | `regularization_factory2(disc_lambda=0.5, tau_lambda=0.3, isolate_on_plateau=True)` in `train_xor.py:67-69` is the last *discretizing* version. Earlier `regularization_factory` added `l1_lambda` (w.relu) + full toggle off. `variance_regularizer` in `run_xor_anneal` is the *fuzzy* benchmark variant (no corner driving). | **confirmed** — `regularization_factory2` |
| **disc_lambda** | `0.5` in `train_xor.py`; report #8–#11 repeatedly "reg=0.5 works the best" | **confirmed** |
| **tau_lambda** | `0.3` in `train_xor.py` | **confirmed** |
| **l1_lambda** | Only in `regularization_factory` (1e-1); not used in `factory2` (successful path has no L1) | **confirmed** (not used in recovered regime) |
| **variance regularization** | Only in `run_xor_anneal` (1e-3 * batch_variance_cost); not in successful discretizing runs | **confirmed** (not part of recovered regime) |
| **plateau behavior** | `PlateauTracker(patience=15, min_err=0.01)` toggling between `_/\` (middle penalty) and `_/` (clamp 0..0.5 pull) in `factory2`; plus `call_fn_on_plateau(noise 0.3, patience 15)` as constraint in `train_xor.py:73-77`. Report #13 notes high RNG dependence but #17-19 show isolation variant still works. | **confirmed** — `isolate_on_plateau=True`, `patience=15` |
| **temperature strategy** | `train_xor.py:49-51` — `shared_temperature=True, learnable_tau=True, init_temperature=1.0` with tau encouraged via `exp(-tau)` regularizer (no annealing). `run_xor_anneal` used per-layer `linear_temperature_anneal 1.0→0.01`. Report tau logs show values ~7–20 (tau, not temperature) growing when regularized. | **confirmed** — fixed init shared learnable, regularized growth (not annealed) |
| **shared vs per-layer temp** | `shared_temperature=True` in successful `train_xor.py`; benchmark uses per-layer. | **confirmed** — shared |
| **epochs** | `train_xor.py:19` default 40, `main()` calls `300, check_grad=True`; report #18 mentions 300-epoch test with recovered regime. `run_xor_anneal` used 40. Successful runs used 100–300. | **likely** — 100 is faithful to original default, 300 is documented best |
| **batch size** | `256` in all historical files | **confirmed** |
| **dataset / task** | `train_xor.py` and `run_xor_anneal` both use **bitwise XOR**: `save_xor_dataset(num_samples=100000, num_bits=16)` (or 32), `split train_ratio=0.8 shuffle=True`, input `2*num_bits`, output `num_bits`. Not 2-bit gate task except report #22. | **confirmed** — bitwise XOR, 100k samples, 16-bit is canonical (32-bit also used) |
| **discrete threshold** | `0.5` everywhere (`discretize`/`to_discrete`) | **confirmed** |
| **known continuous accuracy** | No numeric logs preserved; report images suggest 80–90% region but no table. Must not fabricate. | **unknown** |
| **known discrete accuracy** | Same — no numeric discrete accuracy preserved in repo. Report describes "really bad 0.4999" for constant init (failure) and qualitative "good behaviour" for alternating init, but no hard numbers. | **unknown** (qualitative only) |

**Indexing clarification (critical):** `models.py:49` — `even_initialization if i%2==0 else odd_initialization`. So "even layers" = Python indices `0,2,4…`, "odd layers" = `1,3,5…`. Historical "odd=0 even=1" therefore means `layer0≈1, layer1≈0, layer2≈1…` — **not** the reverse. Benchmark's `odd=0.5 even=0.5` collapses this alternation (fuzzy).

---

## 4. What the Previous Benchmark Got Wrong

Benchmark `research/configs/baseline_suite.json` used: `odd=0.5 even=0.5 bias=1.0`, `variance_weight=1e-3`, `temp anneal 1→0.01`, `20→100 epochs`, no discretizing regularizer. This is precisely the `run_xor_anneal@1e87bb8` fuzzy variant identified in `report/note.md` experiment 12 as suboptimal ("odd_even=(0.5,0.5) ... 0.4999 / weight distribution not right"). No pathway to `D_w≈0` exists under that config → thresholding is invalid → 0/55 recovery is not architecture evidence.

---

## 5. Recommendation for Reproduction

Reproduce `train_xor.py@065f64d` faithfully on current architecture (already compatible — same OR gate): bitwise XOR 16-bit (100k, 80/20 split), `(64,32,16→32?)` adapted to `hidden=[64,32]` + output, alternating `even=1 odd=0 bias=1`, `Adam lr0.01`, `MSELoss`, `regularization_factory2(0.5,0.3, isolate=True)`, shared learnable tau init 1.0, noise on plateau 0.3, 100–300 epochs, seed 0. This will be codified as `research/configs/recovered_discretizing_baseline.json` next.
