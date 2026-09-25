# Research cleanup manifest

**Date:** 2026-09-25

This cleanup removes heavyweight and generated state while preserving
canonical results, reports, source runners, tests, configs, and provenance.
No scientific training run was performed as part of cleanup.

## Size summary

| Measure | Before | After | Change |
|---|---:|---:|---:|
| Repository files (excluding `.git`) | 1,348,387,131 bytes | 488,093,504 bytes | -860,293,627 bytes |
| `research/` files | 461,037,615 bytes | 124,221,848 bytes | -336,815,767 bytes |

The final repository total includes retained external dataset/model assets in
`artifacts/`; those are not part of the research source snapshot and remain
ignored by Git.

## Removed artifacts

| Category | Count | Bytes | Notes |
|---|---:|---:|---|
| Research checkpoints (`.pt/.pth/.ckpt/.bin`) | 670 | 333,920,370 | Detailed per-file SHA256 provenance is in `DELETED_CHECKPOINTS.md`. |
| Cache directories/files | 39 directories | 920,149 inventory bytes | Includes repository and research Python/tool caches; nested cache bytes overlap generated Kaggle bundle accounting. |
| Generated Kaggle output/result/source bundles | 560 files in directories + 2 scripts | 205,285,776 | Reproducible runners and canonical research results are retained. |
| Root archive snapshot `research2.zip` | 1 | 314,868,635 | Non-scientific repository archive; SHA256 was recorded before deletion. |
| Unreferenced figures | 35 | 3,284,489 | Figures directly referenced by retained reports were kept. |
| Retrieval/tool cache `.rag_db/` | 6 files | 2,524,958 | Rebuilt automatically when needed; not scientific data. |
| Raw logs/console dumps | 0 | 0 | No retained research logs met the deletion criteria. |
| Duplicate source snapshots under `research/` | 0 | 0 | No unique research source copy was removed. Generated Kaggle source bundles are counted above. |
| Empty checkpoint directories | 61 | 0 | Removed after checkpoint deletion. |

The authoritative physical reduction is the size table above; category
inventories can overlap when a generated directory contained a cache.

## Retained large files

The two initialization JSON files exceed 10 MB because they contain historical
experiment trajectories and are retained as canonical scientific evidence.
External files under `artifacts/` (MNIST/XOR datasets and an older model) are
ignored project assets needed by existing loaders and were not silently
deleted during a research-folder cleanup.

## Cleanup assertions

- No research `.pt`, `.pth`, `.ckpt`, or `.bin` files remain.
- No research or repository Python/tool caches remain.
- No nested research archives remain.
- A1's runner disables checkpoint saving by default; heavyweight checkpoints
  require an explicit opt-in flag.

## Remaining files larger than 1 MiB

These files were audited after cleanup. `KEEP` means the file is retained with
an explicit scientific or project purpose; ignored `artifacts/` files are
external datasets/model assets rather than research snapshot contents.

| Classification | File | Size |
|---|---|---:|
| KEEP: canonical historical JSON | `operator_results/initialization_i2_results.json` | 26,262,954 |
| KEEP: canonical historical JSON | `operator_results/initialization_i2f_results.json` | 21,704,564 |
| KEEP: ignored external dataset | `../artifacts/mnist_binary.pt` | 220,081,933 |
| KEEP: ignored external dataset | `../artifacts/xor_dataset.pt` | 38,401,925 |
| KEEP: ignored external dataset | `../artifacts/burn_export/dataset.npz` | 30,720,490 |
| KEEP: canonical historical JSON | `operator_results/stage_b2_results.json` | 9,473,456 |
| KEEP: canonical historical JSON | `operator_results/stage_b3r_v3_results.json` | 8,695,275 |
| KEEP: canonical historical JSON | `operator_results/stage_b3r_results.json` | 8,045,002 |
| KEEP: canonical historical JSON | `operator_results/stage_b3_kaggle_v6.json` | 7,994,492 |
| KEEP: canonical MNIST JSON | `operator_results/m2_balanced_mnist_loss_results.json` | 5,871,214 |
| KEEP: canonical historical JSON | `operator_results/i5_gate_regularization_results.json` | 3,688,659 |
| KEEP: canonical historical JSON | `operator_results/stage_b_results.json` | 3,485,262 |
| KEEP: canonical historical JSON | `operator_results/i3_endpoint_results.json` | 2,495,819 |
| KEEP: canonical MNIST JSON | `operator_results/m1_mnist_pilot_results.json` | 2,335,200 |
| KEEP: ignored external dataset | `../artifacts/xor16_dataset.pt` | 19,201,941 |
| KEEP: ignored external dataset | `../artifacts/xor8_dataset.pt` | 9,601,933 |
| KEEP: ignored external dataset | `../artifacts/xor4_dataset.pt` | 4,801,933 |
| KEEP: project report | `../report/「note」.pdf` | 4,567,421 |
| KEEP: project report | `../report/and_gate_analysis.png` | 1,932,443 |
| KEEP: ignored external model asset | `../artifacts/mnist_transformer_checkpoint.pt` | 1,138,917 |
| KEEP: canonical historical JSON | `operator_results/operator_stress_results.json` | 1,780,249 |
| KEEP: canonical historical JSON | `operator_results/i13_loss_curriculum_results.json` | 1,406,009 |
| KEEP: canonical historical JSON | `operator_results/i6_or_consistency_results.json` | 1,393,136 |
| KEEP: canonical historical JSON | `operator_results/i11_loss_geometry_results.json` | 1,104,688 |
| KEEP: canonical historical JSON | `operator_results/initialization_i1r_full.json` | 1,079,659 |
