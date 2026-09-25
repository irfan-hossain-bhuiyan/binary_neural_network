# Experiment index

This is a navigation index. Numerical claims remain in each experiment's
canonical JSON and report. Checkpoint binaries are intentionally not retained
in this source snapshot; see `DELETED_CHECKPOINTS.md`.

| ID | Task | Main intervention | Seeds | Continuous result | Boolean result | Status | Results | Report |
|---|---|---|---:|---|---|---|---|---|
| I1 | XOR truth table | Early operator/architecture baseline | documented | see report | see report | complete | operator results | `initialization_research.md` |
| I2 | XOR truth table | I2-B mean-field initialization | 5 | all verified continuous solutions | basin-dependent | complete | `operator_results/initialization_i2_results.json` | `initialization_research.md` |
| I3 | XOR truth table | Endpoint continuation, MSE vs BCE | 3 | continuous exact | BCE can recover discrete topology | complete | `operator_results/i3_endpoint_results.json` | `i3_endpoint_report.md` |
| I4 | XOR truth table | Mean vs top-k worst-case losses | 2–3 | continuous endpoint improvement | seed3 remained a two-row miss | complete | `operator_results/i4_worstcase_results.json` | `i4_worstcase_report.md` |
| I5 | XOR truth table | Gate polarization and finite margin | 1 | endpoint sharpening | wrong topology hardened | complete | `operator_results/i5_gate_regularization_results.json` | `i5_gate_regularization_report.md` |
| I6 | XOR truth table | Lehmer/hard-OR mismatch forensic analysis | 1 | local mismatch identified | one-bit Boolean repair found | complete | `operator_results/i6_or_consistency_results.json` | `i6_or_consistency_report.md` |
| I7 | XOR truth table | Lehmer-to-max operator homotopy | 1 | semantic sensitivity measured | continuation insufficient | complete | `operator_results/i7_operator_homotopy_results.json` | `i7_operator_homotopy_report.md` |
| I8 | XOR truth table | Exact Boolean-forward sigmoid STE | 1 | preserved continuous solution | repaired seed3 basin | complete | `operator_results/i8_boolean_ste_results.json` | `i8_boolean_ste_report.md` |
| I9 | XOR truth table | Causal-agnostic global STE calibration | 5 | continuous exact | limited replication | complete | `operator_results/i9_ste_replication_results.json` | `i9_ste_replication_report.md` |
| I10 | XOR truth table | Layer-balanced STE composition | seed3 then replication | continuous exact | tested generic layer pressure | complete | `operator_results/i10_layer_balanced_ste_results.json` | `i10_layer_balanced_ste_report.md` |
| I11 | XOR with contradictory labels | Loss geometry: MSE/BCE/power/MAE | seed3 | power losses endpoint-seeking | noise tradeoff | complete | `operator_results/i11_loss_geometry_results.json` | `i11_loss_geometry_report.md` |
| I12 | Clean XOR | Five-seed fixed-loss replication | 5 | all continuous exact | power-loss Boolean recovery was basin-sensitive | complete | `operator_results/i12_loss_seed_replication_results.json` | `i12_loss_seed_replication_report.md` |
| I13 | Clean XOR | POWER 1.5 → 1.25 curriculum | 5 | all continuous exact | no seed-robustness improvement | complete | `operator_results/i13_loss_curriculum_results.json` | `i13_loss_curriculum_report.md` |
| M1 | Binary-input MNIST | Scale/discretization pilot | 3 | continuous classifier learned partially | large continuous→Boolean gap | complete | `operator_results/m1_mnist_pilot_results.json` | `m1_mnist_pilot_report.md` |
| M2 | Binary-input MNIST | Balanced BCE and balanced POWER 1.25 | 3 | longer training and balanced outputs | gap remained measurable | complete | `operator_results/m2_balanced_mnist_loss_results.json` | `m2_balanced_mnist_loss_report.md` |
| A1 | 4-bit unsigned addition | New deterministic compositional benchmark | 5 × 4 losses | exact continuous recovery: 0/20; MSE strongest in budget | exact Boolean recovery: 0/20 | complete; no exact recovery in 3000 steps | `operator_results/a1_binary_addition_results.json` | `a1_binary_addition_report.md` |
