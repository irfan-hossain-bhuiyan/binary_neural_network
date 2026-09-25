# Research result storage policy

New experiment JSON files use a compact, text-first format:

- static configuration and provenance appear once at the top level;
- ordinary trajectory points retain only task-independent metrics, timing,
  tolerance fractions, and continuous/Boolean disagreement;
- full carry-chain, per-bit, gradient, and topology diagnostics are retained
  only at named milestone steps;
- topology transitions are stored as small event records;
- model and optimizer state are never embedded in canonical JSON;
- research runners do not save `.pt`, `.pth`, or `.ckpt` files by default.

Historical result JSON files are preserved unchanged. They may be compacted in
a later provenance-preserving maintenance task by rerunning their existing
postprocessors and recording the original SHA256.
