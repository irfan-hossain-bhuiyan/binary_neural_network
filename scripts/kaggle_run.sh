#!/usr/bin/env bash
# End-to-end reproducible Kaggle experiment run.
#
#   verify Git tree clean
#     -> get commit hash
#     -> generate Kaggle script from HEAD
#     -> submit
#     -> poll status
#     -> download output
#     -> verify commit hash
#     -> archive local result
#     -> print summary
#
# NOTE on kaggle/train.py: it is a *generated* artifact (produced by
# scripts/prepare_kaggle.py from the committed HEAD). Modifications to
# that single file do not count as a dirty tree; every other tracked
# modification blocks the run.
set -euo pipefail

KERNEL_ID="irfanhossainbhuiyan/binary-neural-network-research"
GENERATED="kaggle/train.py"

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

COMMIT="$(git rev-parse HEAD)"
SHORT_COMMIT="$(git rev-parse --short HEAD)"
echo "Experiment commit: $COMMIT ($SHORT_COMMIT)"

# ---- refuse dirty experiments (except the generated bootstrap) ----
DIRTY="$(git status --porcelain | grep -v "kaggle/train.py" || true)"
# Untracked files (??) are allowed: ignored outputs (kaggle/output,
# kaggle/results, artifacts) never appear here anyway.
DIRTY_TRACKED="$(echo "$DIRTY" | grep -v '^??' | grep -v '^$' || true)"
if [ -n "$DIRTY_TRACKED" ]; then
    echo "Working tree is dirty."
    echo "Commit the experiment before submitting it."
    echo "$DIRTY_TRACKED"
    exit 1
fi

# ---- generate the Kaggle script from HEAD ----
python scripts/prepare_kaggle.py

test -f "$GENERATED"
python -m py_compile "$GENERATED"
echo "Generated $GENERATED from $SHORT_COMMIT"

# ---- submit to Kaggle ----
kaggle kernels push \
    -p kaggle \
    --accelerator NvidiaTeslaT4

# ---- poll status (max ~60 min) ----
for _ in $(seq 1 120); do
    STATUS="$(kaggle kernels status "$KERNEL_ID")"
    echo "$STATUS"

    if echo "$STATUS" | grep -qi "complete"; then
        break
    fi

    if echo "$STATUS" | grep -qi "error"; then
        echo "Kaggle run failed."
        exit 1
    fi

    sleep 30
done

if ! echo "$STATUS" | grep -qi "complete"; then
    echo "Timed out waiting for Kaggle kernel to complete."
    exit 1
fi

# ---- download result ----
mkdir -p kaggle/output
kaggle kernels output \
    "$KERNEL_ID" \
    -p kaggle/output \
    -o

if [ ! -f kaggle/output/result.json ]; then
    echo "ERROR: kaggle/output/result.json missing after download."
    ls -la kaggle/output
    exit 1
fi

# ---- verify commit identity (mandatory) ----
RESULT_COMMIT="$(python3 -c "import json; print(json.load(open('kaggle/output/result.json')).get('git_commit', ''))")"
if [ "$RESULT_COMMIT" != "$COMMIT" ]; then
    echo "ERROR: result commit mismatch."
    echo "  expected: $COMMIT"
    echo "  got:      $RESULT_COMMIT"
    exit 1
fi
echo "Commit verification passed: $RESULT_COMMIT"

# ---- archive local result by commit ----
mkdir -p kaggle/results
EXP_NAME="$(python3 -c "import json; print(json.load(open('kaggle/output/result.json')).get('metrics', {}).get('experiment_name', 'experiment'))")"
SEED="$(python3 -c "import json; print(json.load(open('kaggle/output/result.json')).get('metrics', {}).get('seed', 'noseed'))")"
ARCHIVED="kaggle/results/${SHORT_COMMIT}_${EXP_NAME}_seed${SEED}.json"
cp kaggle/output/result.json "$ARCHIVED"
echo "Archived result to $ARCHIVED"

# ---- print summary ----
python3 -c "
import json
r = json.load(open('kaggle/output/result.json'))
m = r.get('metrics', {})
print('status               :', r.get('status'))
print('experiment           :', m.get('experiment_name'))
print('seed                 :', m.get('seed'))
print('continuous_accuracy  :', m.get('continuous_accuracy'))
print('continuous_exact_acc :', m.get('continuous_exact_accuracy'))
print('discrete_accuracy    :', m.get('discrete_accuracy'))
print('cont-disc gap        :', m.get('continuous_discrete_gap'))
print('final_loss           :', m.get('final_loss'))
print('runtime_seconds      :', m.get('runtime_seconds'))
print('gpu                  :', m.get('gpu_name'))
"
