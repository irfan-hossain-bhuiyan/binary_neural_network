#!/usr/bin/env bash
# End-to-end reproducible Kaggle experiment run.
#
#   verify Git tree clean
#     -> get commit hash
#     -> generate Kaggle script from HEAD (single or suite mode)
#     -> submit
#     -> poll status
#     -> download output
#     -> verify commit hash
#     -> archive local result
#     -> append to local ledger (kaggle/results/experiments.jsonl)
#     -> print summary
#
# Modes (via environment):
#   KAGGLE_MODE=single  one baseline run (default, preserves original behavior)
#   KAGGLE_MODE=suite   task x seed suite in one Kaggle job
#   SUITE=baseline_suite
#   SEEDS=0             comma-separated (suite mode)
#
# NOTE on kaggle/train.py: it is a *generated, git-ignored* artifact
# (produced by scripts/prepare_kaggle.py from the committed HEAD).
set -euo pipefail

KERNEL_ID="irfanhossainbhuiyan/binary-neural-network-research"
GENERATED="kaggle/train.py"
KAGGLE_MODE="${KAGGLE_MODE:-single}"
SUITE="${SUITE:-baseline_suite}"
SEEDS="${SEEDS:-0}"
CONFIG="${CONFIG:-research/configs/baseline.json}"

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

COMMIT="$(git rev-parse HEAD)"
SHORT_COMMIT="$(git rev-parse --short HEAD)"
echo "Experiment commit: $COMMIT ($SHORT_COMMIT)"
echo "Mode: $KAGGLE_MODE suite=$SUITE seeds=$SEEDS"

# ---- refuse dirty experiments ----
# kaggle/train.py is generated+ignored so it never appears here.
DIRTY_TRACKED="$(git status --porcelain | grep -v '^??' | grep -v '^$' || true)"
if [ -n "$DIRTY_TRACKED" ]; then
    echo "Working tree is dirty."
    echo "Commit the experiment before submitting it."
    echo "$DIRTY_TRACKED"
    exit 1
fi

# ---- generate the Kaggle script from HEAD ----
if [ "$KAGGLE_MODE" = "suite" ]; then
    python scripts/prepare_kaggle.py --mode suite --suite "$SUITE" --seeds "$SEEDS"
else
    python scripts/prepare_kaggle.py --mode single --config "$CONFIG"
fi

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

# ---- archive local result by commit (+ append to ledger) ----
mkdir -p kaggle/results
export SHORT_COMMIT KAGGLE_MODE SUITE SEEDS
ARCHIVED="$(python3 - <<'EOF'
import json, os
r = json.load(open('kaggle/output/result.json'))
m = r.get('metrics', {})
short = os.environ['SHORT_COMMIT']
mode = os.environ['KAGGLE_MODE']
if mode == 'suite' and isinstance(m.get('runs'), list):
    suite = m.get('suite_name', os.environ['SUITE'])
    seeds = os.environ['SEEDS'].replace(',', '-')
    dest = f"kaggle/results/{short}_suite_{suite}_seeds{seeds}.json"
else:
    exp = m.get('experiment_name', m.get('task_name', 'experiment'))
    seed = m.get('seed', 'noseed')
    dest = f"kaggle/results/{short}_{exp}_seed{seed}.json"
print(dest)
EOF
)"
cp kaggle/output/result.json "$ARCHIVED"
echo "Archived result to $ARCHIVED"

python3 - "$ARCHIVED" <<'EOF'
import json, sys
# Append one ledger line per run; skip experiment_ids already present.
archived = sys.argv[1]
r = json.load(open('kaggle/output/result.json'))
m = r.get('metrics', {})
ledger_path = 'kaggle/results/experiments.jsonl'
seen = set()
try:
    with open(ledger_path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    seen.add(json.loads(line).get('experiment_id'))
                except json.JSONDecodeError:
                    pass
except FileNotFoundError:
    pass
entries = []
if isinstance(m.get('runs'), list):
    for run in m['runs']:
        entries.append({
            'experiment_id': run.get('experiment_id'),
            'git_commit': r.get('git_commit'),
            'task_name': run.get('task_name'),
            'seed': run.get('seed'),
            'result_file': archived,
            'status': run.get('status'),
        })
else:
    entries.append({
        'experiment_id': m.get('experiment_id'),
        'git_commit': r.get('git_commit'),
        'task_name': m.get('task_name', m.get('experiment_name')),
        'seed': m.get('seed'),
        'result_file': archived,
        'status': r.get('status'),
    })
added = 0
with open(ledger_path, 'a') as f:
    for e in entries:
        if e['experiment_id'] and e['experiment_id'] in seen:
            continue
        f.write(json.dumps(e) + '\n')
        seen.add(e['experiment_id'])
        added += 1
print(f"Ledger: +{added} entries ({ledger_path})")
EOF

# ---- print summary ----
if [ "$KAGGLE_MODE" = "suite" ]; then
    python3 -c "
import json
r = json.load(open('kaggle/output/result.json'))
m = r.get('metrics', {})
print('suite:', m.get('suite_name'), '| runs:', len(m.get('runs', [])))
for task, s in m.get('summary', {}).items():
    d = (s.get('discrete_exact_accuracy') or {})
    rec = s.get('discrete_exact_recovery_count', '?')
    print(f'{task:18s} ok={s.get(\"num_success\")}/{s.get(\"num_runs\")} '
          f'recovery={rec} disc_exact_mean={d.get(\"mean\")} gap_mean={(s.get(\"continuous_discrete_gap\") or {}).get(\"mean\")}')
"
else
    python3 -c "
import json
r = json.load(open('kaggle/output/result.json'))
m = r.get('metrics', {})
print('status               :', r.get('status'))
print('experiment           :', m.get('experiment_name', m.get('task_name')))
print('seed                 :', m.get('seed'))
print('continuous_accuracy  :', m.get('continuous_accuracy'))
print('continuous_exact_acc :', m.get('continuous_exact_accuracy'))
print('discrete_accuracy    :', m.get('discrete_accuracy'))
print('cont-disc gap        :', m.get('continuous_discrete_gap'))
print('final_loss           :', m.get('final_loss'))
print('runtime_seconds      :', m.get('runtime_seconds'))
print('gpu                  :', m.get('gpu_name'))
"
fi
