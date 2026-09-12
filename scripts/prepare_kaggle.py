"""Generate kaggle/train.py from the exact committed Git state.

Flow:
  1. Refuse if tracked files are modified/staged but uncommitted
     (kaggle/train.py itself is exempt: it is the generated artifact).
  2. Refuse if untracked *.py source files exist (they would be silently
     excluded from the archive since packaging reads from HEAD).
  3. Collect relevant tracked source files, read their contents from HEAD
     (never from the possibly-modified working tree).
  4. Build an in-memory tar.gz, base64-encode it, embed it in
     kaggle/train.py together with the full commit hash.

The generated kaggle/train.py extracts the archive to
/kaggle/working/repo on the Kaggle worker, runs
research/run_experiment.py, and always writes
/kaggle/working/result.json with the git commit hash.
"""

from __future__ import annotations

import base64
import io
import subprocess
import sys
import tarfile
from pathlib import Path

# Rough guardrail: refuse to upload unexpectedly huge payloads.
MAX_COMPRESSED_BYTES = 20 * 1024 * 1024

# Directories that must never be packaged (generated output, caches,
# checkpoints, reports, datasets, vendored indexes).
EXCLUDE_DIR_PREFIXES = (
    "artifacts/",
    "kaggle/",
    ".rag_db/",
    "report/",
    "__pycache__/",
    ".ipynb_checkpoints/",
    ".pytest_cache/",
    ".git/",
)

# Binary / generated / irrelevant file types.
EXCLUDE_SUFFIXES = (
    ".pt", ".pth", ".ckpt",
    ".png", ".jpg", ".jpeg", ".gif", ".svg",
    ".pdf", ".mp4", ".avi", ".mov",
    ".sqlite3", ".db",
    ".pyc", ".pyo",
    ".ipynb",
    ".log",
)

# Source/config file types to include.
INCLUDE_SUFFIXES = (
    ".py", ".toml", ".yaml", ".yml", ".json", ".txt", ".md", ".cfg",
)

# The generated bootstrap itself is exempt from the clean-tree check.
GENERATED_BOOTSTRAP = "kaggle/train.py"


def run_git(args: list[str], cwd: Path) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=str(cwd), text=True
    ).strip()


def ensure_clean_tree(root: Path) -> None:
    """Refuse to package unless tracked source state is committed.

    Ignored files (kaggle/output, results, artifacts, ...) do not appear
    in `git status --porcelain` and are therefore allowed. Untracked files
    (`??`) are allowed here but *.py sources are checked separately.
    """
    out = run_git(["status", "--porcelain"], root)
    dirty = [
        line
        for line in out.splitlines()
        if line.strip()
        and not line.startswith("??")
        and GENERATED_BOOTSTRAP not in line
    ]
    if dirty:
        print("ERROR: Commit the experiment before running Kaggle.", file=sys.stderr)
        print("Tracked files with uncommitted changes:", file=sys.stderr)
        for line in dirty:
            print(f"  {line}", file=sys.stderr)
        sys.exit(1)

    # Untracked *.py files would be silently excluded (packaging reads
    # from HEAD), so refuse rather than produce a misleading archive.
    untracked_py = [
        line[3:].strip()
        for line in out.splitlines()
        if line.startswith("??")
        and line[3:].strip().endswith(".py")
        and not any(
            line[3:].strip().startswith(p) for p in EXCLUDE_DIR_PREFIXES
        )
    ]
    if untracked_py:
        print(
            "ERROR: Untracked Python files would be excluded from the "
            "HEAD-based package. git add + commit them first:",
            file=sys.stderr,
        )
        for path in untracked_py:
            print(f"  {path}", file=sys.stderr)
        sys.exit(1)


def select_files(root: Path) -> list[str]:
    out = run_git(["ls-files", "-z"], root)
    selected: list[str] = []
    for path in out.split("\0"):
        if not path:
            continue
        if path.startswith(EXCLUDE_DIR_PREFIXES):
            continue
        if path == "temp":  # stray local file, not source
            continue
        suffix = Path(path).suffix.lower()
        if suffix in EXCLUDE_SUFFIXES:
            continue
        if suffix not in INCLUDE_SUFFIXES:
            continue
        selected.append(path)
    return sorted(selected)


def build_archive(root: Path, files: list[str]) -> tuple[bytes, int]:
    """Read each file from HEAD, pack into gzip tar. Returns (data, raw_size)."""
    raw_size = 0
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for path in files:
            content = subprocess.check_output(
                ["git", "show", f"HEAD:{path}"], cwd=str(root)
            )
            raw_size += len(content)
            info = tarfile.TarInfo(name=path)
            info.size = len(content)
            info.mtime = 0  # reproducible archive
            tar.addfile(info, io.BytesIO(content))
    return buf.getvalue(), raw_size


BOOTSTRAP_TEMPLATE = '''"""Generated Kaggle bootstrap. DO NOT EDIT BY HAND.

Source of truth: Git commit __COMMIT__ (see COMMIT below).
Regenerate with: python scripts/prepare_kaggle.py
"""

import base64
import io
import json
import subprocess
import sys
import tarfile
import traceback
from pathlib import Path

COMMIT = "__COMMIT__"

ARCHIVE_B64 = """__ARCHIVE_B64__"""

REPO_DIR = Path("/kaggle/working/repo")
RESULT_PATH = Path("/kaggle/working/result.json")


def _json_from_stdout(stdout: str) -> dict:
    # The experiment prints exactly one JSON document; recover it
    # robustly even if other log lines (e.g. trainer progress) precede it.
    start = stdout.find("{")
    end = stdout.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON metrics found in experiment stdout")
    return json.loads(stdout[start:end + 1])


def main() -> None:
    REPO_DIR.mkdir(parents=True, exist_ok=True)
    raw = base64.b64decode(ARCHIVE_B64)
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tar:
        tar.extractall(path=str(REPO_DIR))

    proc = subprocess.run(
        [sys.executable, "research/run_experiment.py",
         "--experiment-name", "baseline", "--seed", "0"],
        cwd=str(REPO_DIR),
        capture_output=True,
        text=True,
    )
    print(proc.stdout)
    print(proc.stderr, file=sys.stderr)

    if proc.returncode != 0:
        raise RuntimeError(
            f"research/run_experiment.py failed (exit={proc.returncode}):\n"
            f"{proc.stderr[-4000:]}"
        )
    metrics_path = REPO_DIR / "metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text())
    else:
        metrics = _json_from_stdout(proc.stdout)

    result = {"status": "success", "git_commit": COMMIT, "metrics": metrics}
    RESULT_PATH.write_text(json.dumps(result, indent=2))
    print(f"Wrote {RESULT_PATH}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001 - must always write result.json
        try:
            RESULT_PATH.write_text(json.dumps({
                "status": "failed",
                "git_commit": COMMIT,
                "error": f"{exc}\n{traceback.format_exc()[-4000:]}",
            }, indent=2))
        except Exception:
            pass
        raise
'''


def main() -> None:
    root = Path(
        subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"], text=True
        ).strip()
    )
    ensure_clean_tree(root)

    commit = run_git(["rev-parse", "HEAD"], root)
    short_commit = run_git(["rev-parse", "--short", "HEAD"], root)

    files = select_files(root)
    if not files:
        print("ERROR: no source files selected for packaging.", file=sys.stderr)
        sys.exit(1)
    if "research/run_experiment.py" not in files:
        print(
            "ERROR: research/run_experiment.py is not tracked in HEAD. "
            "Commit it before packaging.",
            file=sys.stderr,
        )
        sys.exit(1)

    archive, raw_size = build_archive(root, files)
    if len(archive) > MAX_COMPRESSED_BYTES:
        print(
            f"ERROR: compressed archive is {len(archive)} bytes "
            f"(>{MAX_COMPRESSED_BYTES}). Refusing to upload.",
            file=sys.stderr,
        )
        sys.exit(1)

    b64 = base64.b64encode(archive).decode("ascii")
    script = BOOTSTRAP_TEMPLATE.replace("__COMMIT__", commit).replace(
        "__ARCHIVE_B64__", b64
    )
    out_path = root / GENERATED_BOOTSTRAP
    out_path.write_text(script)

    print(f"files packaged          : {len(files)}")
    print(f"uncompressed size       : {raw_size} bytes")
    print(f"compressed archive size : {len(archive)} bytes")
    print(f"generated train.py size : {len(script)} bytes")
    print(f"commit                  : {commit} ({short_commit})")
    for path in files:
        print(f"  {path}")


if __name__ == "__main__":
    main()
