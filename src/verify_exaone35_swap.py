"""
CLI to verify the EXAONE-3.5-7.8B swap (PRD §9.51/§9.52) end-to-end on real
GPU hardware. This sandbox has no GPU, so the swap itself could only be
syntax-checked (py_compile) when it was made -- this script is what
actually exercises it: runs each swapped file's own smoke test first
(cheapest check, catches a crash or a missing trust_remote_code patch
immediately), auto-applies patch_exaone35_remote_code.py once if the known
create_causal_mask bug appears, then re-runs the 4 validation scripts and
diffs their scalar summary numbers against the committed baseline JSONs
(generated with the pre-swap Qwen3-4B / EXAONE-4.0-1.2B models), so a
quality regression shows up as a number, not a guess.

Usage (from repo root, venv activated):
    python3 src/verify_exaone35_swap.py                  # full sequence
    python3 src/verify_exaone35_swap.py --skip-validate   # smoke tests only
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = Path(__file__).resolve().parent

# Order matters: cheapest/lowest-risk swapped file first (PRD §9.51),
# exaone_provider.py (highest blast radius, PRD §9.52) last.
# domain_vocab_substitution.py/curriculum_item_generator.py removed from
# this list (PRD §9.60 codebase cleanup -- the curriculum-upload feature
# they belonged to was never connected to the paper and was deleted).
_SMOKE_TEST_FILES = [
    "qwen_feedback.py",
    "solver_check.py",
    "exaone_provider.py",
]

_CAUSAL_MASK_MARKER = "create_causal_mask"

_VALIDATION_SCRIPTS = [
    ("validate_novelty1.py", "1_validation_results.json"),
    ("validate_item_discrimination.py", "item_discrimination_results.json"),
    ("validate_lm_plausibility.py", "lm_plausibility_results.json"),
    ("validate_log_likelihood.py", "log_likelihood_results.json"),
]


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=_REPO_ROOT, capture_output=True, text=True)


def _print_stderr_tail(stderr: str, n: int = 15) -> None:
    tail = "\n".join(stderr.strip().splitlines()[-n:])
    print("  --- stderr tail ---")
    print("  " + tail.replace("\n", "\n  "))


def run_smoke_tests() -> bool:
    print(f"=== Step 1/2: smoke tests ({len(_SMOKE_TEST_FILES)} files) ===\n")
    patched = False
    all_ok = True
    for filename in _SMOKE_TEST_FILES:
        path = _SRC / filename
        print(f"--- {filename} ---")
        result = _run([sys.executable, str(path)])
        if result.returncode != 0 and _CAUSAL_MASK_MARKER in result.stderr and not patched:
            print("  create_causal_mask error detected -- applying "
                  "patch_exaone35_remote_code.py and retrying once...")
            patch_result = _run([sys.executable, str(_SRC / "patch_exaone35_remote_code.py")])
            print("  " + patch_result.stdout.strip().replace("\n", "\n  "))
            patched = True
            result = _run([sys.executable, str(path)])
        if result.returncode == 0:
            print(f"  [OK] {filename}")
        else:
            all_ok = False
            print(f"  [FAIL] {filename} (exit {result.returncode})")
            _print_stderr_tail(result.stderr)
        print()
    return all_ok


def _find_committed_baseline(out_filename: str) -> Path | None:
    # The 4 existing artifacts are split between repo root and src/ (a
    # pre-existing inconsistency, not introduced by this script) -- check
    # both locations rather than assume one.
    for candidate in (_REPO_ROOT / out_filename, _SRC / out_filename):
        if candidate.exists():
            return candidate
    return None


def _diff_scalars(old: dict, new: dict) -> tuple[list[str], bool]:
    lines = []
    regressed = False
    for key in sorted(set(old) | set(new)):
        old_v, new_v = old.get(key), new.get(key)
        is_num = (isinstance(old_v, (int, float)) and not isinstance(old_v, bool)
                  and isinstance(new_v, (int, float)) and not isinstance(new_v, bool))
        if not is_num:
            continue
        delta = new_v - old_v
        flag = ""
        if delta < -1e-9:
            flag = "  <-- regression"
            regressed = True
        elif delta > 1e-9:
            flag = "  (improved)"
        lines.append(f"    {key}: {old_v:.4f} -> {new_v:.4f} (delta {delta:+.4f}){flag}")
    return lines, regressed


def run_validation_diff() -> bool:
    print("=== Step 2/2: validation scripts vs committed baseline ===\n")
    no_regression = True
    for script, out_filename in _VALIDATION_SCRIPTS:
        baseline_path = _find_committed_baseline(out_filename)
        new_out = _REPO_ROOT / f"_new_{out_filename}"
        print(f"--- {script} ---")
        result = _run([sys.executable, str(_SRC / script), "--out", str(new_out)])
        if result.returncode != 0:
            no_regression = False
            print(f"  [FAIL] script crashed (exit {result.returncode})")
            _print_stderr_tail(result.stderr)
            print()
            continue
        if baseline_path is None:
            print(f"  [WARN] no committed baseline found for {out_filename}, nothing to diff against")
            print()
            continue
        old = json.loads(baseline_path.read_text())
        new = json.loads(new_out.read_text())
        diff_lines, regressed = _diff_scalars(old, new)
        if diff_lines:
            print("\n".join(diff_lines))
        else:
            print("  no comparable scalar metrics found")
        if regressed:
            no_regression = False
        print()
    return no_regression


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skip-validate", action="store_true",
                         help="only run the 5 smoke tests, skip the 4 validation scripts")
    args = parser.parse_args()

    if not run_smoke_tests():
        print("SMOKE TESTS FAILED -- stopping before validation (fix the crash first).")
        return 1
    if args.skip_validate:
        print("Smoke tests passed. Validation skipped (--skip-validate).")
        return 0

    if run_validation_diff():
        print("ALL CHECKS PASSED -- no regression detected vs committed baselines.")
        return 0
    print("REGRESSION OR FAILURE DETECTED -- see details above. New result files kept as "
          "_new_*.json in the repo root for inspection; committed baselines were NOT overwritten.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
