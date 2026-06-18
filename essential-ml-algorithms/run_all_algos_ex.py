from pathlib import Path
import subprocess
import sys
import time


SCRIPTS = [
    "01-linear-regression.py",
    "02-logistic-regression.py",
    "03-decision-trees-random-forest.py",
    "04-xgboost.py",
    "05-k-means-clustering.py",
    "06-support-vector-machine.py",
    "07-neural-networks.py",
]


def run_script(script_path: Path, project_root: Path) -> int:
    print(f"\n=== Running {script_path.name} ===")
    start = time.perf_counter()

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )

    elapsed = time.perf_counter() - start

    if result.stdout:
        print(result.stdout.rstrip())

    if result.stderr:
        print("\n[stderr]")
        print(result.stderr.rstrip())

    print(f"=== Finished {script_path.name} in {elapsed:.2f}s (exit {result.returncode}) ===")
    return result.returncode


def main() -> int:
    project_root = Path(__file__).resolve().parent
    failures = []

    for script_name in SCRIPTS:
        script_path = project_root / script_name

        if not script_path.exists():
            print(f"\n=== Skipping {script_name}: file not found ===")
            failures.append((script_name, "missing"))
            continue

        exit_code = run_script(script_path, project_root)
        if exit_code != 0:
            failures.append((script_name, exit_code))

    print("\n=== Run Summary ===")
    if not failures:
        print("All scripts completed successfully.")
        return 0

    print("Some scripts did not complete successfully:")
    for script_name, status in failures:
        print(f"- {script_name}: {status}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
