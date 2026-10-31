"""Reproduce all experiments from a fresh checkout with one command."""

import argparse
from datetime import datetime
import hashlib
from pathlib import Path
import shutil
import subprocess
import sys

from src.common import ROOT, spark_path, write_json
from src.runtime import environment, prepare_python, validate_prerequisites

STEPS = [
    ("Prepare the fixed dataset", "download_data"),
    ("Train baseline experiments", "run_project"),
    ("Reload and verify the baseline model", "verify_model"),
    ("Independently verify baseline results", "check_results"),
    ("Train 23 extended candidates", "run_extended"),
    ("Reload and independently verify nine methods", "verify_extended"),
    ("Draw baseline figures", "plot_results"),
    ("Draw validation-selected model figures", "plot_extended"),
    ("Review the recommendation and recompute its errors", "review_selection"),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hdfs", help="Optional existing HDFS root, e.g. hdfs://host:9000/course/bike"
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Use previously cached data, dependencies and Windows runtime.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "results",
        help="Parent folder for a new, timestamped run.",
    )
    args = parser.parse_args()
    validate_prerequisites()
    if args.hdfs and (not args.hdfs.startswith("hdfs://") or not shutil.which("hdfs")):
        raise RuntimeError(
            "--hdfs requires an hdfs:// URI and the Hadoop command on PATH."
        )
    python = prepare_python(offline=args.offline)
    env = environment(python, offline=args.offline)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    result = args.output_root.resolve() / stamp
    logs = result / "logs"
    logs.mkdir(parents=True)
    baseline, extended, review = [
        result / name for name in ("baseline", "extended", "review")
    ]
    storage = (
        args.hdfs.rstrip("/") + "/reproduced/" + stamp
        if args.hdfs
        else spark_path(result / "storage")
    )
    source = str(ROOT / "data/day.csv")
    arguments = [
        ["--offline"] if args.offline else [],
        ["--input", source, "--storage", storage, "--output", str(baseline)],
        ["--output", str(baseline)],
        ["--output", str(baseline)],
        ["--storage", storage, "--output", str(extended)],
        ["--storage", storage, "--output", str(extended)],
        ["--output", str(baseline)],
        ["--output", str(extended)],
        [
            "--model-root",
            storage + "/extended_models",
            "--experiments",
            str(extended),
            "--output",
            str(review),
        ],
    ]
    manifest = dict(
        status="running",
        started=datetime.now().astimezone().isoformat(),
        config_sha256=hashlib.sha256(
            (ROOT / "extended_config.json").read_bytes()
        ).hexdigest(),
        steps=[],
    )
    write_json(result / "manifest.json", manifest)
    for index, ((label, module), extra) in enumerate(zip(STEPS, arguments), 1):
        print(f"[{index}/{len(STEPS)}] {label}", flush=True)
        log = logs / f"{index:02d}-{module}.log"
        command = [python, "-u", "-m", "src." + module, *extra]
        with log.open("w", encoding="utf-8") as stream:
            process = subprocess.run(
                command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT
            )
        manifest["steps"].append(
            dict(
                module=module,
                returncode=process.returncode,
                log=str(log.relative_to(result)),
            )
        )
        if process.returncode:
            manifest["status"] = "failed"
            write_json(result / "manifest.json", manifest)
            print(
                log.read_text(encoding="utf-8", errors="replace")[-6500:],
                file=sys.stderr,
            )
            raise RuntimeError(f"{label} failed. See {log}")
        if index == 1 and args.hdfs:
            subprocess.run(
                ["hdfs", "dfs", "-mkdir", "-p", storage + "/raw"], check=True, env=env
            )
            subprocess.run(
                ["hdfs", "dfs", "-put", "-f", source, storage + "/raw/day.csv"],
                check=True,
                env=env,
            )
            arguments[1][1] = storage + "/raw/day.csv"
    from src.reporting import write_run_summary

    write_run_summary(result)
    manifest.update(status="complete", finished=datetime.now().astimezone().isoformat())
    write_json(result / "manifest.json", manifest)
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "latest.txt").write_text(str(result) + "\n", encoding="utf-8")
    print(f"\nCompleted: {result / 'summary.md'}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as error:
        print(f"Reproduction stopped: {error}", file=sys.stderr)
        sys.exit(1)
