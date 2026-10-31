"""Standard-library bootstrap and platform-specific Spark environment."""

import hashlib
import importlib.metadata
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.request
import venv

from .common import ROOT

NATIVE = {
    "winutils.exe": "a0ca6e358357c41ef56ebdb02c38e4a4d55da7ca7a13001678bb2ef7d644adea",
    "hadoop.dll": "d3dd64afdc85f2a7eb5345abf2ecaa744b0a157de40859313337d47f81ee1c7b",
}


def dependencies_ready():
    for line in (ROOT / "requirements.txt").read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        name, expected = line.split("==")
        try:
            if importlib.metadata.version(name) != expected:
                return False
        except importlib.metadata.PackageNotFoundError:
            return False
    return True


def prepare_python(offline=False):
    if dependencies_ready():
        return sys.executable
    folder = ROOT / ".venv"
    executable = folder / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    wanted = dict(
        line.split("==")
        for line in (ROOT / "requirements.txt").read_text().splitlines()
        if line.strip() and not line.startswith("#")
    )
    if executable.exists():
        probe = subprocess.run(
            [
                str(executable),
                "-c",
                "import sys,json,importlib.metadata as m; w=json.loads(sys.argv[1]); sys.exit(0 if all(m.version(k)==v for k,v in w.items()) else 1)",
                json.dumps(wanted),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if probe.returncode == 0:
            return str(executable)
    if offline:
        raise RuntimeError(
            "Offline run needs the dependencies already installed. Run once with network access."
        )
    if not executable.exists():
        venv.EnvBuilder(with_pip=True, system_site_packages=True).create(folder)
    print("Preparing dependencies (first run may take a few minutes)...", flush=True)
    setup_log = ROOT / "runtime/setup.log"
    setup_log.parent.mkdir(exist_ok=True)
    with setup_log.open("w", encoding="utf-8") as stream:
        subprocess.run(
            [
                str(executable),
                "-m",
                "pip",
                "install",
                "-r",
                str(ROOT / "requirements.txt"),
            ],
            check=True,
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
    return str(executable)


def environment(python, offline=False):
    env = os.environ.copy()
    env["PYSPARK_PYTHON"] = python
    env["PYSPARK_DRIVER_PYTHON"] = python
    env["SPARK_LOCAL_IP"] = "127.0.0.1"
    env["PYTHONIOENCODING"] = "utf-8"
    if os.name == "nt":
        home = ROOT / "runtime/hadoop"
        binary = home / "bin"
        binary.mkdir(parents=True, exist_ok=True)
        records = []
        for name, digest in NATIVE.items():
            dest = binary / name
            url = (
                "https://raw.githubusercontent.com/cdarlint/winutils/master/hadoop-3.3.5/bin/"
                + name
            )
            if not dest.exists():
                if offline:
                    raise RuntimeError(
                        "Windows Spark runtime is not cached. Run once with network access."
                    )
                with urllib.request.urlopen(url, timeout=60) as response:
                    data = response.read()
                if hashlib.sha256(data).hexdigest() != digest:
                    raise RuntimeError("Windows dependency checksum mismatch: " + name)
                dest.write_bytes(data)
            if hashlib.sha256(dest.read_bytes()).hexdigest() != digest:
                raise RuntimeError("Windows dependency checksum mismatch: " + name)
            records.append(dict(file=name, source=url, sha256=digest))
        (home / "manifest.json").write_text(
            json.dumps(records, indent=2), encoding="utf-8"
        )
        env["HADOOP_HOME"] = str(home)
        env["PATH"] = str(binary) + os.pathsep + env.get("PATH", "")
    return env


def validate_prerequisites():
    if not (3, 10) <= sys.version_info[:2] <= (3, 12):
        raise RuntimeError("Please use Python 3.10–3.12.")
    if not shutil.which("java"):
        raise RuntimeError("Install Java 11 or 17 and make java available on PATH.")
    probe = subprocess.run(["java", "-version"], capture_output=True, text=True)
    version = re.search(r'version\s+"([^"]+)"', probe.stderr + probe.stdout)
    if probe.returncode or version is None:
        raise RuntimeError("Unable to read the Java version.")
    major = int(version.group(1).split(".")[0])
    if major not in (11, 17):
        raise RuntimeError(
            f"Java {major} detected; use Java 11 or 17 for this project."
        )
