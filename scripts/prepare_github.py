"""Export a source-only repository and ZIP without changing existing Git history."""

import argparse
from pathlib import Path
import shutil
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
FILES = (
    "README.md",
    "reproduce.py",
    "requirements.txt",
    "extended_config.json",
    ".gitignore",
    ".gitattributes",
    "NOTICE.md",
    "data/SOURCE.md",
    "docs/实验说明.md",
    "docs/模型选择与误差分析说明.md",
    "docs/上传说明.md",
    "docs/复现验证.md",
    "scripts/prepare_github.py",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--destination", type=Path, default=ROOT.parent / "bike-spark-github"
    )
    args = parser.parse_args()
    dest = args.destination.resolve()
    archive = dest.with_suffix(".zip")
    if dest.exists() or archive.exists():
        raise SystemExit(
            "Destination or ZIP already exists; choose a new --destination."
        )
    paths = [ROOT / name for name in FILES] + sorted((ROOT / "src").glob("*.py"))
    paths += sorted((ROOT / "reference").glob("*"))
    if any(not path.is_file() for path in paths):
        raise SystemExit("A required source file is missing.")
    dest.mkdir(parents=True)
    for source in paths:
        target = dest / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    subprocess.run(["git", "init", "-b", "main", str(dest)], check=True)
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as package:
        for source in paths:
            relative = source.relative_to(ROOT)
            package.write(dest / relative, str(Path(dest.name) / relative))
    print(f"Repository: {dest}")
    print(f"Source ZIP: {archive}")


if __name__ == "__main__":
    main()
