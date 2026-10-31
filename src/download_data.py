"""Download and verify the unchanged UCI daily dataset."""

import argparse
import hashlib
import urllib.request
import zipfile
from .common import ROOT, write_json

URL = "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip"
ARCHIVE_SHA256 = "b70182d0d0508e9abbb79306ce5c0cec34869000f8220175ac83d11dbe845401"
DAILY_SHA256 = "a6bcf826782d3c0fbfdcbeead17cd0884185a0dafe8ff10cd48a874ee7ba18be"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_data(offline=False):
    folder = ROOT / "data"
    folder.mkdir(exist_ok=True)
    archive = folder / "bike-sharing.zip"
    daily = folder / "day.csv"
    readme = folder / "Readme.txt"
    if archive.exists() and digest(archive) != ARCHIVE_SHA256:
        raise ValueError("Cached UCI archive checksum differs from the fixed dataset.")
    if not (daily.exists() and digest(daily) == DAILY_SHA256 and readme.exists()):
        if not archive.exists():
            if offline:
                raise RuntimeError(
                    "Dataset is not cached. Run once with network access."
                )
            with urllib.request.urlopen(URL, timeout=60) as response:
                payload = response.read()
            if hashlib.sha256(payload).hexdigest() != ARCHIVE_SHA256:
                raise ValueError(
                    "Downloaded archive checksum differs from the fixed dataset."
                )
            archive.write_bytes(payload)
        with zipfile.ZipFile(archive) as package:
            payload = package.read("day.csv")
            if hashlib.sha256(payload).hexdigest() != DAILY_SHA256:
                raise ValueError("Daily CSV checksum differs from the fixed dataset.")
            daily.write_bytes(payload)
            readme.write_bytes(package.read("Readme.txt"))
    write_json(
        folder / "manifest.json",
        dict(
            source=URL,
            archive_sha256=ARCHIVE_SHA256,
            daily_sha256=digest(daily),
            selected_file="day.csv",
            rows=731,
        ),
    )
    return daily


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true")
    print("Data ready:", prepare_data(parser.parse_args().offline))


if __name__ == "__main__":
    main()
