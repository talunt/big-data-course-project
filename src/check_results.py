"""Independent arithmetic and split checks on Spark exports (stdlib only)."""

import argparse
import csv
import json
import math
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--output", default=str(Path(__file__).resolve().parents[1] / "outputs")
    )
    a = ap.parse_args()
    root = Path(a.output)

    def read(name):
        with (root / name).open(encoding="utf-8-sig") as f:
            return list(csv.DictReader(f))

    splits = {
        r["split"]: r
        for r in json.loads((root / "split_manifest.json").read_text(encoding="utf-8"))
    }
    assert (
        splits["train"]["end"]
        < splits["validation"]["start"]
        <= splits["validation"]["end"]
        < splits["test"]["start"]
    )
    assert sum(r["rows"] for r in splits.values()) == 731
    preds = read("predictions.csv")
    checks = []
    for m in read("metrics.csv"):
        rows = [r for r in preds if r["model"] == m["model"]]
        assert len(rows) == 110 and len({r["date"] for r in rows}) == 110
        assert all(
            splits["test"]["start"] <= r["date"] <= splits["test"]["end"] for r in rows
        )
        y = [float(r["cnt"]) for r in rows]
        p = [float(r["prediction"]) for r in rows]
        sse = sum((v - w) ** 2 for v, w in zip(y, p))
        avg = sum(y) / len(y)
        calculated = dict(
            rmse=math.sqrt(sse / len(y)),
            mae=sum(abs(v - w) for v, w in zip(y, p)) / len(y),
            r2=1 - sse / sum((v - avg) ** 2 for v in y),
        )
        assert all(
            math.isclose(calculated[k], float(m[k]), rel_tol=1e-10, abs_tol=1e-8)
            for k in calculated
        )
        checks.append(dict(model=m["model"], passed=True, **calculated))
    assert json.loads((root / "verification.json").read_text())["passed"]
    (root / "independent_checks.json").write_text(
        json.dumps(
            dict(passed=True, chronological_split=True, metrics=checks), indent=2
        ),
        encoding="utf-8",
    )
    print(
        "PASS: chronological split, 110 unique test dates per model, independent RMSE/MAE/R2"
    )


if __name__ == "__main__":
    main()
