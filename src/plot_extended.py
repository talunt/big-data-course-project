"""Figures made from the small exported result tables."""

import argparse
import csv
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--output",
        default=str(Path(__file__).resolve().parents[1] / "outputs-extended"),
    )
    folder = Path(ap.parse_args().output)
    dest = folder / "figures"
    dest.mkdir(exist_ok=True)

    def rows(name):
        with (folder / name).open(encoding="utf-8-sig") as f:
            return list(csv.DictReader(f))

    def save(name):
        plt.tight_layout()
        plt.savefig(dest / name, dpi=160)
        plt.close()

    plt.rcParams.update(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    )
    metrics = rows("metrics.csv")
    names = [r["family"] for r in metrics]
    plt.figure(figsize=(11, 5))
    plt.bar(names, [float(r["rmse"]) for r in metrics], color="#147d92")
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Test RMSE")
    save("test_comparison.png")

    x = np.arange(len(names))
    plt.figure(figsize=(12, 5))
    for i, (key, title) in enumerate(
        [
            ("train_rmse", "Train (KNN: LOO)"),
            ("validation_rmse", "Validation"),
            ("rmse", "Test"),
        ]
    ):
        plt.bar(
            x + (i - 1) * 0.25,
            [float(r[key]) for r in metrics],
            width=0.25,
            label=title,
        )
    plt.xticks(x, names, rotation=30, ha="right")
    plt.ylabel("RMSE")
    plt.legend()
    save("generalization.png")

    from json import loads

    selected = loads((folder / "run.json").read_text(encoding="utf-8"))[
        "selected_family"
    ]
    preds = sorted(
        (r for r in rows("predictions.csv") if r["family"] == selected),
        key=lambda r: r["date"],
    )
    actual = [float(r["cnt"]) for r in preds]
    predicted = [float(r["prediction"]) for r in preds]
    plt.figure(figsize=(11, 4))
    plt.plot(actual, label="Actual")
    plt.plot(predicted, label=selected)
    plt.xticks(range(0, len(preds), 20), [r["date"] for r in preds][::20], rotation=25)
    plt.ylabel("Daily rentals")
    plt.legend()
    save("selected_timeline.png")
    plt.figure(figsize=(6, 5))
    plt.scatter(actual, predicted, alpha=0.65)
    bounds = [min(actual + predicted), max(actual + predicted)]
    plt.plot(bounds, bounds, "--", color="gray")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    save("selected_scatter.png")

    groups = sorted(
        (r for r in rows("groups.csv") if r["group"] == "weathersit"),
        key=lambda r: float(r["value"]),
    )
    plt.figure(figsize=(7, 4))
    plt.bar(
        [f"Weather {r['value']} (n={r['n']})" for r in groups],
        [float(r["rmse"]) for r in groups],
    )
    plt.ylabel("Test RMSE")
    save("weather_groups.png")
    print("Five extended figures generated")


if __name__ == "__main__":
    main()
