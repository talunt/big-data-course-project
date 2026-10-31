"""Plot only exported small result tables; model computation stays in Spark."""

from pathlib import Path
from .common import ROOT, read_csv
import argparse
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default=str(ROOT / "outputs"))
    a = ap.parse_args()
    output = Path(a.output)

    def rows(name):
        return read_csv(output / name)

    dest = output / "figures"
    dest.mkdir(exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 150,
        }
    )

    def save(name):
        plt.tight_layout()
        plt.savefig(dest / name)
        plt.close()

    m = rows("monthly.csv")
    plt.figure(figsize=(10, 4))
    plt.plot(
        [f"{r['year']}-{int(r['month']):02}" for r in m],
        [float(r["avg_count"]) for r in m],
        marker="o",
        color="#147d92",
    )
    plt.xticks(rotation=60)
    plt.ylabel("Mean daily rentals")
    save("monthly.png")
    v = [r for r in rows("validation.csv") if r["model"] == "LR-F1"]
    plt.figure(figsize=(7, 4))
    plt.bar([r["regParam"] for r in v], [float(r["rmse"]) for r in v], color="#147d92")
    plt.xlabel("regParam")
    plt.ylabel("Validation RMSE")
    save("regularization.png")
    p = [r for r in rows("predictions.csv") if r["model"] == "LR-F1"]
    y = [float(r["cnt"]) for r in p]
    pred = [float(r["prediction"]) for r in p]
    plt.figure(figsize=(10, 4))
    plt.plot([r["date"] for r in p], y, label="Actual", color="#173751")
    plt.plot([r["date"] for r in p], pred, label="Predicted", color="#d57933")
    plt.xticks(range(0, len(p), 15), [r["date"] for r in p][::15], rotation=30)
    plt.ylabel("Daily rentals")
    plt.legend()
    save("test_timeline.png")
    plt.figure(figsize=(6, 5))
    plt.scatter(y, pred, alpha=0.7, color="#147d92")
    lo = min(y + pred)
    hi = max(y + pred)
    plt.plot([lo, hi], [lo, hi], "--", color="#d57933")
    plt.xlabel("Actual rentals")
    plt.ylabel("Predicted rentals")
    save("scatter.png")
    print("Four result figures generated")


if __name__ == "__main__":
    main()
