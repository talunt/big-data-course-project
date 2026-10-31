"""Draw the recomputed recommendation, grouped errors and error distribution."""

import argparse
from datetime import datetime
import json
from pathlib import Path
from .common import MODEL_NAMES, read_csv
from .plotting import EN_MODEL_NAMES, configure, plt, save_figure


def render(folder):
    folder = Path(folder)
    output = folder / "figures"
    chinese = configure()

    def label(zh, en):
        return zh if chinese else en

    family = json.loads((folder / "review.json").read_text(encoding="utf-8"))[
        "recommended_family"
    ]
    name = MODEL_NAMES[family] if chinese else EN_MODEL_NAMES[family]
    rows = read_csv(folder / "predictions.csv")
    dates = [datetime.fromisoformat(row["date"]) for row in rows]
    actual = [float(row["cnt"]) for row in rows]
    predicted = [float(row["prediction"]) for row in rows]
    import matplotlib.dates as mdates

    plt.figure(figsize=(12, 4))
    plt.plot(
        dates, actual, label=label("实际租赁量", "Actual rentals"), color="#CD1642"
    )
    plt.plot(dates, predicted, label=name, color="#666666")
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=14))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    plt.ylabel(label("每日租赁量", "Daily rentals"))
    plt.xlabel(label("2012年测试日期", "Test dates in 2012"))
    plt.legend()
    save_figure(output, "daily_predictions.png")
    groups = [
        row for row in read_csv(folder / "groups.csv") if row["group"] == "weathersit"
    ]
    weather = [
        label("晴朗等", "Clear"),
        label("雾或多云", "Mist/cloudy"),
        label("小雨雪等", "Light rain/snow"),
    ]
    plt.figure(figsize=(7, 4))
    plt.bar(
        [f"{text}\nn={row['n']}" for text, row in zip(weather, groups)],
        [float(row["rmse"]) for row in groups],
        color="#CD1642",
    )
    plt.ylabel("RMSE")
    plt.title(name)
    save_figure(output, "weather_errors.png")
    plt.figure(figsize=(6, 5))
    plt.scatter(actual, predicted, s=20, color="#CD1642", alpha=0.7)
    limit = max(actual + predicted)
    plt.plot([0, limit], [0, limit], color="#777777", linewidth=1)
    plt.xlabel(label("实际租赁量", "Actual rentals"))
    plt.ylabel(label("预测租赁量", "Predicted rentals"))
    plt.title(name)
    save_figure(output, "actual_predicted.png")
    plt.figure(figsize=(7, 4))
    plt.hist(
        [estimate - value for value, estimate in zip(actual, predicted)],
        bins=16,
        color="#CD1642",
        edgecolor="white",
    )
    plt.xlabel(label("预测减实际", "Predicted minus actual"))
    plt.ylabel(label("天数", "Days"))
    plt.title(name)
    save_figure(output, "error_distribution.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    render(parser.parse_args().output)


if __name__ == "__main__":
    main()
