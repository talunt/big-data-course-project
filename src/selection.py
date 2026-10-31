"""Explicit retrospective comparison; the validation-only winner stays separate."""

import math


def comparison_rows(metric_rows, rolling_rows):
    result = []
    for metric in metric_rows:
        family = metric["family"]
        row = dict(
            family=family,
            validation_rmse=float(metric["validation_rmse"]),
            test_rmse=float(metric["rmse"]),
            test_mae=float(metric["mae"]),
            test_r2=float(metric["r2"]),
        )
        windows = sorted(
            (item for item in rolling_rows if item["family"] == family),
            key=lambda item: int(item["train_end"]),
        )
        row.update(
            window1_rmse="",
            window2_rmse="",
            pooled_rolling_rmse="",
            worst_window_rmse="",
            pooled_rolling_mae="",
        )
        if windows:
            if len(windows) != 2:
                raise ValueError(
                    "The registered historical comparison needs two windows per method"
                )
            count = sum(int(item["horizon"]) for item in windows)
            row.update(
                window1_rmse=float(windows[0]["rmse"]),
                window2_rmse=float(windows[1]["rmse"]),
                pooled_rolling_rmse=math.sqrt(
                    sum(
                        int(item["horizon"]) * float(item["rmse"]) ** 2
                        for item in windows
                    )
                    / count
                ),
                worst_window_rmse=max(float(item["rmse"]) for item in windows),
                pooled_rolling_mae=sum(
                    int(item["horizon"]) * float(item["mae"]) for item in windows
                )
                / count,
            )
        result.append(row)
    return result


def recommend(comparison):
    candidates = [row for row in comparison if row["pooled_rolling_rmse"] != ""]
    if not candidates:
        raise ValueError(
            "Historical recommendation requires completed rolling-window results"
        )
    return min(
        candidates,
        key=lambda row: (
            row["pooled_rolling_rmse"],
            row["worst_window_rmse"],
            row["validation_rmse"],
        ),
    )["family"]
