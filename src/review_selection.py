"""Reload the historical recommendation and recompute every error-analysis input."""

import argparse
import hashlib
import json
import math
from pathlib import Path
from pyspark.sql import functions as F
from .common import (
    ROOT,
    MODEL_NAMES,
    TOTAL_DAYS,
    TEST_DAYS,
    read_csv,
    export_csv,
    write_json,
    scores,
    create_spark,
    chronological_data,
    raw_daily_frame,
    spark_path,
)
from .models import load_family
from .selection import comparison_rows, recommend


def write_summary(
    output, evidence, validation_family, recommended, overall, difference
):
    by_family = {row["family"]: row for row in evidence}
    displayed = list(dict.fromkeys([recommended, "Ridge", validation_family]))
    lines = [
        "# 模型选择与误差分析复核",
        "",
        f"单次验证集选择：{MODEL_NAMES[validation_family]}。",
        f"综合历史比较推荐：{MODEL_NAMES[recommended]}。",
        "",
        "| 方法 | 验证RMSE | 测试RMSE | 第一窗口 | 第二窗口 | 合并RMSE | 最差窗口 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    keys = [
        "validation_rmse",
        "test_rmse",
        "window1_rmse",
        "window2_rmse",
        "pooled_rolling_rmse",
        "worst_window_rmse",
    ]
    for family in displayed:
        item = by_family[family]
        values = [f"{item[key]:.2f}" if item[key] != "" else "—" for key in keys]
        lines.append("|" + MODEL_NAMES[family] + "|" + "|".join(values) + "|")
    lines += [
        "",
        "综合推荐采用两个历史窗口的合并RMSE最低规则。合并时汇总平方误差和天数，各窗口沿用主验证期所选参数。",
        "",
        f"已重新读取推荐模型，对{overall['n']}个测试日期预测；与训练时导出预测的最大差异为{difference:.3g}。",
        f"复算测试RMSE {overall['rmse']:.2f}、MAE {overall['mae']:.2f}、R² {overall['r2']:.3f}。",
        "",
        "这是已观察历史结果的回顾性比较。后续用新增日期和滚动验证检验结论。",
        "",
        "图表位于 figures/，逐日预测和分组指标位于 predictions.csv、groups.csv。",
    ]
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--experiments", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    original = read_csv(args.experiments / "predictions.csv")
    metric_rows = read_csv(args.experiments / "metrics.csv")
    rolling = read_csv(args.experiments / "rolling.csv")
    parameters = json.loads(
        (args.experiments / "selected_params.json").read_text(encoding="utf-8")
    )
    validation_family = json.loads(
        (args.experiments / "run.json").read_text(encoding="utf-8")
    )["selected_family"]
    evidence = comparison_rows(metric_rows, rolling)
    recommended = recommend(evidence)
    assert (
        validation_family
        == min(evidence, key=lambda row: row["validation_rmse"])["family"]
    )
    spark = create_spark("BikeModelSelectionReview")
    try:
        data = chronological_data(raw_daily_frame(spark, ROOT / "data/day.csv"))
        assert data.count() == TOTAL_DAYS
        test = data.filter(F.col("split") == "test").cache()
        assert test.count() == TEST_DAYS
        path = spark_path(args.model_root) + "/" + recommended
        model = load_family(spark, recommended, path, parameters[recommended])
        predictions = (
            model.transform(test)
            .withColumn("error", F.col("prediction") - F.col("cnt"))
            .cache()
        )
        rows = [
            dict(
                date=str(row.dteday),
                cnt=float(row.cnt),
                prediction=float(row.prediction),
                error=float(row.error),
                weathersit=int(row.weathersit),
                workingday=int(row.workingday),
                mnth=int(row.mnth),
            )
            for row in predictions.orderBy("dteday")
            .select(
                "dteday",
                "cnt",
                "prediction",
                "error",
                "weathersit",
                "workingday",
                "mnth",
            )
            .collect()
        ]
        before = {
            row["date"]: float(row["prediction"])
            for row in original
            if row["family"] == recommended
        }
        assert len(rows) == TEST_DAYS and set(before) == {row["date"] for row in rows}
        difference = max(abs(row["prediction"] - before[row["date"]]) for row in rows)
        assert difference < 1e-8
        overall = scores(rows)
        expected = next(row for row in metric_rows if row["family"] == recommended)
        for name in ["rmse", "mae", "r2"]:
            assert math.isclose(
                overall[name], float(expected[name]), rel_tol=1e-10, abs_tol=1e-8
            )
        groups = []
        for column in ["weathersit", "workingday", "mnth"]:
            aggregate = (
                predictions.groupBy(column)
                .agg(
                    F.count("*").alias("n"),
                    F.sqrt(F.avg(F.col("error") ** 2)).alias("rmse"),
                    F.avg(F.abs("error")).alias("mae"),
                    F.avg("error").alias("bias"),
                )
                .orderBy(column)
            )
            for row in aggregate.collect():
                independent = scores(
                    [item for item in rows if item[column] == int(row[column])]
                )
                for name in ["rmse", "mae", "bias"]:
                    assert math.isclose(
                        float(row[name]), independent[name], rel_tol=1e-10, abs_tol=1e-8
                    )
                groups.append(
                    dict(
                        group=column,
                        value=int(row[column]),
                        n=int(row.n),
                        rmse=float(row.rmse),
                        mae=float(row.mae),
                        bias=float(row.bias),
                    )
                )
        competitors = {
            row["family"]: scores(
                [item for item in original if item["family"] == row["family"]]
            )
            for row in metric_rows
        }
        last = model.stages[-1] if hasattr(model, "stages") else None
        report = dict(
            analysis_kind="retrospective historical comparison",
            original_validation_selection=validation_family,
            recommended_family=recommended,
            recommendation_rule="minimum pooled RMSE across the two registered historical windows; tie-break by worst-window and validation RMSE",
            recommendation_scope="Historical results were already observed; confirm on new dates.",
            parameters=parameters[recommended],
            feature_coefficients=len(last.coefficients)
            if last is not None and hasattr(last, "coefficients")
            else None,
            model_path=path,
            test_date_range=[rows[0]["date"], rows[-1]["date"]],
            overall=overall,
            reloaded_prediction_max_difference=difference,
            comparisons=competitors,
            rolling_rank=sorted(
                (row for row in evidence if row["pooled_rolling_rmse"] != ""),
                key=lambda row: row["pooled_rolling_rmse"],
            ),
            spark_version=spark.version,
            application_id=spark.sparkContext.applicationId,
            data_sha256=hashlib.sha256(
                (ROOT / "data/day.csv").read_bytes()
            ).hexdigest(),
        )
        export_csv(output / "predictions.csv", rows)
        export_csv(output / "groups.csv", groups)
        export_csv(
            output / "worst_errors.csv",
            sorted(rows, key=lambda row: abs(row["error"]), reverse=True)[:10],
        )
        export_csv(output / "comparison.csv", evidence)
        write_json(output / "review.json", report)
        write_summary(
            output, evidence, validation_family, recommended, overall, difference
        )
        from .plot_selection_review import render

        render(output)
        print(
            json.dumps(
                dict(
                    recommended=recommended,
                    overall=overall,
                    reload_difference=difference,
                ),
                ensure_ascii=False,
            )
        )
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
