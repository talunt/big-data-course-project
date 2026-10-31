"""Run the mean/weather/calendar baseline and audit the fixed dataset."""

import argparse
import json
import platform
import time
from pathlib import Path
from pyspark.sql import functions as F
from .common import (
    ROOT,
    HEADERS,
    NUMERIC,
    CATEGORIES,
    export_csv,
    write_json,
    metrics,
    raw_daily_frame,
    chronological_data,
    spark_path,
    create_spark,
)
from .models import baseline_pipeline as pipeline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(ROOT / "data/day.csv"))
    ap.add_argument(
        "--storage",
        help="Spark-writable storage root, e.g. hdfs://namenode:9000/course/bike",
    )
    ap.add_argument("--master", default="local[2]")
    ap.add_argument("--output", default=str(ROOT / "outputs"))
    args = ap.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    events = out / "events"
    spark = create_spark("BikeDailyRegression", args.master, events)
    try:
        started = time.time()
        headers = HEADERS
        raw = raw_daily_frame(spark, args.input)
        n = raw.count()
        nulls = {c: raw.filter(F.col(c).isNull()).count() for c in headers}
        duplicate_dates = n - raw.select("dteday").distinct().count()
        bad_counts = raw.filter(
            (F.col("cnt") < 0) | (F.col("casual") + F.col("registered") != F.col("cnt"))
        ).count()
        invalid_ranges = raw.filter(
            ~F.col("temp").between(0, 1)
            | ~F.col("hum").between(0, 1)
            | ~F.col("windspeed").between(0, 1)
            | ~F.col("yr").isin(0, 1)
            | ~F.col("mnth").between(1, 12)
            | ~F.col("weekday").between(0, 6)
            | ~F.col("workingday").isin(0, 1)
            | ~F.col("weathersit").between(1, 4)
        ).count()
        quality = dict(
            rows=n,
            nulls=nulls,
            duplicate_dates=duplicate_dates,
            invalid_counts=bad_counts,
            invalid_ranges=invalid_ranges,
        )
        write_json(out / "quality.json", quality)
        assert (
            n == 731
            and not any(nulls.values())
            and duplicate_dates == 0
            and bad_counts == 0
            and invalid_ranges == 0
        ), quality
        df = chronological_data(raw)
        assert df.filter(F.col("dteday").isNull()).count() == 0
        df.createOrReplaceTempView("bikes")
        for name, sql in {
            "monthly": "SELECT year(dteday) year, month(dteday) month, avg(cnt) avg_count FROM bikes GROUP BY 1,2 ORDER BY 1,2",
            "workingday": "SELECT workingday, count(*) days, avg(cnt) avg_count FROM bikes GROUP BY workingday ORDER BY workingday",
            "weather": "SELECT weathersit, count(*) days, avg(cnt) avg_count FROM bikes GROUP BY weathersit ORDER BY weathersit",
        }.items():
            export_csv(out / (name + ".csv"), spark.sql(sql).collect())
        df = df.cache()
        split_info = [
            r.asDict()
            for r in df.groupBy("split")
            .agg(
                F.count("*").alias("rows"),
                F.min("dteday").cast("string").alias("start"),
                F.max("dteday").cast("string").alias("end"),
            )
            .collect()
        ]
        write_json(out / "split_manifest.json", split_info)
        assert sorted(r["rows"] for r in split_info) == [110, 110, 511]
        storage = spark_path(args.storage or ROOT / "storage")
        df.write.mode("overwrite").parquet(storage + "/clean")
        df = spark.read.parquet(storage + "/clean").cache()
        train, val, test = [
            df.filter(F.col("split") == s) for s in ["train", "validation", "test"]
        ]
        mean = train.agg(F.avg("cnt")).first()[0]
        records = []
        candidates = []
        for name, full, reg in [("LR-F0", False, 0.01)] + [
            ("LR-F1", True, r) for r in [0, 0.01, 0.1]
        ]:
            t = time.time()
            model = pipeline(full, reg).fit(train)
            score = metrics(model.transform(val))
            records.append(
                dict(model=name, regParam=reg, **score, train_seconds=time.time() - t)
            )
            candidates.append((name, reg, model, score["rmse"]))
        export_csv(out / "validation.csv", records)
        chosen = min(
            [c for c in candidates if c[0] == "LR-F1"], key=lambda c: (c[3], -c[1])
        )
        results = []
        predictions = []
        baseline = test.withColumn("prediction", F.lit(mean))
        for name, reg, pred in [
            ("Mean", None, baseline),
            ("LR-F0", 0.01, candidates[0][2].transform(test)),
            ("LR-F1", chosen[1], chosen[2].transform(test)),
        ]:
            row = dict(
                model=name,
                regParam=reg,
                **metrics(pred),
                rows=pred.count(),
                negative_predictions=pred.filter(F.col("prediction") < 0).count(),
            )
            results.append(row)
            predictions.extend(
                pred.select(
                    F.col("dteday").cast("string").alias("date"), "cnt", "prediction"
                )
                .withColumn("model", F.lit(name))
                .orderBy("date")
                .collect()
            )
        export_csv(out / "metrics.csv", results)
        export_csv(out / "predictions.csv", predictions)
        best_path = storage + "/best_model"
        chosen[2].write().overwrite().save(best_path)
        sample = test.orderBy("dteday").limit(5)
        export_csv(
            out / "sample_expected.csv",
            chosen[2]
            .transform(sample)
            .select(F.col("dteday").cast("string").alias("date"), "prediction")
            .orderBy("date")
            .collect(),
        )
        sample.write.mode("overwrite").parquet(storage + "/sample")
        export_csv(
            out / "worst_errors.csv",
            chosen[2]
            .transform(test)
            .withColumn("absolute_error", F.abs(F.col("cnt") - F.col("prediction")))
            .orderBy(F.desc("absolute_error"))
            .select(
                F.col("dteday").cast("string").alias("date"),
                "cnt",
                "prediction",
                "absolute_error",
            )
            .limit(3)
            .collect(),
        )
        write_json(
            out / "run.json",
            dict(
                spark=spark.version,
                python=platform.python_version(),
                master=args.master,
                input=args.input,
                storage=storage,
                model_path=best_path,
                selected_regParam=chosen[1],
                application_id=spark.sparkContext.applicationId,
                elapsed_seconds=time.time() - started,
                feature_columns=NUMERIC + ["yr", "workingday"] + CATEGORIES,
                baseline_mean=mean,
            ),
        )
        print(json.dumps(results, indent=2))
        print("OUTPUT", out)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
