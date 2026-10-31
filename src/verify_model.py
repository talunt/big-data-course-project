"""Run separately after training; compare reloaded predictions with saved values."""

import argparse
import csv
import json
from pathlib import Path
from pyspark.sql import SparkSession, functions as F
from .common import load_pipeline

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--output", default=str(Path(__file__).resolve().parents[1] / "outputs")
    )
    args = ap.parse_args()
    out = Path(args.output)
    run = json.loads((out / "run.json").read_text(encoding="utf-8"))
    spark = (
        SparkSession.builder.master("local[2]")
        .appName("VerifySavedBikeModel")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    try:
        # All stages are JVM Spark stages. JVM loader avoids launching a Python
        # RDD worker solely to read metadata (Windows/Python 3.12 compatibility).
        model = load_pipeline(spark, run["model_path"])
        data = spark.read.parquet(run["storage"] + "/sample")
        actual = {
            str(r.dteday): r.prediction
            for r in model.transform(data).select("dteday", "prediction").collect()
        }
        with (out / "sample_expected.csv").open(encoding="utf-8-sig") as f:
            expected = {r["date"]: float(r["prediction"]) for r in csv.DictReader(f)}
        error = max(abs(actual[k] - v) for k, v in expected.items())
        assert set(actual) == set(expected) and error < 1e-8
        # Unknown category should be accepted by the saved pipeline.
        unknown = data.withColumn("weathersit", F.lit(99.0))
        assert model.transform(unknown).filter(F.isnan("prediction")).count() == 0
        (out / "verification.json").write_text(
            json.dumps(
                dict(
                    passed=True,
                    max_prediction_difference=error,
                    rows=len(actual),
                    unknown_category_passed=True,
                ),
                indent=2,
            )
        )
        print(
            "PASS: independent model reload, exact sample comparison and unknown category"
        )
    finally:
        spark.stop()
