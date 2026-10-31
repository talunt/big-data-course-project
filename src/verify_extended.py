"""Independent process: arithmetic checks and recovery of nine saved families."""

import json
import csv
import math
import argparse
from pathlib import Path
from pyspark.sql import SparkSession, functions as F
from .common import load_pipeline
from .models import KNN


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--storage", default="hdfs://master:9000/course/bike")
    ap.add_argument("--output", default="outputs-extended")
    a = ap.parse_args()
    out = Path(a.output)

    def rows(name):
        with (out / name).open(encoding="utf-8-sig") as f:
            return list(csv.DictReader(f))

    preds = rows("predictions.csv")
    report = []
    spark = (
        SparkSession.builder.master("local[2]")
        .appName("VerifyExtendedModels")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    try:
        sample = (
            spark.read.parquet(a.storage + "/clean")
            .filter(F.col("split") == "test")
            .orderBy("dteday")
            .limit(5)
        )
        for m in rows("metrics.csv"):
            family = m["family"]
            p = [x for x in preds if x["family"] == family]
            assert len(p) == 110 and len({x["date"] for x in p}) == 110
            assert all("2012-09-13" <= x["date"] <= "2012-12-31" for x in p)
            y = [float(x["cnt"]) for x in p]
            v = [float(x["prediction"]) for x in p]
            assert all(math.isfinite(x) for x in v)
            sse = sum((x - z) ** 2 for x, z in zip(y, v))
            mean = sum(y) / len(y)
            calc = dict(
                rmse=math.sqrt(sse / len(y)),
                mae=sum(abs(x - z) for x, z in zip(y, v)) / len(y),
                r2=1 - sse / sum((x - mean) ** 2 for x in y),
            )
            assert all(
                math.isclose(x, float(m[k]), rel_tol=1e-10, abs_tol=1e-8)
                for k, x in calc.items()
            )
            path = a.storage + "/extended_models/" + family
            if family == "KNN":
                model = object.__new__(KNN)
                model.k = json.loads(m["params"])["k"]
                model.points = spark.read.parquet(path + "/points")
                model.prep = load_pipeline(spark, path + "/preprocessing")
            else:
                model = load_pipeline(spark, path)
            actual = {
                str(r.dteday): r.prediction for r in model.transform(sample).collect()
            }
            expected = {x["date"]: float(x["prediction"]) for x in p}
            error = max(abs(v - expected[k]) for k, v in actual.items())
            assert len(actual) == 5 and error < 1e-8
            if family == "KNN":
                # Independent brute-force oracle for the SQL distance/window implementation.
                points = model.points.collect()
                for query in (
                    model.prep.transform(sample)
                    .select("dteday", "row_number", "features")
                    .collect()
                ):
                    q = list(query.features)
                    nearest = sorted(
                        (
                            sum((x - y) ** 2 for x, y in zip(q, t.train_x)),
                            t.train_id,
                            t.train_label,
                        )
                        for t in points
                        if t.train_id != query.row_number
                    )[: model.k]
                    reference = sum(t[2] for t in nearest) / len(nearest)
                    assert math.isclose(
                        reference,
                        actual[str(query.dteday)],
                        rel_tol=1e-10,
                        abs_tol=1e-8,
                    )
            report.append(
                dict(
                    family=family,
                    arithmetic_passed=True,
                    reload_passed=True,
                    max_difference=error,
                )
            )
        (out / "independent_checks.json").write_text(
            json.dumps(
                dict(
                    passed=True,
                    separate_process=True,
                    knn_brute_force_oracle_passed=True,
                    families=report,
                ),
                indent=2,
            ),
            encoding="utf-8",
        )
        print("PASS: independent metrics and saved-model recovery of all nine families")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
