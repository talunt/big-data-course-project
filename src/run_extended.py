"""Fixed-budget comparison: validation selection, saved models and historical windows."""

import argparse
import hashlib
import json
import time
from pathlib import Path
from pyspark.sql import functions as F
from .common import (
    ROOT,
    export_csv,
    write_json,
    metrics,
    create_spark,
    load_pipeline,
    spark_path,
)
from .models import fit, KNN


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--storage", default="hdfs://master:9000/course/bike")
    ap.add_argument("--output", default=str(ROOT / "outputs-extended"))
    a = ap.parse_args()
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    config_path = ROOT / "extended_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    a.storage = spark_path(a.storage)
    events = out / "events"
    events.mkdir(exist_ok=True)
    spark = create_spark("BikeExtendedRegression", event_dir=events)
    start = time.time()
    try:
        data = spark.read.parquet(a.storage + "/clean").cache()
        assert data.count() == 731
        train, val, test = [
            data.filter(F.col("split") == s).cache()
            for s in ["train", "validation", "test"]
        ]
        assert [x.count() for x in [train, val, test]] == [511, 110, 110]
        registry = dict(
            config=config,
            config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
            input=a.storage + "/clean",
            application_id=spark.sparkContext.applicationId,
        )
        write_json(out / "registry.json", registry)
        search = []
        selected = {}
        chosen_models = {}
        summary = []
        all_preds = []
        interpret = []
        for family, grid in config["families"].items():
            candidates = []
            for idx, params in enumerate(grid):
                t = time.time()
                model = fit(family, params, train)
                score = metrics(model.transform(val))
                row = dict(
                    family=family,
                    candidate=idx,
                    params=json.dumps(params, sort_keys=True),
                    validation_rmse=score["rmse"],
                    validation_mae=score["mae"],
                    validation_r2=score["r2"],
                    fit_eval_seconds=time.time() - t,
                )
                search.append(row)
                candidates.append((score["rmse"], idx, params, model))
                print("CANDIDATE", family, params, score, flush=True)
            winner = min(candidates, key=lambda x: (x[0], x[1]))
            _, _, params, model = winner
            selected[family] = params
            chosen_models[family] = model
            export_csv(out / "search.csv", search)
        # Selection is frozen on validation before any new test predictions.
        write_json(out / "selected_params.json", selected)
        winner_family = min(
            selected,
            key=lambda name: min(
                r["validation_rmse"] for r in search if r["family"] == name
            ),
        )
        for family, model in chosen_models.items():
            train_score = metrics(model.transform(train))
            val_score = metrics(model.transform(val))
            pred = model.transform(test).cache()
            score = metrics(pred)
            assert (
                pred.count() == 110
                and pred.filter(
                    F.isnan("prediction") | F.col("prediction").isNull()
                ).count()
                == 0
            )
            summary.append(
                dict(
                    family=family,
                    params=json.dumps(selected[family], sort_keys=True),
                    train_rmse=train_score["rmse"],
                    validation_rmse=val_score["rmse"],
                    rmse=score["rmse"],
                    mae=score["mae"],
                    r2=score["r2"],
                    rows=110,
                    negative_predictions=pred.filter(F.col("prediction") < 0).count(),
                )
            )
            all_preds += (
                pred.select(
                    F.col("dteday").cast("string").alias("date"),
                    "cnt",
                    "prediction",
                    "weathersit",
                    "workingday",
                    "mnth",
                )
                .withColumn("family", F.lit(family))
                .collect()
            )
            path = a.storage + "/extended_models/" + family
            if family == "KNN":
                model.save(path)
            else:
                model.write().overwrite().save(path)
                attrs = (
                    model.transform(train)
                    .schema["features"]
                    .metadata.get("ml_attr", {})
                    .get("attrs", {})
                )
                names = {x["idx"]: x["name"] for items in attrs.values() for x in items}
                last = model.stages[-1]
                weights = (
                    last.coefficients
                    if hasattr(last, "coefficients")
                    else last.featureImportances
                )
                for i, v in enumerate(weights):
                    interpret.append(
                        dict(
                            family=family,
                            feature=names.get(i, str(i)),
                            value=float(v),
                            kind="coefficient"
                            if hasattr(last, "coefficients")
                            else "importance",
                        )
                    )
                if hasattr(last, "summary"):
                    hist = list(last.summary.objectiveHistory)
                    write_json(
                        out / (family + "_objective.json"),
                        dict(
                            objective=hist,
                            iterations=last.summary.totalIterations,
                            nonzero_coefficients=sum(
                                abs(float(x)) > 1e-8 for x in weights
                            ),
                        ),
                    )
            pred.unpersist()
        export_csv(out / "metrics.csv", summary)
        export_csv(out / "predictions.csv", all_preds)
        export_csv(out / "interpretation.csv", interpret)
        checks = []
        # Restore every family in a separate object, on a fixed test sample.
        for family, original in chosen_models.items():
            path = a.storage + "/extended_models/" + family
            sample = test.orderBy("dteday").limit(5)
            if family == "KNN":
                prep = load_pipeline(spark, path + "/preprocessing")
                restored = object.__new__(KNN)
                restored.prep = prep
                restored.k = selected[family]["k"]
                restored.points = spark.read.parquet(path + "/points")
            else:
                restored = load_pipeline(spark, path)
            before = {
                r.dteday: r.prediction for r in original.transform(sample).collect()
            }
            after = {
                r.dteday: r.prediction for r in restored.transform(sample).collect()
            }
            err = max(abs(after[k] - v) for k, v in before.items())
            assert err < 1e-8
            checks.append(
                dict(family=family, rows=len(after), max_difference=err, passed=True)
            )
        write_json(out / "reload_checks.json", checks)
        rolling = []
        for end in config["rolling_train_ends"]:
            tr = data.filter(F.col("row_number") <= end)
            hold = data.filter(
                (F.col("row_number") > end)
                & (F.col("row_number") <= end + config["rolling_horizon"])
            )
            for family in config["rolling_families"]:
                m = fit(family, selected[family], tr)
                score = metrics(m.transform(hold))
                rolling.append(
                    dict(family=family, train_end=end, horizon=hold.count(), **score)
                )
                print("ROLLING", end, family, score, flush=True)
                if family == "KNN":
                    m.points.unpersist()
        export_csv(out / "rolling.csv", rolling)
        final = (
            chosen_models[winner_family]
            .transform(test)
            .withColumn("error", F.col("prediction") - F.col("cnt"))
            .cache()
        )
        groups = []
        for column in ["weathersit", "workingday", "mnth"]:
            for r in (
                final.groupBy(column)
                .agg(
                    F.count("*").alias("n"),
                    F.sqrt(F.avg(F.col("error") ** 2)).alias("rmse"),
                    F.avg(F.abs("error")).alias("mae"),
                    F.avg("error").alias("bias"),
                )
                .collect()
            ):
                row = r.asDict()
                value = row.pop(column)
                groups.append(dict(group=column, value=value, **row))
        export_csv(out / "groups.csv", groups)
        export_csv(
            out / "worst_errors.csv",
            final.orderBy(F.desc(F.abs("error")))
            .select(
                F.col("dteday").cast("string").alias("date"),
                "cnt",
                "prediction",
                "error",
            )
            .limit(5)
            .collect(),
        )
        write_json(
            out / "run.json",
            dict(
                selected_family=winner_family,
                selection="validation RMSE only",
                exploratory=True,
                config_sha256=registry["config_sha256"],
                candidate_count=len(search),
                rolling_fits=len(rolling),
                spark=spark.version,
                application_id=spark.sparkContext.applicationId,
                elapsed_seconds=time.time() - start,
                storage=a.storage,
                model_path=a.storage + "/extended_models/" + winner_family,
            ),
        )
        print("SELECTED", winner_family, "ALL CHECKS PASSED", flush=True)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
