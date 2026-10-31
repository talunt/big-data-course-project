"""Shared data definitions, file I/O and evaluation helpers."""

from pathlib import Path
import csv
import json
import math

ROOT = Path(__file__).resolve().parents[1]
HEADERS = "instant dteday season yr mnth holiday weekday workingday weathersit temp atemp hum windspeed casual registered cnt".split()
NUMERIC = ["temp", "hum", "windspeed"]
CATEGORIES = ["mnth", "weekday", "weathersit"]
TRAIN_DAYS = 511
VALIDATION_END = 621
TOTAL_DAYS = 731
TEST_DAYS = 110
MODEL_NAMES = {
    "Mean": "均值预测",
    "OLS": "普通线性回归",
    "Ridge": "岭回归",
    "Lasso": "Lasso回归",
    "ElasticNet": "弹性网络回归",
    "PolyRidge": "新增计算项的岭回归",
    "DecisionTree": "决策树回归",
    "RandomForest": "随机森林",
    "GBT": "梯度提升树",
    "KNN": "K近邻回归",
}


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def export_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    records = [row.asDict() if hasattr(row, "asDict") else row for row in rows]
    if not records:
        raise ValueError(f"No records to export: {path}")
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def spark_path(path):
    """Keep non-ASCII local paths literal; preserve explicit storage URIs."""
    value = str(path)
    if "://" in value:
        return value.rstrip("/")
    return Path(value).resolve().as_posix()


def create_spark(name, master="local[2]", event_dir=None):
    from pyspark.sql import SparkSession

    builder = (
        SparkSession.builder.appName(name)
        .master(master)
        .config("spark.sql.shuffle.partitions", "2")
    )
    if event_dir is not None:
        Path(event_dir).mkdir(parents=True, exist_ok=True)
        builder = builder.config("spark.eventLog.enabled", "true").config(
            "spark.eventLog.dir", Path(event_dir).resolve().as_uri()
        )
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark


def raw_daily_frame(spark, path):
    from pyspark.sql.types import StructType, StructField, StringType, DoubleType

    schema = StructType(
        [
            StructField(name, StringType() if name == "dteday" else DoubleType(), False)
            for name in HEADERS
        ]
    )
    return spark.read.option("header", True).schema(schema).csv(spark_path(path))


def chronological_data(raw):
    from pyspark.sql import functions as F, Window

    data = raw.withColumn("dteday", F.to_date("dteday")).drop(
        "instant", "casual", "registered"
    )
    data = data.withColumn("row_number", F.row_number().over(Window.orderBy("dteday")))
    return data.withColumn(
        "split",
        F.when(F.col("row_number") <= TRAIN_DAYS, "train")
        .when(F.col("row_number") <= VALIDATION_END, "validation")
        .otherwise("test"),
    )


def metrics(predictions):
    from pyspark.ml.evaluation import RegressionEvaluator

    return {
        name: RegressionEvaluator(
            labelCol="cnt", predictionCol="prediction", metricName=name
        ).evaluate(predictions)
        for name in ["rmse", "mae", "r2"]
    }


def scores(rows):
    """Independent, driver-side verification of exported predictions."""
    actual = [float(row["cnt"]) for row in rows]
    errors = [float(row["prediction"]) - value for row, value in zip(rows, actual)]
    if not actual or not all(math.isfinite(value) for value in actual + errors):
        raise ValueError("Predictions must be non-empty and finite")
    mean = sum(actual) / len(actual)
    sse = sum(value * value for value in errors)
    sst = sum((value - mean) ** 2 for value in actual)
    return dict(
        n=len(actual),
        rmse=math.sqrt(sse / len(actual)),
        mae=sum(abs(value) for value in errors) / len(actual),
        r2=1 - sse / sst if sst else (1.0 if not sse else 0.0),
        bias=sum(errors) / len(actual),
    )


def load_pipeline(spark, path):
    from pyspark.ml import PipelineModel

    # JVM stages can be restored directly, including on Windows/Python 3.12.
    return PipelineModel._from_java(
        spark._jvm.org.apache.spark.ml.PipelineModel.load(spark_path(path))
    )
