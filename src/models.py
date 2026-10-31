"""Reusable feature transformations, regressors and exact SQL KNN."""

from pyspark.sql import functions as F, Window
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    SQLTransformer,
    StringIndexer,
    OneHotEncoder,
    VectorAssembler,
    StandardScaler,
)
from pyspark.ml.regression import (
    LinearRegression,
    DecisionTreeRegressor,
    RandomForestRegressor,
    GBTRegressor,
)
from pyspark.ml.functions import vector_to_array
from .common import CATEGORIES, NUMERIC, load_pipeline


def baseline_pipeline(full, reg):
    cols = NUMERIC.copy()
    stages = []
    if full:
        for c in CATEGORIES:
            stages.append(
                StringIndexer(inputCol=c, outputCol=c + "_idx", handleInvalid="keep")
            )
        stages.append(
            OneHotEncoder(
                inputCols=[c + "_idx" for c in CATEGORIES],
                outputCols=[c + "_vec" for c in CATEGORIES],
            )
        )
        cols += ["yr", "workingday"] + [c + "_vec" for c in CATEGORIES]
    assert not set(cols) & {"cnt", "casual", "registered", "instant"}
    stages += [
        VectorAssembler(inputCols=cols, outputCol="features"),
        LinearRegression(
            labelCol="cnt",
            maxIter=100,
            regParam=reg,
            elasticNetParam=0.0,
            standardization=True,
        ),
    ]
    return Pipeline(stages=stages)


CATS = ["mnth", "weekday", "weathersit"]
BASE = ["temp", "hum", "windspeed", "yr", "workingday"]


def preprocessing(poly=False, scale=False):
    stages = []
    cols = BASE.copy()
    if poly:
        stages.append(
            SQLTransformer(
                statement="""SELECT *, temp*temp AS temp_sq,
          hum*hum AS hum_sq, temp*hum AS temp_hum, temp*workingday AS temp_work,
          sin(2*pi()*mnth/12) AS month_sin, cos(2*pi()*mnth/12) AS month_cos FROM __THIS__"""
            )
        )
        cols += ["temp_sq", "hum_sq", "temp_hum", "temp_work", "month_sin", "month_cos"]
    for c in CATS:
        stages.append(
            StringIndexer(inputCol=c, outputCol=c + "_idx", handleInvalid="keep")
        )
    stages.append(
        OneHotEncoder(
            inputCols=[c + "_idx" for c in CATS], outputCols=[c + "_vec" for c in CATS]
        )
    )
    cols += [c + "_vec" for c in CATS]
    assert not set(cols) & {"cnt", "casual", "registered", "instant", "row_number"}
    stages.append(
        VectorAssembler(
            inputCols=cols, outputCol="raw_features" if scale else "features"
        )
    )
    if scale:
        stages.append(
            StandardScaler(
                inputCol="raw_features",
                outputCol="features",
                withStd=True,
                withMean=False,
            )
        )
    return stages


def regressor(family, params):
    if family in ["OLS", "Ridge", "Lasso", "ElasticNet", "PolyRidge"]:
        return LinearRegression(
            labelCol="cnt", maxIter=100, standardization=True, **params
        )
    if family == "DecisionTree":
        return DecisionTreeRegressor(
            labelCol="cnt", seed=42, minInstancesPerNode=5, **params
        )
    if family == "RandomForest":
        return RandomForestRegressor(
            labelCol="cnt", seed=42, minInstancesPerNode=5, **params
        )
    if family == "GBT":
        return GBTRegressor(
            labelCol="cnt", seed=42, stepSize=0.1, minInstancesPerNode=5, **params
        )
    raise ValueError(family)


class KNN:
    def __init__(self, prep, train, k):
        self.prep = prep
        self.k = k
        self.points = (
            prep.transform(train)
            .select(
                F.col("row_number").alias("train_id"),
                F.col("cnt").alias("train_label"),
                vector_to_array("features").alias("train_x"),
            )
            .cache()
        )
        self.points.count()

    def transform(self, data):
        q = self.prep.transform(data).select(
            "row_number",
            "dteday",
            "cnt",
            "weathersit",
            "workingday",
            "mnth",
            vector_to_array("features").alias("query_x"),
        )
        pairs = q.crossJoin(F.broadcast(self.points)).filter(
            F.col("row_number") != F.col("train_id")
        )
        # Exact SQL Euclidean distance. Training error is leave-one-out.
        pairs = pairs.withColumn(
            "distance_sq",
            F.aggregate(
                F.zip_with("query_x", "train_x", lambda a, b: (a - b) * (a - b)),
                F.lit(0.0),
                lambda a, b: a + b,
            ),
        )
        w = Window.partitionBy("row_number").orderBy("distance_sq", "train_id")
        return (
            pairs.withColumn("neighbor_rank", F.row_number().over(w))
            .filter(F.col("neighbor_rank") <= self.k)
            .groupBy("row_number", "dteday", "cnt", "weathersit", "workingday", "mnth")
            .agg(F.avg("train_label").alias("prediction"))
        )

    def save(self, path):
        self.prep.write().overwrite().save(path + "/preprocessing")
        self.points.write.mode("overwrite").parquet(path + "/points")


def fit(family, params, train):
    if family == "KNN":
        return KNN(
            Pipeline(stages=preprocessing(scale=True)).fit(train), train, params["k"]
        )
    return Pipeline(
        stages=preprocessing(poly=family == "PolyRidge") + [regressor(family, params)]
    ).fit(train)


def load_family(spark, family, path, params):
    if family != "KNN":
        return load_pipeline(spark, path)
    model = object.__new__(KNN)
    model.k = params["k"]
    model.prep = load_pipeline(spark, path + "/preprocessing")
    model.points = spark.read.parquet(path + "/points")
    return model
