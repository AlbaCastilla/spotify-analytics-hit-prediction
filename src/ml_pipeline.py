# %% [markdown]
# # Spotify Hit Prediction
# 
# ## Objective
# 
# Can we predict whether a Spotify track belongs to the top 20% of popularity using its audio characteristics and explicit-content metadata?
# 
# We will use PySpark and Spark MLlib to train and compare Logistic Regression, Random Forest and Gradient-Boosted Trees classifiers.
# 
# The test set will remain separate from model training and preprocessing.

# %%
# import and spark session
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = (
    SparkSession.builder
    .appName("Spotify Hit Prediction")
    .master("local[*]")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

# %%
#load data

data_path = "../data/processed/spotify_tracks_ml.csv"

clean_df = spark.read.csv(data_path, header=True, inferSchema=True)

print("Rows:", clean_df.count())
print("Columns:", len(clean_df.columns))

clean_df.printSchema()
clean_df.show(5, truncate=False)

# %%
print("Unique track IDs:",
      clean_df.select("track_id").distinct().count())

print("Missing track IDs:",
      clean_df.filter(
          F.col("track_id").isNull() |
          (F.trim(F.col("track_id")) == "")
      ).count())

print("Missing popularity:",
      clean_df.filter(
          F.col("popularity").isNull()
      ).count())

# %%
audio_features = [
    "danceability",
    "energy",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "loudness",
    "duration_ms",
    "key",
    "mode",
    "time_signature"
]

feature_columns = audio_features + ["explicit"]

# %%
#even though already cleaned add code just in case

# %%
#clean the data to remove rows with missing track_id or popularity
model_source = clean_df.filter(
    F.col("track_id").isNotNull() &
    (F.trim(F.col("track_id")) != "") &
    F.col("popularity").isNotNull()
)

# %%
# aggregate 1 row per track_id, averaging popularity and audio features, and taking max of explicit
track_df = (
    model_source
    .groupBy("track_id")
    .agg(
        F.avg("popularity").alias("popularity"),

        *[
            F.avg(c).alias(c)
            for c in audio_features
        ],

        F.max(
            F.col("explicit").cast("int")
        ).alias("explicit")
    )
)

print("Rows:", track_df.count())
print("Unique tracks:",
      track_df.select("track_id").distinct().count())

track_df.show(5, truncate=False)

# %%
#check for duplicate track IDs just in case, although already filtered when db created
duplicate_track_ids = (
    track_df
    .groupBy("track_id")
    .count()
    .filter(F.col("count") > 1)
    .count()
)

print("Duplicate track IDs:", duplicate_track_ids)

track_df.select("popularity").describe().show()

# %% [markdown]
# # SPLIT DATA BEFORE TESTING 
# 
# 80% of tracks will be used for training 
# 20% of tracks will be used for testing

# %%
train_raw, test_raw = track_df.randomSplit(
    [0.8, 0.2],
    seed=42
)

print("Training rows:", train_raw.count())
print("Test rows:", test_raw.count())

# %%
# calculate threshold for 80th percentile of popularity in training set
threshold = train_raw.approxQuantile(
    "popularity",
    [0.80],
    0.001
)[0]

print("Training 80th percentile:", threshold)

# %%
#create target for both datasets using training threshold
train_df = train_raw.withColumn(
    "hit",
    F.when(F.col("popularity") >= threshold, 1)
     .otherwise(0)
)

test_df = test_raw.withColumn(
    "hit",
    F.when(F.col("popularity") >= threshold, 1)
     .otherwise(0)
)

# %%
#check class distribution in training and test sets
print("TRAINING DISTRIBUTION")
train_df.groupBy("hit").count().orderBy("hit").show()

print("TEST DISTRIBUTION")
test_df.groupBy("hit").count().orderBy("hit").show()

# %%
#check for missing values in feature columns, there should be 0 becaue of the cleaning before creating the db
for c in feature_columns:
    missing = train_df.filter(
        F.col(c).isNull() |
        F.isnan(F.col(c).cast("double"))
    ).count()

    print(f"{c}: {missing} missing values")

# %%
# create preprocessing pipeline, just in case although this shouldnt have to be used for feature columns: impute missing values, assemble into vector, and scale
from pyspark.ml import Pipeline
from pyspark.ml.feature import Imputer, VectorAssembler, StandardScaler

imputer = Imputer(
    inputCols=feature_columns,
    outputCols=[f"{c}_imputed" for c in feature_columns],
    strategy="median"
)

assembler = VectorAssembler(
    inputCols=[f"{c}_imputed" for c in feature_columns],
    outputCol="features",
    handleInvalid="error"
)

scaler = StandardScaler(
    inputCol="features",
    outputCol="scaled_features",
    withMean=False,
    withStd=True
)

# %% [markdown]
# # Training 3 models

# %%
# classifiers to be used
from pyspark.ml.classification import (
    LogisticRegression,
    RandomForestClassifier,
    GBTClassifier
)

# %%
#define models with hyperparameters/pipelines for each model
lr = LogisticRegression(
    labelCol="hit",
    featuresCol="scaled_features",
    maxIter=50,
    regParam=0.01
)

rf = RandomForestClassifier(
    labelCol="hit",
    featuresCol="features",
    numTrees=100,
    maxDepth=8,
    seed=42
)

gbt = GBTClassifier(
    labelCol="hit",
    featuresCol="features",
    maxIter=50,
    maxDepth=5,
    seed=42
)

lr_pipeline = Pipeline(
    stages=[imputer, assembler, scaler, lr]
)

rf_pipeline = Pipeline(
    stages=[imputer, assembler, rf]
)

gbt_pipeline = Pipeline(
    stages=[imputer, assembler, gbt]
)

# %%
#fit models to training data

lr_model = lr_pipeline.fit(train_df)

rf_model = rf_pipeline.fit(train_df)

gbt_model = gbt_pipeline.fit(train_df)

# %%
#see predictions on test data
lr_predictions = lr_model.transform(test_df)

rf_predictions = rf_model.transform(test_df)

gbt_predictions = gbt_model.transform(test_df)

# %%
lr_predictions.select(
    "track_id",
    "hit",
    "prediction",
    "probability"
).show(10, truncate=False)

# %%
rf_predictions.select(
    "track_id",
    "hit",
    "prediction",
    "probability"
).show(10, truncate=False)

# %%
gbt_predictions.select(
    "track_id",
    "hit",
    "prediction",
    "probability"
).show(10, truncate=False)

# %% [markdown]
# # Evaluate models
# 
# Accuracy alone can be misleading when only around 20% of tracks are labelled as hits.
# 
# We'll measure:
# 
# Accuracy - Proportion of all predictions that are correct
# Precision - Of the tracks predicted as hits, how many really are hits?
# Recall- Of all actual hits, how many did the model identify?
# F1-score - Balance between precision and recall
# ROC-AUC - How well the model ranks positive cases above negative cases
# PR-AUC -How well it identifies positive cases, especially useful with an imbalanced target

# %%
from pyspark.ml.evaluation import (
    MulticlassClassificationEvaluator,
    BinaryClassificationEvaluator
)

accuracy_evaluator = MulticlassClassificationEvaluator(
    labelCol="hit",
    predictionCol="prediction",
    metricName="accuracy"
)

precision_evaluator = MulticlassClassificationEvaluator(
    labelCol="hit",
    predictionCol="prediction",
    metricName="precisionByLabel",
    metricLabel=1.0
)

recall_evaluator = MulticlassClassificationEvaluator(
    labelCol="hit",
    predictionCol="prediction",
    metricName="recallByLabel",
    metricLabel=1.0
)

f1_evaluator = MulticlassClassificationEvaluator(
    labelCol="hit",
    predictionCol="prediction",
    metricName="fMeasureByLabel",
    metricLabel=1.0,
    beta=1.0
)

roc_evaluator = BinaryClassificationEvaluator(
    labelCol="hit",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderROC"
)

pr_evaluator = BinaryClassificationEvaluator(
    labelCol="hit",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderPR"
)

# %%
# evaluate model 
def evaluate_model(predictions):
    return {
        "accuracy": accuracy_evaluator.evaluate(predictions),
        "precision": precision_evaluator.evaluate(predictions),
        "recall": recall_evaluator.evaluate(predictions),
        "f1": f1_evaluator.evaluate(predictions),
        "roc_auc": roc_evaluator.evaluate(predictions),
        "pr_auc": pr_evaluator.evaluate(predictions)
    }

# %%
#compare the 3 models based on evaluation metrics
results = {
    "Logistic Regression": evaluate_model(lr_predictions),
    "Random Forest": evaluate_model(rf_predictions),
    "Gradient-Boosted Trees": evaluate_model(gbt_predictions)
}

results_df = (
    spark.createDataFrame([
        (model_name, *metrics.values())
        for model_name, metrics in results.items()
    ], [
        "model",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc"
    ])
    .orderBy(F.col("pr_auc").desc())
)

results_df.show(truncate=False)

# %% [markdown]
# # Inspect errors
# 
# hit = 1 | prediction = 1 -> true positive
# hit = 0 | prediction = 0 -> true negative
# hit = 1 | prediction = 0 -> false negative
# hit = 0 | prediction = 1 -> false positive

# %%
# confusion matrix - random forest 
rf_predictions.groupBy(
    "hit",
    "prediction"
).count().orderBy(
    "hit",
    "prediction"
).show()

# %%
# confusion matrix - logistic regression
lr_predictions.groupBy(
    "hit",
    "prediction"
).count().orderBy(
    "hit",
    "prediction"
).show()

# %%
# confusion matrix - gradient boosted trees
gbt_predictions.groupBy(
    "hit",
    "prediction"
).count().orderBy(
    "hit",
    "prediction"
).show()

# %%
#indiv miscassifications for random forest
rf_predictions.filter(
    F.col("hit") != F.col("prediction")
).select(
    "track_id",
    "popularity",
    "hit",
    "prediction",
    "probability"
).show(20, truncate=False)

# %%
#indiv miscassifications for logistic regression
lr_predictions.filter(
    F.col("hit") != F.col("prediction")
).select(
    "track_id",
    "popularity",
    "hit",
    "prediction",
    "probability"
).show(20, truncate=False)

# %%
#indiv miscassifications for gradient boosted trees
gbt_predictions.filter(
    F.col("hit") != F.col("prediction")
).select(
    "track_id",
    "popularity",
    "hit",
    "prediction",
    "probability"
).show(20, truncate=False)

# %% [markdown]
# # Select best model
# 
# PR-AUC as initial selection criterion
# 
# In a more complete project, we'd tune models using cross-validation on the training set and use the test set only for the final comparison.

# %%
best_model_name = max(
    results,
    key=lambda name: results[name]["pr_auc"]
)

print("Best model:", best_model_name)
print("Metrics:", results[best_model_name])

# %% [markdown]
# # saving model

# %%
print(type(gbt_model))
print("Spark version:", spark.version)


