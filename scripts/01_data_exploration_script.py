# %% [markdown]
# # Exploring dataset

# %%
from pyspark.sql import SparkSession

# %%
spark = (
    SparkSession.builder
    .appName("Spotify Exploration")
    .master("local[*]")
    .getOrCreate()
)

# %%
df = (
    spark.read
    .option("header", True)
    .option("inferSchema", True)
    .option("quote", '"')
    .option("escape", '"')
    .option("multiLine", True)
    .option("mode", "PERMISSIVE")
    .csv("../data/raw/spotify_tracks.csv")
)

# %%
df.show(5)

# %%
df.show(5, truncate=False)

# %%
df.printSchema()

# %%
print("Rows:", df.count())

# %%
print("Columns:", len(df.columns))

# %%
print(df.columns)

# %%
from pyspark.sql.functions import col

# %%
df.filter(col("explicit") != "False").select(
    "track_id",
    "artists",
    "track_name",
    "popularity",
    "duration_ms",
    "explicit"
).show(20, truncate=False)

# %%
df.select("explicit").distinct().show(20, truncate=False)

# %%
# first trial i had a problem with the column "explicit", so i restarted the kernel and redid everything, which is why there is so much extra code, i will keep it for now, just in case i need to go back to it.

# %% [markdown]
# #### Eliminating unnecessary column

# %%
clean_df = df.drop("_c0")

# %%

raw_df_test = (
    spark.read
    .option("header", True)
    .option("inferSchema", True)
    .option("quote", '"')
    .option("escape", '"')
    .option("multiLine", True)
    .option("mode", "PERMISSIVE")
    .csv("../data/raw/spotify_tracks.csv")
)
clean_df = raw_df_test
raw_df_test.printSchema()

raw_df_test.select(
    "track_name",
    "artists",
    "popularity",
    "danceability",
    "energy",
    "explicit"
).show(10, truncate=False)

# %% [markdown]
# ### Examining + eliminating exact duplicates 

# %%
from pyspark.sql.functions import col, sum

null_counts = clean_df.select(
    [
        sum(col(c).isNull().cast("int")).alias(c)
        for c in clean_df.columns
    ]
)

null_counts.show()

# %%
clean_df.filter(
    col("artists").isNull() |
    col("album_name").isNull() |
    col("track_name").isNull()
).show(truncate=False)

# %%
print("Rows after removing missing values:", clean_df.count())

# %%
clean_df.filter(
    col("artists").isNull() |
    col("album_name").isNull() |
    col("track_name").isNull()
).count()

# %%
# drops the row where artist, album_name, and track_name are all null, but keeps the rows where at least one of them is not null
clean_df = clean_df.filter(
    ~(
        col("artists").isNull() &
        col("album_name").isNull() &
        col("track_name").isNull()
    )
)

# %%
print("Total rows:", clean_df.count())
print("Unique track IDs:", clean_df.select("track_id").distinct().count())

# %%
print(
    "Duplicate rows:",
    clean_df.count() - clean_df.dropDuplicates().count()
)

# %%
from pyspark.sql.functions import count

duplicate_rows = (
    clean_df
    .groupBy(clean_df.columns)
    .count()
    .filter(col("count") > 1)
)

duplicate_rows.show(10, truncate=False)

# %%
print("Duplicated groups:", duplicate_rows.count())

# %%
print(
    "Rows involved in duplicates:",
    duplicate_rows.select(
        sum(col("count"))
    ).collect()[0][0]
)

# %%
clean_df = clean_df.dropDuplicates()

# %%
print("Rows after removing exact duplicates:", clean_df.count())

# %%
print("Unique track IDs:", clean_df.select("track_id").distinct().count())

# %%
track_counts = (
    clean_df
    .groupBy("track_id")
    .count()
    .orderBy(col("count").desc())
)

track_counts.show(20)

# %%
print(
    "Tracks appearing more than once:",
    track_counts.filter(col("count") > 1).count()
)

# %%
repeated_tracks = (
    track_counts
    .filter(col("count") > 1)
    .select("track_id")
)

clean_df.join(
    repeated_tracks,
    on="track_id",
    how="inner"
).select(
    "track_id",
    "track_name",
    "artists",
    "track_genre"
).orderBy("track_id").show(30, truncate=False)

# %% [markdown]
# ### making sure all are valid values

# %%
features_0_1 = [
    "danceability",
    "energy",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence"
]

for feature in features_0_1:
    invalid = clean_df.filter(
        (col(feature) < 0) | (col(feature) > 1)
    ).count()

    print(f"{feature}: {invalid} invalid values")

# %%
print(
    "Invalid popularity:",
    clean_df.filter(
        (col("popularity") < 0) | (col("popularity") > 100)
    ).count()
)

# %%
print(
    "Invalid duration:",
    clean_df.filter(col("duration_ms") <= 0).count()
)

# %% [markdown]
# ### Understanding dataset

# %%
clean_df.select("popularity").describe().show()

# %%
from pyspark.sql.functions import when

popularity_distribution = (
    clean_df
    .withColumn(
        "popularity_range",
        when(col("popularity") < 10, "0-9")
        .when(col("popularity") < 20, "10-19")
        .when(col("popularity") < 30, "20-29")
        .when(col("popularity") < 40, "30-39")
        .when(col("popularity") < 50, "40-49")
        .when(col("popularity") < 60, "50-59")
        .when(col("popularity") < 70, "60-69")
        .when(col("popularity") < 80, "70-79")
        .when(col("popularity") < 90, "80-89")
        .otherwise("90-100")
    )
    .groupBy("popularity_range")
    .count()
    .orderBy("popularity_range")
)

popularity_distribution.show()

# %%
median_popularity = clean_df.approxQuantile(
    "popularity",
    [0.5],
    0.01
)[0]

print("Median popularity:", median_popularity)

# %%
percentiles = clean_df.approxQuantile(
    "popularity",
    [0.50, 0.75, 0.80, 0.90, 0.95],
    0.01
)

for p, value in zip(
    [50, 75, 80, 90, 95],
    percentiles
):
    print(f"{p}th percentile: {value}")

    # help decide what a hit song is bc there are currently waaaaayyy more unpopular songs than popular ones. Only 1,199 songs have popularity ≥80. And only 98 have ≥90. That means if we define a hit as popularity >= 80, we'd have a very imbalanced ML problem.

# %%
# hit distribution
from pyspark.sql.functions import when, count, col

ml_df = clean_df.withColumn(
    "hit",
    when(col("popularity") >= 53, 1).otherwise(0)
)

ml_df.groupBy("hit").count().show()

# %%
# percentages of hit and non-hit songs
total = ml_df.count()

ml_df.groupBy("hit").count().withColumn(
    "percentage",
    col("count") / total * 100
).show()

# %%
clean_df.select(
    "track_name",
    "artists",
    "popularity",
    "track_genre"
).orderBy(
    col("popularity").desc()
).show(20, truncate=False)

# %%
from pyspark.sql.functions import avg, count

genre_popularity = (
    clean_df
    .groupBy("track_genre")
    .agg(
        avg("popularity").alias("average_popularity"),
        count("*").alias("track_count")
    )
    .orderBy(
        col("average_popularity").desc()
    )
)

genre_popularity.show(20, truncate=False)

# %%
from pyspark.sql.functions import avg, count, col

genre_analysis = (
    ml_df
    .groupBy("track_genre")
    .agg(
        avg("popularity").alias("average_popularity"),
        count("*").alias("track_count"),
        avg("hit").alias("hit_rate")
    )
    .withColumn(
        "hit_rate_percent",
        col("hit_rate") * 100
    )
    .orderBy(
        col("hit_rate_percent").desc()
    )
)

genre_analysis.show(20, truncate=False)

# %%
from pyspark.sql.functions import count

clean_df.select(
    count("*").alias("total_rows"),
    count("popularity").alias("popularity_non_null")
).show()

# make sure all rows have non-null popularity values and that values are correct 

# %%
clean_df.count()

# %% [markdown]
# ### Data Cleaning

# %%
# Check for null or empty values in string columns
from pyspark.sql.functions import (
    col, trim, when, sum, count, avg, min, max,
    percentile_approx, first, collect_set
)

string_columns = [
    "track_id",
    "artists",
    "album_name",
    "track_name",
    "track_genre"
]

for c in string_columns:
    empty_count = clean_df.filter(
        col(c).isNull() | (trim(col(c)) == "")
    ).count()

    print(f"{c}: {empty_count} null/empty values")

# %%
# Clean string columns by trimming whitespace
for c in string_columns:
    clean_df = clean_df.withColumn(
        c,
        trim(col(c))
    )

# %%
# Check for null or empty values in string columns after trimming
missing_track_id = clean_df.filter(
    col("track_id").isNull() | (col("track_id") == "")
).count()

print("Rows without track_id:", missing_track_id)

# %%
# Remove rows with missing track IDs
clean_df = clean_df.filter(
    col("track_id").isNotNull() &
    (col("track_id") != "")
)

print("Rows after removing missing track IDs:", clean_df.count())

# %%
# Check for null values in all columns
null_counts = clean_df.select(
    [
        sum(col(c).isNull().cast("int")).alias(c)
        for c in clean_df.columns
    ]
)

null_counts.show(truncate=False)

# %%
# Check for distinct values in categorical columns
for c in ["key", "mode", "time_signature"]:
    print(f"\nValues in {c}:")
    clean_df.select(c).distinct().orderBy(c).show(20)

# %%
# Check for invalid values in numerical columns
print(
    "Invalid key:",
    clean_df.filter(
        col("key").isNull() |
        (col("key") < -1) |
        (col("key") > 11)
    ).count()
)

print(
    "Invalid mode:",
    clean_df.filter(
        col("mode").isNull() |
        (~col("mode").isin(0, 1))
    ).count()
)

print(
    "Invalid time_signature:",
    clean_df.filter(
        col("time_signature").isNull() |
        (col("time_signature") < 1)
    ).count()
)

# %%
clean_df.groupBy("time_signature") \
    .count() \
    .orderBy("time_signature") \
    .show()

# %%
# Check for rows with time_signature equal to 0 because it could b incorrect data. 
clean_df.filter(
    col("time_signature") == 0
).select(
    "track_name",
    "artists",
    "time_signature"
).show(20, truncate=False)

# %%
# Check for invalid tempo values
clean_df.filter(
    (col("tempo") <= 0) | (col("tempo") > 250)
).select(
    "track_id",
    "track_name",
    "tempo"
).show(20, truncate=False)

# %%
# Check for invalid loudness values
clean_df.select(
    "tempo",
    "loudness",
    "duration_ms"
).describe().show()

# %%
# Check for tracks with unusually long durations bc 87.3 minutes is long
clean_df.orderBy(
    col("duration_ms").desc()
).select(
    "track_id",
    "track_name",
    "artists",
    "duration_ms"
).show(20, truncate=False)

# %%
# Check for invalid tempo values
print(
    "Invalid tempo:",
    clean_df.filter(
        col("tempo").isNull() |
        (col("tempo") <= 0)
    ).count()
)

# %%
# Check for missing loudness values
print(
    "Missing loudness:",
    clean_df.filter(
        col("loudness").isNull()
    ).count()
)

# %% [markdown]
# ### Final validation of clean_df

# %%
# Check for invalid values in features again, that should be between 0 and 1
features_0_1 = [
    "danceability",
    "energy",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence"
]

for feature in features_0_1:
    invalid = clean_df.filter(
        col(feature).isNull() |
        (col(feature) < 0) |
        (col(feature) > 1)
    ).count()

    print(f"{feature}: {invalid} invalid values")

# %%
# Check for invalid popularity values again
print(
    "Invalid popularity:",
    clean_df.filter(
        col("popularity").isNull() |
        (col("popularity") < 0) |
        (col("popularity") > 100)
    ).count()
)

# %%
# Check for invalid duration values again
print(
    "Invalid duration:",
    clean_df.filter(
        col("duration_ms").isNull() |
        (col("duration_ms") <= 0)
    ).count()
)

# %% [markdown]
# #### FINAL DATASET SUMMARY CLEAN_DF

# %%
print("FINAL CLEAN DATASET")
print("-------------------")

print("Total rows:", clean_df.count())

print(
    "Unique track IDs:",
    clean_df.select("track_id").distinct().count()
)

print(
    "Exact duplicate rows:",
    clean_df.count() - clean_df.dropDuplicates().count()
)

print("Number of columns:", len(clean_df.columns))

# %% [markdown]
# ## Create analytics_df

# %%
analytics_df = clean_df

# %%
print("Analytics rows:", analytics_df.count())

print(
    "Unique tracks:",
    analytics_df.select("track_id").distinct().count()
)

# %% [markdown]
# ## Create ml_df

# %%
# ML dataset shpuld contain one observation per track.
# clean_df = 113,550 rows but only 89,740 unique tracks

# %%
#identidy tracks with multiple genres, which could be a problem for ML models that expect one label per observation.
from pyspark.sql.functions import countDistinct

track_genre_counts = (
    clean_df
    .groupBy("track_id")
    .agg(
        countDistinct("track_genre").alias("genre_count")
    )
)

track_genre_counts.filter(
    col("genre_count") > 1
).orderBy(
    col("genre_count").desc()
).show(20)

# %%
# check if repeated tracks have different popularity values, which could be a problem for ML models that expect one label per observation.
popularity_consistency = (
    clean_df
    .groupBy("track_id")
    .agg(
        min("popularity").alias("min_popularity"),
        max("popularity").alias("max_popularity")
    )
)

popularity_consistency.filter(
    col("min_popularity") != col("max_popularity")
).show(20)

# %%
# number of tracks with different popularity values
print(
    "Tracks with different popularity values:",
    popularity_consistency.filter(
        col("min_popularity") != col("max_popularity")
    ).count()
)

# %%
# Check consistency of audio features
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

consistency_checks = []

for feature in audio_features:
    inconsistent = (
        clean_df
        .groupBy("track_id")
        .agg(
            min(feature).alias("min_value"),
            max(feature).alias("max_value")
        )
        .filter(
            col("min_value") != col("max_value")
        )
        .count()
    )

    consistency_checks.append((feature, inconsistent))

for feature, inconsistent in consistency_checks:
    print(f"{feature}: {inconsistent} inconsistent tracks")

# %%
# Build ML dataset with one observation per track, aggregating audio features and popularity values.
from pyspark.sql.functions import max as spark_max

ml_df = (
    clean_df
    .groupBy("track_id")
    .agg(
        avg("popularity").alias("popularity"),

        *[
            avg(c).alias(c)
            for c in audio_features
        ],

        spark_max("explicit").alias("explicit")
    )
)

# %%
# check result of creating the ML dataset
print("ML rows:", ml_df.count())

print(
    "Unique track IDs:",
    ml_df.select("track_id").distinct().count()
)

print(
    "Duplicate track IDs:",
    ml_df.count() - ml_df.select("track_id").distinct().count()
)

# %%
from pyspark.sql.functions import when

popularity_distribution_ml = (
    ml_df
    .withColumn(
        "popularity_range",
        when(col("popularity") < 10, "0-9")
        .when(col("popularity") < 20, "10-19")
        .when(col("popularity") < 30, "20-29")
        .when(col("popularity") < 40, "30-39")
        .when(col("popularity") < 50, "40-49")
        .when(col("popularity") < 60, "50-59")
        .when(col("popularity") < 70, "60-69")
        .when(col("popularity") < 80, "70-79")
        .when(col("popularity") < 90, "80-89")
        .otherwise("90-100")
    )
    .groupBy("popularity_range")
    .count()
    .orderBy("popularity_range")
)

popularity_distribution_ml.show()

# %%
# calculating hit percentiles on ml_df bc there is 1 per track and no repetition

percentiles_ml = ml_df.approxQuantile(
    "popularity",
    [0.50, 0.75, 0.80, 0.90, 0.95],
    0.01
)

for p, value in zip(
    [50, 75, 80, 90, 95],
    percentiles_ml
):
    print(f"{p}th percentile: {value}")

    # help decide what a hit song is bc there are currently waaaaayyy more unpopular songs than popular ones. Only 585 songs have popularity ≥80. And only 42 have ≥90. That means if we define a hit as popularity >= 80, we'd have a very imbalanced ML problem.

# %%
# hit distribution
from pyspark.sql.functions import when, count, col

ml_df = ml_df.withColumn(
    "hit",
    when(col("popularity") >= 52, 1).otherwise(0)
)

ml_df.groupBy("hit").count().show()

# %%
# percentages of hit and non-hit songs
total = ml_df.count()

ml_df.groupBy("hit").count().withColumn(
    "percentage",
    col("count") / total * 100
).show()

# %%
# checking class distribution again ml dataser 
total_tracks = ml_df.count()

hit_distribution = (
    ml_df
    .groupBy("hit")
    .count()
    .withColumn(
        "percentage",
        col("count") / total_tracks * 100
    )
    .orderBy("hit")
)

hit_distribution.show()

# %% [markdown]
# ### Save datasets

# %%
processed_path = "../data/processed"

# %%
print(spark.sparkContext)

# %%
import os

# Crear la carpeta processed si no existe
os.makedirs(processed_path, exist_ok=True)

# Convertir a Pandas y guardar como CSV
clean_pd = clean_df.toPandas()
clean_pd.to_csv(
    os.path.join(processed_path, "spotify_tracks_clean.csv"),
    index=False
)

# %%
#analytics_df
# Convertir a Pandas y guardar como CSV
analytics_pd = analytics_df.toPandas()
analytics_pd.to_csv(
    os.path.join(processed_path, "spotify_tracks_analytics.csv"),
    index=False
)

# %%
#ml_df
# Convertir a Pandas y guardar como CSV
ml_pd = ml_df.toPandas()
ml_pd.to_csv(
    os.path.join(processed_path, "spotify_tracks_ml.csv"),
    index=False
)

# %%



