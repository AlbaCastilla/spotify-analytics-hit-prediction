# %% [markdown]
# # Questions
# 
# Q1:	Which genres have the highest average popularity?
# Q2:	Which genres are the most danceable?
# Q3:	Which genres are the most energetic?
# Q4:	Does danceability correlate with popularity?
# Q5:	Does energy correlate with popularity?
# Q6:	What characteristics are associated with highly popular tracks?
# Q7:	How do popular and less-popular songs differ?
# Q8:	Are explicit songs different from non-explicit songs?
# Q9:	Which artists are the most popular?
# Q10: Which tracks are the most popular?
# 
# 
# PySpark for data loading, aggregation and analysis.
# 
# Preserve genre-track associations for genre-level questions.
# 
# Use one row per track for track-level comparisons.
# 
# Use Pandas and Matplotlib/Seaborn to visualise aggregated results.
# 
# Interpret correlations as associations, not evidence of causation.

# %%
# Imports

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

# %%

spark = (
    SparkSession.builder
    .appName("Spotify Analytics")
    .master("local[*]")
    .getOrCreate()
)

analytics_path = "../data/processed/spotify_tracks_analytics.csv"
clean_path = "../data/processed/spotify_tracks_clean.csv"

def read_spotify_csv(path):
    return (
        spark.read
        .option("header", True)
        .option("inferSchema", True)
        .option("quote", '"')
        .option("escape", '"')
        .option("multiLine", True)
        .option("mode", "PERMISSIVE")
        .csv(path)
    )

analytics_df = read_spotify_csv(analytics_path)
clean_df = read_spotify_csv(clean_path)

print("Analytics rows:", analytics_df.count())
print("Unique tracks:", analytics_df.select("track_id").distinct().count())

analytics_df.printSchema()
clean_df.printSchema()


# %%

clean_df.select(
    "track_id",
    "track_name",
    "artists",
    "popularity",
    "duration_ms",
    "explicit",
    "danceability",
    "energy",
    "track_genre"
).show(10, truncate=False)


# %%
# track_df has one row per track_id dont want the same track appearing multiple times bc of different genres or something
track_features = [
    "popularity",
    "duration_ms",
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "key",
    "mode",
    "time_signature"
]

track_df = (
    clean_df
    .groupBy("track_id")
    .agg(
        *[
            F.avg(c).alias(c)
            for c in track_features
        ],
        F.first("track_name", ignorenulls=True).alias("track_name"),
        F.first("artists", ignorenulls=True).alias("artists"),
        F.first("explicit", ignorenulls=True).alias("explicit"),
        F.first("album_name", ignorenulls=True).alias("album_name")
    )
)

print("Track-level rows:", track_df.count())
print("Unique track IDs:", track_df.select("track_id").distinct().count())

# %%
print("=== TRACK_DF ===")
track_df.printSchema()

print("=== ANALYTICS_DF ===")
analytics_df.printSchema()

# %%
from pyspark.sql import functions as F

# Columnas que queremos validarn por si acaso, porque tienen unos rangos de valores muy concretos
columnas = {
    "popularity": (0, 100),
    "danceability": (0, 1),
    "energy": (0, 1)
}

# Convertir las columnas a double de forma segura
df_validado = clean_df

for columna in columnas:
    df_validado = df_validado.withColumn(
        columna,
        F.expr(f"try_cast(`{columna}` AS DOUBLE)")
    )

# Comprobar los rangos permitidos
for columna, (minimo, maximo) in columnas.items():
    print(f"\n--- Validación de {columna} ---")

    # Valores fuera de rango
    fuera_rango = df_validado.filter(
        F.col(columna).isNotNull()
        & (
            (F.col(columna) < minimo)
            | (F.col(columna) > maximo)
        )
    )

    # Valores nulos o que no se pudieron convertir
    nulos = df_validado.filter(F.col(columna).isNull())

    print("Valores fuera de rango:", fuera_rango.count())
    print("Valores nulos o no numéricos:", nulos.count())

    # Mostrar ejemplos incorrectos
    fuera_rango.select(columna).show(10, truncate=False)

# %%

from pyspark.sql import functions as F

# ver Valores nulos por columna
null_counts = analytics_df.select([
    F.sum(F.col(c).isNull().cast("int")).alias(c)
    for c in analytics_df.columns
])

null_counts.show(truncate=False)

# ver nº registros y canciones únicas
print("Total de registros:", analytics_df.count())
print("Canciones únicas:", analytics_df.select("track_id").distinct().count())

# ver número de géneros musicales
print("Géneros únicos:", analytics_df.select("track_genre").distinct().count())


# %%
from pyspark.sql import functions as F

numeric_columns = [
    "popularity",
    "duration_ms",
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "key",
    "mode",
    "time_signature"
]

for column in numeric_columns:
    try:
        track_df.select(F.avg(F.col(column))).collect()
        print(f"OK: {column}")
    except Exception as e:
        print(f"ERROR en {column}: {str(e)[:300]}")

# %%
print("Columnas de clean_df:")
clean_df.printSchema()

print("Columnas de analytics_df:")
analytics_df.printSchema()

print("Columnas de track_df:")
print(track_df.columns)

# %%
print("=== ANALYTICS DF ===")

analytics_df.select(
    "popularity",
    "duration_ms",
    "danceability",
    "energy",
    "tempo"
).describe().show()

# %%
# overview 
print("Number of tracks:", track_df.count())
print("Number of genre-track associations:", analytics_df.count())
print("Number of genres:", analytics_df.select("track_genre").distinct().count())

track_df.select(
    "popularity",
    "duration_ms",
    "danceability",
    "energy",
    "tempo"
).describe().show()

# %% [markdown]
# ##  Q1 - Which genres have the highest average popularity?

# %%

# Contar cuántas veces aparece cada canción dentro de cada génerom para saber si algunas canciones estan duplicadas/triplicadas/...
duplicate_tracks = (
    analytics_df
    .groupBy("track_genre", "track_id")
    .agg(
        F.count("*").alias("num_apariciones"),
        F.first("track_name").alias("track_name"),
        F.first("artists").alias("artists")
    )
    .filter(F.col("num_apariciones") > 1)
    .orderBy(
        F.desc("num_apariciones"),
        "track_genre"
    )
)

print("Canciones repetidas dentro del mismo género:")
duplicate_tracks.show(50, truncate=False)


# %%
# av popularity by genre
genre_popularity = (
    analytics_df
    .groupBy("track_genre")
    .agg(
        F.avg("popularity").alias("average_popularity"),
        F.count("*").alias("track_count")
    )
    .orderBy(F.desc("average_popularity"))
)

genre_popularity.show(20, truncate=False)

# %%
# top 15 generes by av. popularity
genre_popularity_pd = genre_popularity.limit(15).toPandas()

plt.figure(figsize=(11, 7))
sns.barplot(
    data=genre_popularity_pd,
    x="average_popularity",
    y="track_genre"
)
plt.title("Top 15 Genres by Average Popularity")
plt.xlabel("Average popularity (0–100)")
plt.ylabel("Genre")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Q2 - Which genres are the most danceable?

# %%
# av dancaeability by genre
genre_danceability = (
    analytics_df
    .groupBy("track_genre")
    .agg(
        F.avg("danceability").alias("average_danceability"),
        F.count("*").alias("track_count")
    )
    .orderBy(F.desc("average_danceability"))
)

genre_danceability.show(15, truncate=False)

# %%
# most danceable genres
danceability_pd = genre_danceability.limit(15).toPandas()

plt.figure(figsize=(11, 7))
sns.barplot(
    data=danceability_pd,
    x="average_danceability",
    y="track_genre"
)
plt.title("Top 15 Genres by Average Danceability")
plt.xlabel("Average danceability")
plt.ylabel("Genre")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Q3 - Which genres are the most energetic?

# %%
# av energy by genre
genre_energy = (
    analytics_df
    .groupBy("track_genre")
    .agg(
        F.avg("energy").alias("average_energy"),
        F.count("*").alias("track_count")
    )
    .orderBy(F.desc("average_energy"))
)

genre_energy.show(15, truncate=False)

# %%
# generes listed by average energy
energy_pd = genre_energy.limit(15).toPandas()

plt.figure(figsize=(11, 7))
sns.barplot(
    data=energy_pd,
    x="average_energy",
    y="track_genre"
)
plt.title("Top 15 Genres by Average Energy")
plt.xlabel("Average energy")
plt.ylabel("Genre")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Q4 - Does danceability correlate with popularity?

# %%
# plot correlation between danceability and popularity. 
# pearson coorrelation coefficient ranges from -1 to 1:
#       1 means perfect positive correlation 
#       -1 means perfect negative correlation,
#       0 means no correlation.
danceability_correlation = track_df.stat.corr(
    "danceability",
    "popularity"
)

print(
    "Correlation between danceability and popularity:",
    round(danceability_correlation, 4)
)

# %%
# scatter plot of danceability vs popularity with regression line
danceability_pd = (
    track_df
    .select("danceability", "popularity")
    .dropna()
    .sample(False, 0.15, seed=42)
    .toPandas()
)

plt.figure(figsize=(10, 6))
sns.regplot(
    data=danceability_pd,
    x="danceability",
    y="popularity",
    scatter_kws={"alpha": 0.2, "s": 15},
    line_kws={"color": "red"}
)
plt.title("Danceability vs Popularity")
plt.xlabel("Danceability")
plt.ylabel("Popularity")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Q5 - Does energy correlate with popularity?

# %%
# coorelation between energy and popularity
energy_correlation = track_df.stat.corr(
    "energy",
    "popularity"
)

print(
    "Correlation between energy and popularity:",
    round(energy_correlation, 4)
)

# %%
# scatter plot of energy vs popularity with regression line
energy_pd = (
    track_df
    .select("energy", "popularity")
    .dropna()
    .sample(False, 0.15, seed=42)
    .toPandas()
)

plt.figure(figsize=(10, 6))
sns.regplot(
    data=energy_pd,
    x="energy",
    y="popularity",
    scatter_kws={"alpha": 0.2, "s": 15},
    line_kws={"color": "red"}
)
plt.title("Energy vs Popularity")
plt.xlabel("Energy")
plt.ylabel("Popularity")
plt.tight_layout()
plt.show()

# %%
# compare coorelations (Danceability vs Energy)
correlation_comparison = pd.DataFrame({
    "Feature": ["Danceability", "Energy"],
    "Correlation with popularity": [
        danceability_correlation,
        energy_correlation
    ]
})

display(correlation_comparison)

# %% [markdown]
# ## Q6 - What characteristics are associated with highly popular tracks?

# %%
# popularity 80th percentile threshold
popularity_threshold = track_df.approxQuantile(
    "popularity",
    [0.80],
    0.001
)[0]

print("80th percentile threshold:", popularity_threshold)

track_df = track_df.withColumn(
    "popularity_group",
    F.when(
        F.col("popularity") >= popularity_threshold,
        "Highly popular"
    ).otherwise("Other tracks")
)

track_df.groupBy("popularity_group").count().show()

# %%
# comparison of average audio features between highly popular tracks and other tracks
comparison_features = [
    "danceability",
    "energy",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "loudness",
    "tempo",
    "duration_ms"
]

feature_comparison = (
    track_df
    .groupBy("popularity_group")
    .agg(
        *[
            F.avg(c).alias(c)
            for c in comparison_features
        ],
        F.count("*").alias("track_count")
    )
)

feature_comparison.show(truncate=False)

# %%
# differences representation
feature_comparison_pd = feature_comparison.toPandas()

feature_comparison_long = feature_comparison_pd.melt(
    id_vars=["popularity_group", "track_count"],
    value_vars=comparison_features,
    var_name="feature",
    value_name="average_value"
)

# Separate plots avoid comparing features on incompatible scales.
for feature in comparison_features:
    plot_data = feature_comparison_long[
        feature_comparison_long["feature"] == feature
    ]

    plt.figure(figsize=(7, 4))
    sns.barplot(
        data=plot_data,
        x="popularity_group",
        y="average_value"
    )
    plt.title(f"Average {feature} by Popularity Group")
    plt.xlabel("")
    plt.ylabel(f"Average {feature}")
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## Q7 - How do popular and less-popular songs differ?

# %%
# creation of top and bottom groups
lower_threshold, upper_threshold = track_df.approxQuantile(
    "popularity",
    [0.20, 0.80],
    0.001
)

print("20th percentile:", lower_threshold)
print("80th percentile:", upper_threshold)

extreme_groups_df = (
    track_df
    .withColumn(
        "popularity_group",
        F.when(
            F.col("popularity") <= lower_threshold,
            "Bottom 20%"
        ).when(
            F.col("popularity") >= upper_threshold,
            "Top 20%"
        )
    )
    .filter(F.col("popularity_group").isNotNull())
)

extreme_groups_df.groupBy("popularity_group").count().show()

# %%
# compare grouos
extreme_comparison = (
    extreme_groups_df
    .groupBy("popularity_group")
    .agg(
        *[
            F.avg(c).alias(c)
            for c in comparison_features
        ],
        F.avg("popularity").alias("average_popularity"),
        F.count("*").alias("track_count")
    )
)

extreme_comparison.show(truncate=False)

# %%
# plot differences between top and bottom groups for each feature
extreme_comparison_pd = extreme_comparison.toPandas()

for feature in comparison_features:
    plt.figure(figsize=(7, 4))

    sns.barplot(
        data=extreme_comparison_pd,
        x="popularity_group",
        y=feature,
        order=["Bottom 20%", "Top 20%"]
    )

    plt.title(f"{feature}: Top vs Bottom Popularity")
    plt.xlabel("")
    plt.ylabel(f"Average {feature}")
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## Q8 - Are explicit songs different from non-explicit songs?

# %%
# comparasion of explicit vs non-explicit tracks
explicit_analysis = (
    track_df
    .filter(F.col("explicit").isNotNull())
    .groupBy("explicit")
    .agg(
        F.count("*").alias("track_count"),
        F.avg("popularity").alias("average_popularity"),
        F.avg("danceability").alias("average_danceability"),
        F.avg("energy").alias("average_energy"),
        F.avg("acousticness").alias("average_acousticness"),
        F.avg("speechiness").alias("average_speechiness"),
        F.avg("loudness").alias("average_loudness"),
        F.avg("duration_ms").alias("average_duration_ms")
    )
    .orderBy("explicit")
)

explicit_analysis.show(truncate=False)

# %%
# plot av. popularity for explicit vs non-explicit tracks
explicit_pd = explicit_analysis.toPandas()

explicit_pd["explicit_label"] = explicit_pd["explicit"].map({
    True: "Explicit",
    False: "Non-explicit",
    1: "Explicit",
    0: "Non-explicit"
})

plt.figure(figsize=(7, 5))
sns.barplot(
    data=explicit_pd,
    x="explicit_label",
    y="average_popularity"
)
plt.title("Average Popularity: Explicit vs Non-explicit Tracks")
plt.xlabel("")
plt.ylabel("Average popularity")
plt.tight_layout()
plt.show()

# %%
# comparation of audio features between explicit and non-explicit tracks
explicit_features_pd = explicit_pd.melt(
    id_vars=["explicit_label"],
    value_vars=[
        "average_danceability",
        "average_energy",
        "average_acousticness",
        "average_speechiness",
        "average_loudness",
        "average_duration_ms"
    ],
    var_name="feature",
    value_name="average_value"
)

for feature in explicit_features_pd["feature"].unique():
    plot_data = explicit_features_pd[
        explicit_features_pd["feature"] == feature
    ]

    plt.figure(figsize=(7, 4))
    sns.barplot(
        data=plot_data,
        x="explicit_label",
        y="average_value"
    )
    plt.title(feature.replace("average_", "").replace("_", " ").title())
    plt.xlabel("")
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## Q9 - Which artists are the most popular?

# %%
# separate artists from the artists column and put them into individual rows
artists_df = (
    track_df
    .filter(
        F.col("artists").isNotNull() &
        (F.trim(F.col("artists")) != "")
    )
    .withColumn(
        "artists_clean",
        F.regexp_replace(
            F.regexp_replace(F.col("artists"), r"[\[\]]", ""),
            r"""['"]""",
            ""
        )
    )
    .withColumn(
        "artist",
        F.explode(F.split(F.col("artists_clean"), ","))
    )
    .withColumn("artist", F.trim(F.col("artist")))
    .filter(F.col("artist") != "")
)

artists_df.select("artist", "track_name", "popularity").show(
    10, truncate=False
)

# %%
# Rank artists by average popularity, considering only those with at least 10 tracks
artist_popularity = (
    artists_df
    .groupBy("artist")
    .agg(
        F.avg("popularity").alias("average_popularity"),
        F.countDistinct("track_id").alias("track_count")
    )
    .filter(F.col("track_count") >= 10)
    .orderBy(F.desc("average_popularity"))
)

# Convert to Pandas DataFrame
artist_popularity_pd = artist_popularity.toPandas()

# Display the top 10 artists
display(artist_popularity_pd.head(10))

# %%
# top 15 artists by av popularity
artist_popularity_pd = artist_popularity.limit(15).toPandas()

plt.figure(figsize=(11, 7))
sns.barplot(
    data=artist_popularity_pd,
    x="average_popularity",
    y="artist"
)
plt.title("Top 15 Artists by Average Track Popularity")
plt.xlabel("Average popularity")
plt.ylabel("Artist")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Q10 - Which tracks are the most popular?

# %%
# top 20 tracks by popularity, with ties included
top_tracks = (
    track_df
    .select(
        "track_id",
        "track_name",
        "artists",
        "album_name",
        "popularity",
        "track_id"
    )
    .orderBy(
        F.desc("popularity"),
        F.asc("track_name")
    )
)

top_tracks.show(20, truncate=False)

# %%

# convert to pandas again
top_tracks_pd = top_tracks.limit(15).toPandas()

# inspect columns to see if there are any duplicates
print("Original columns:", top_tracks_pd.columns.tolist())

# remive duplicates keeping only first occurrence
top_tracks_pd = top_tracks_pd.loc[
    :, ~top_tracks_pd.columns.duplicated()
].copy()

print("Columns after removing duplicates:", top_tracks_pd.columns.tolist())

# chose the columns for the table 
top_tracks_table = top_tracks_pd[
    ["track_name", "artists", "popularity", "track_id"]
].copy()

top_tracks_table = top_tracks_table.rename(columns={
    "track_name": "Track",
    "artists": "Artist",
    "popularity": "Popularity",
    "track_id": "Track ID"
})

# rank, adding a +1 bc array starts in 0
top_tracks_table.insert(0, "Rank", range(1, len(top_tracks_table) + 1))

display(top_tracks_table.style.hide(axis="index"))


# %%
# top 15 tracks by popularity
top_tracks_pd = top_tracks.limit(15).toPandas()

top_tracks_pd["track_label"] = (
    top_tracks_pd["track_name"].fillna("Unknown track")
    + " — "
    + top_tracks_pd["artists"].fillna("Unknown artist")
)

plt.figure(figsize=(12, 8))
sns.barplot(
    data=top_tracks_pd,
    x="popularity",
    y="track_label"
)
plt.title("Top 15 Tracks by Popularity")
plt.xlabel("Popularity")
plt.ylabel("Track — Artist")
plt.tight_layout()
plt.show()


