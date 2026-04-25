# %%
import kagglehub

dataset_path = kagglehub.dataset_download(
    "undefinenull/million-song-dataset-spotify-lastfm"
)

print(dataset_path)

# %%
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# %%
data_path = Path(r"C:\Users\suman\.cache\kagglehub\datasets\undefinenull\million-song-dataset-spotify-lastfm\versions\1")

songs_data_path = data_path / "Music Info.csv"
users_data_path = data_path / "User Listening History.csv"

print(songs_data_path.exists(), users_data_path.exists())

# %%
os.getcwd()

# %%
from pathlib import Path

data_path = Path(r"C:\Users\suman\.cache\kagglehub\datasets\undefinenull\million-song-dataset-spotify-lastfm\versions\1")

songs_data_path = data_path / "Music Info.csv"
users_data_path = data_path / "User Listening History.csv"

print(songs_data_path.exists(), users_data_path.exists())


# %%
# load the songs data
df_songs = pd.read_csv(songs_data_path)
df_songs.head()
# %%
df_songs.shape

# %%
df_songs.info()
# %%

# drop columns from data

columns_to_drop = ["spotify_preview_url"]

df_songs.drop(columns=columns_to_drop,inplace=True)

df_songs.head()
# %%
df_songs.isna().sum()
# %%
import missingno as msno
# %%
msno.matrix(df_songs)

# %%
# ratio of missing values in data

(
    df_songs
    .isna()
    .mean()
    .sort_values(ascending=False)
    .head(2)
    .mul(100)
)
# %%
# check for duplicates based on name of the song

(
    df_songs
    .assign(name=df_songs['name'].str.lower())
    .duplicated(subset='name')
    .sum()
)
# %%
# rows that are duplicate

(
    df_songs
    .loc[
        df_songs
        .assign(name=df_songs['name'].str.lower())
        .duplicated(subset='name',keep=False)
    ]
    .assign(name=df_songs['name'].str.lower())
    .sort_values("name")
)
# %%

# duplicates in the data based on spotify_id

(
    df_songs
    .duplicated(subset="spotify_id")
    .sum()
)

# %%
# duplicate songs in the data

(
    df_songs
    .duplicated(subset=["spotify_id","year","duration_ms"])
    .sum()
)
# %%

# rows that have duplicate data

(
    df_songs
    .loc[
        df_songs
        .duplicated(subset=["spotify_id","year","duration_ms"],keep=False)
    ]
    .sort_values(["spotify_id","year","duration_ms"])
)
# %%

# drop duplicates

df_songs.drop_duplicates(subset=["spotify_id","year","duration_ms"],inplace=True)
# %%
# check for duplicates

(
    df_songs
    .duplicated(subset=["spotify_id","year","duration_ms"])
    .sum()
)

# %%
# Column Wise Analysis

# %%

# list of columns in data

df_songs.columns
# %%

df_songs.dtypes

# %%
# shape of data

df_songs.shape
# %%
# Categorical Columns

categorical_features = df_songs.select_dtypes(include="object").columns
categorical_features
# %%

def categorical_analysis(df,feature_names,k_artists=15):
    for feature in feature_names:
        print(f"Number of categories in column {feature} are ",df[feature].str.lower().nunique())

        if feature in ["artist","genre"]:
            print(df[feature].value_counts().head(k_artists))

        if feature == "genre":
            print(f"The unique categories in {feature} column are: ", df[feature].dropna().unique())
        print("#" * 75)
# %%

# perform catagorical analysis

categorical_analysis(df_songs,categorical_features)
# %%
## Observations

# The Track ID and Spotify IDs are unique for every row.  
# Song names have repetitions because some songs share the same name but are performed by different artists.  
# The dataset contains songs from approximately **8,317 artists**.  
# There are **15 distinct categories** in the Genre column.
# %%

# countplot of genre

sns.countplot(df_songs,x="genre")
plt.xticks(rotation=90)
plt.show()

# %%

# genre group

genre_group = df_songs.groupby("genre")

genre_group[['genre','tags']].sample(3)
# %%
# song titles in the data that are not in english

(   df_songs
    .loc[
        df_songs
        .loc[:,"name"]
        .str.contains("[^\d\w\s.?!':;-_(){},\.#-&/-]")
    ]
)
# %%
# artists in the data that are not in english

(   df_songs
    .loc[
        df_songs
        .loc[:,"artist"]
        .str.contains("[^\d\w\s.?!':;-_\{\},\.#-+&\/\-\"]")
    ]
)
# %%

df_songs['tags'][0]
# %%

all_tags = []

for tags in df_songs["tags"].dropna().str.replace(" ","").str.split(","):
    all_tags.extend(tags)

print("The number of unique tags are ",len(set(all_tags)))

set(all_tags)
# %%
# unique tags in the data

(
    df_songs
    .loc[:,"tags"]
    .dropna()
    .str.split(",")
    .explode()
    .str.strip()
    .unique()
)
# %%
integer_columns = df_songs.select_dtypes(include="int").columns
integer_columns

df_songs[integer_columns]
# %%
# statistical summary

(
    df_songs
    .loc[:,integer_columns]
    .drop(columns=["duration_ms"])
    .assign(**{
        col: df_songs[col].astype("object")
        for col in integer_columns.drop("duration_ms")
    })
    .describe()
)
# %%

# range of data

(
    df_songs
    .loc[:,integer_columns]
    .assign(duration_minutes=df_songs["duration_ms"].div(1000).div(60))
    .drop(columns=["duration_ms"])
    .agg(["min","max"])
)
# %%

# number of songs per year in data

sns.histplot(df_songs,x="year",bins=df_songs["year"].max() - df_songs["year"].min(),stat="count")
plt.show()
# %%
# most songs from which year(top 5)

(
    df_songs
    .loc[:,"year"]
    .value_counts()
    .head(5)
    .sort_index()
)
# %%
# unique values in the key column

(
    np.sort(df_songs
            .loc[:,"key"]
            .unique())
)
# %%
# percentage of songs wrt to key in the data

(
    df_songs['key']
    .value_counts(normalize=True)
    .mul(100)
    .sort_index()
    .plot(kind='barh',title="Percentage of Songs wrt to Key",xlabel="Percentage")
)
# %%
# countplot for mode

sns.countplot(df_songs,x="mode")
plt.show()




# %%
# unique values for time signature

(
    np.sort(df_songs
            .loc[:,"time_signature"]
            .unique())
)
# %%
# countplot for time signature

sns.countplot(df_songs,x="time_signature")
plt.show()
# %%
(
    df_songs['time_signature']
    .value_counts(normalize=True)
    .mul(100)
)





# %%
# statistical summary of time duration

(
    df_songs
    .loc[:,["duration_ms"]]
    .assign(duration_minutes=df_songs["duration_ms"].div(1000).div(60))
    .drop(columns="duration_ms")



    .describe()
)
# %%
# time duration histogram

time_duration_mins = df_songs["duration_ms"].div(1000).div(60)

sns.histplot(time_duration_mins)
plt.xlabel("Time Duration (mins)")
plt.show()
# %%

# time duration boxplot

sns.boxplot(time_duration_mins)
plt.ylabel("Time Duration (mins)")
plt.show()
# %%

# song that is longer than 60 mins

(
    df_songs
    .loc[time_duration_mins > 60]
)

# %%
continuous_columns = df_songs.select_dtypes(include="float").columns
continuous_columns

# %%
def numerical_analysis(df,columns):
    for column in columns:
        print(f"Numerical Analysis for column {column}")
        print("Statistical Summary")
        print(df[column].describe())

        fig = plt.figure(figsize=(12,4))
        # hitogram for column
        plt.subplot(1,2,1)
        sns.histplot(df[column])
        plt.title(f"Histogram for {column}")
        # boxplot for column
        plt.subplot(1,2,2)
        sns.boxplot(df[column])
        plt.title(f"Boxplot for {column}")
        plt.show()

        print("#" * 120)
    print("*" * 120)
    print("Pairplot")
    sns.pairplot(df[columns])
    plt.show()
# %%
numerical_analysis(df_songs,continuous_columns)
# %%







# load the dataset

df_users = pd.read_csv(users_data_path)

df_users.head()
#
# %%
# dataset info

df_users.info()
# %%
# check for duplicates

df_users.duplicated(subset=["track_id","user_id"]).sum()

# %%

# check for missing values

df_users.isna().sum()
# %%
# unqiue users in the data

(
    df_users
    .loc[:,"user_id"]
    .nunique()
)
# %%
# unique songs in the data

(
    df_users
    .loc[:,"track_id"]
    .nunique()
)
# %%
# unique songs in the data

(
    df_users
    .loc[:,"track_id"]
    .nunique()
)
# %%
# top 10 most played songs in user data

(
    df_users
    .loc[:,"track_id"]
    .value_counts()
    .head(10)
)

# %%
top_10_songs = (
    df_users
    .loc[:,"track_id"]
    .value_counts()
    .head(10)
)

top_10_songs
# %%
(
    df_songs
    .loc[df_songs["track_id"].isin(top_10_songs.index.tolist()),:]
)
# %%
# most playcounts for songs

top_10_played_songs = (
    df_users.groupby("track_id")['playcount']
    .agg("sum")
    .sort_values(ascending=False)
    .head(10)
)

top_10_played_songs
# %%

(
    df_songs
    .loc[df_songs["track_id"].isin(top_10_played_songs.index.tolist()),:]
)
# %%
pd.concat([top_10_songs,top_10_played_songs],axis=1)
# %%

# most diverse users
# top 10

most_diverse_users = (
                        df_users.groupby("user_id")['track_id']
                        .agg("count")
                        .sort_values(ascending=False)
                        .head(10)
                    )

most_diverse_users

# %%
# most playcounts for users
# top 10

most_active_users = (
                        df_users.groupby("user_id")['playcount']
                        .agg("sum")
                        .sort_values(ascending=False)
                        .head(10)
                    )

most_active_users
# %%

pd.concat([most_diverse_users,most_active_users],axis=1)
# %%
import sys
print(sys.executable)
# %%
