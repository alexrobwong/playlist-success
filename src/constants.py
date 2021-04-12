BASE_FEATURES = [
    "n_tracks",
    "n_artists",
    "n_albums",
    "tracks_per_album",
    "artists_per_album",
    "title_length",
]
PERCENTILE_FEATURES = [
    "popularity_percentile_25p0",
    "acousticness_percentile_25p0",
    "danceability_percentile_25p0",
    "duration_percentile_25p0",
    "energy_percentile_25p0",
    "instrumentalness_percentile_25p0",
    "liveness_percentile_25p0",
    "loudness_percentile_25p0",
    "speechiness_percentile_25p0",
    "valence_percentile_25p0",
    "popularity_percentile_75p0",
    "acousticness_percentile_75p0",
    "danceability_percentile_75p0",
    "duration_percentile_75p0",
    "energy_percentile_75p0",
    "instrumentalness_percentile_75p0",
    "liveness_percentile_75p0",
    "loudness_percentile_75p0",
    "speechiness_percentile_75p0",
    "valence_percentile_75p0",
]
MEAN_FEATURES = [
    "popularity_mean",
    "acousticness_mean",
    "danceability_mean",
    "duration_mean",
    "energy_mean",
    "instrumentalness_mean",
    "liveness_mean",
    "loudness_mean",
    "speechiness_mean",
    "valence_mean",
]
STD_FEATURES = [
    "popularity_std",
    "acousticness_std",
    "danceability_std",
    "duration_std",
    "energy_std",
    "instrumentalness_std",
    "liveness_std",
    "loudness_std",
    "speechiness_std",
    "valence_std",
]
SKEW_FEATURES = [
    "acousticness_skew_unbiased",
    "danceability_skew_unbiased",
    "duration_skew_unbiased",
    "energy_skew_unbiased",
    "instrumentalness_skew_unbiased",
    "liveness_skew_unbiased",
    "loudness_skew_unbiased",
    "popularity_skew_unbiased",
    "speechiness_skew_unbiased",
    "valence_skew_unbiased",
]
KURTOSIS_FEATURES = [
    "acousticness_kurt_unbiased",
    "danceability_kurt_unbiased",
    "duration_kurt_unbiased",
    "energy_kurt_unbiased",
    "instrumentalness_kurt_unbiased",
    "liveness_kurt_unbiased",
    "loudness_kurt_unbiased",
    "popularity_kurt_unbiased",
    "speechiness_kurt_unbiased",
    "valence_kurt_unbiased",
]
MODEL_NUMERICAL_FEATURES = (
    BASE_FEATURES
    + PERCENTILE_FEATURES
    + MEAN_FEATURES
    + STD_FEATURES
)
MODEL_CATEGORICAL_FEATURES = [
    "genre_1",
    "genre_2",
    "genre_3",
    "mood_1",
    "mood_2",
    "mood_3"
]
