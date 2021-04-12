import numpy as np
import logging
from copy import deepcopy

logging.basicConfig(level=logging.INFO)


def create_features(raw_frame):
    """
    Derive features from the orginal ones provided in the spotify playlist success data

    :params raw_frame: original dataframe from the case provided file provided "playlist_summary_external.txt"
    """
    features_frame = (
        raw_frame.assign(monhtly_skips=lambda f: (f["streams"] - f["stream30s"]) * 30)
        .assign(tracks_per_album=lambda f: f["n_tracks"] / f["n_albums"])
        .assign(artists_per_album=lambda f: f["n_artists"] / f["n_albums"])
        .assign(
            owner_stream=lambda f: np.where(f["monthly_owner_stream30s"] == 0, 0, 1)
        )
        .assign(
            mau_adjusted=lambda f: np.where(
                f["owner_stream"] == 1, f["mau"] - 1, f["mau"]
            )
        )
        .assign(
            users_adjusted=lambda f: np.where(
                f["owner_stream"] == 1, f["users"] - 1, f["users"]
            )
        )
        .assign(
            monhtly_non_owner_stream30s=lambda f: f["monthly_stream30s"]
            - f["monthly_owner_stream30s"]
        )
        .assign(
            streaming_ratio_mau=lambda f: f["monhtly_non_owner_stream30s"]
            / f["mau_adjusted"]
        )
        .assign(
            streaming_ratio_users=lambda f: f["monhtly_non_owner_stream30s"]
            / f["users_adjusted"]
        )
        .assign(skip_ratio_users=lambda f: f["monhtly_skips"] / f["users"])
        .assign(mau_perc=lambda f: f["mau"] / f["users"])
        .assign(mau_new=lambda f: f["mau"] - f["mau_previous_month"])
        .assign(
            mau_new_perc=lambda f: np.where(
                f["mau_previous_month"] == 0,
                0,
                f["mau_new"] / f["mau_previous_month"] * 100,
            )
        )
    )
    # How many tokens in each playlist title?
    count_tokens = []
    for token in list(features_frame["tokens"]):
        count_tokens.append(len(eval(token)))

    features_frame["title_length"] = count_tokens

    # Extracting user_id and playlist_id
    list_user = []
    list_playlist = []
    for playlist_uri in features_frame["playlist_uri"]:
        tokens = playlist_uri.split(":")
        list_user.append(tokens[2])
        list_playlist.append(tokens[4])

    features_frame["user_id"] = list_user
    features_frame["playlist_id"] = list_playlist

    return features_frame.reset_index(drop=True)


def classify_success(feature_frame, users_threshold=10, success_threshold=0.75):
    """
    Label playlists as successful based on if their streaming ratio is above a certain threshold

    """
    assert (
        users_threshold >= 10,
        "Acoustic features from Spotify API only obtained for playlists with more than 10 "
        "monthly users",
    )

    # Filtering out playlists with an outlier number of tracks
    n_tracks_upper_quantile = feature_frame["n_tracks"].quantile(0.75)
    n_tracks_lower_quantile = feature_frame["n_tracks"].quantile(0.25)
    iqr = n_tracks_upper_quantile - n_tracks_lower_quantile

    upper_track_limit = n_tracks_upper_quantile + (1.5 * iqr)
    lower_track_limit = n_tracks_lower_quantile - (1.5 * iqr)

    target_frame = (
        feature_frame.loc[lambda f: f["n_tracks"] <= upper_track_limit]
        .loc[lambda f: f["n_tracks"] >= lower_track_limit]
        .loc[lambda f: f["users_adjusted"] > users_threshold]
    ).reset_index(drop=True)

    num_playlists_all = len(feature_frame)
    num_playlists_thresh = len(target_frame)
    logging.info(f"# of playlists: {num_playlists_all}")
    logging.info(f"# of playlists above the users_threshold: {num_playlists_thresh}")
    logging.info(f"% of playlists removed: {num_playlists_all - num_playlists_thresh}")
    logging.info(
        f"% of playlists remaining: {round(num_playlists_thresh / num_playlists_all * 100, 1)}"
    )

    threshold_frame_plays = target_frame.groupby("genre_1").quantile(
        q=success_threshold
    )[["streaming_ratio_users"]]
    threshold_frame_plays.columns = [
        str(col) + "_thresh" for col in threshold_frame_plays.columns
    ]

    success_frame = (
        target_frame.merge(
            threshold_frame_plays.reset_index()[
                [
                    "genre_1",
                    "streaming_ratio_users_thresh",
                ]
            ],
            on="genre_1",
            how="left",
        )
        .assign(
            success_streaming_ratio_users=lambda f: np.where(
                f["streaming_ratio_users"] >= f["streaming_ratio_users_thresh"], 1, 0
            )
        )
    )
    return success_frame


def add_suffixes(frame, suffix):
    renamed_frame = deepcopy(frame)
    renamed_frame.columns = [
        str(col) + suffix if col not in ["track_id", "user_id", "playlist_id"] else col
        for col in renamed_frame.columns
    ]
    return renamed_frame
