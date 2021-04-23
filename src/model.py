import pandas as pd

from src.constants import MODEL_NUMERICAL_FEATURES, MODEL_CATEGORICAL_FEATURES


class ShapObject:
    def __init__(self, base_values, data, values, feature_names):
        self.base_values = base_values  # Single value
        self.data = data  # Raw feature values for 1 row of data
        self.values = values  # SHAP values for the same row of data
        self.feature_names = feature_names  # Column names


def create_holdout(frame, holdout_fraction, default_columns=True):
    if default_columns:
        target_frame = frame[
            ["success_streaming_ratio_users", "playlist_uri"]
            + MODEL_NUMERICAL_FEATURES
            + MODEL_CATEGORICAL_FEATURES
        ]
    else:
        target_frame = frame

    success_frame = target_frame.loc[lambda f: f["success_streaming_ratio_users"] == 1]
    non_success_frame = target_frame.loc[
        lambda f: f["success_streaming_ratio_users"] == 0
    ]

    holdout_size = int(holdout_fraction * len(success_frame))

    holdout_frame_success = success_frame.sample(
        n=holdout_size, replace=False, random_state=69
    )
    holdout_frame_non_success = non_success_frame.sample(
        n=holdout_size, replace=False, random_state=69
    )

    train_frame_success = success_frame.loc[
        lambda f: ~f["playlist_uri"].isin(holdout_frame_success["playlist_uri"])
    ]
    train_frame_non_success = non_success_frame.loc[
        lambda f: ~f["playlist_uri"].isin(holdout_frame_non_success["playlist_uri"])
    ]

    holdout_frame = pd.concat(
        [holdout_frame_success, holdout_frame_non_success]
    ).reset_index(drop=True)

    train_frame = pd.concat([train_frame_success, train_frame_non_success]).reset_index(
        drop=True
    )
    return train_frame, holdout_frame
