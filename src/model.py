import numpy as np
import pandas as pd
import xgboost
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score

from src.constants import MODEL_NUMERICAL_FEATURES, MODEL_CATEGORICAL_FEATURES


class PlaylistSuccessPredictor:
    def __init__(self, frame, genre):
        self.genre = genre
        self.genre_frame = frame.loc[lambda f: f["genre_1"] == genre].dropna()
        self.dummies_frame = pd.get_dummies(
            self.genre_frame[
                ["success_streaming_ratio_users", "playlist_uri"]
                + MODEL_NUMERICAL_FEATURES
                + MODEL_CATEGORICAL_FEATURES
            ],
            columns=MODEL_CATEGORICAL_FEATURES,
        )
        self.success_frame = self.dummies_frame.loc[
            lambda f: f["success_streaming_ratio_users"] == 1
        ]
        self.non_success_frame = self.dummies_frame.loc[
            lambda f: f["success_streaming_ratio_users"] == 0
        ]
        self.test_frame = None
        self.train_frame = None
        self.X_train = None
        self.y_train = None
        self.shap_frame = None
        self.X_test = None
        self.y_test = None
        self.model = None
        self.y_pred = None

    def train_model(self, train_size, n_estimators):
        test_size = int((1 - train_size) * len(self.success_frame))

        test_frame_success = self.success_frame.sample(
            n=test_size, replace=False, random_state=69
        )
        test_frame_non_success = self.non_success_frame.sample(
            n=test_size, replace=False, random_state=69
        )

        train_frame_success = self.success_frame.loc[
            lambda f: ~f["playlist_uri"].isin(test_frame_success["playlist_uri"])
        ]
        train_frame_non_success = self.non_success_frame.loc[
            lambda f: ~f["playlist_uri"].isin(test_frame_non_success["playlist_uri"])
        ]

        self.test_frame = pd.concat(
            [test_frame_success, test_frame_non_success]
        ).reset_index(drop=True)
        self.train_frame = pd.concat(
            [train_frame_success, train_frame_non_success]
        ).reset_index(drop=True)

        # Upsample the imbalanced success class until equal number of training observations
        sm = SMOTE(random_state=69)
        x_drop_cols = ["success_streaming_ratio_users", "playlist_uri"]
        self.X_train, self.y_train = sm.fit_resample(
            self.train_frame.drop(columns=x_drop_cols),
            self.train_frame["success_streaming_ratio_users"],
        )
        self.shap_frame = pd.merge(
            self.X_train, self.y_train, left_index=True, right_index=True
        )

        self.X_test = self.test_frame.drop(columns=x_drop_cols)
        self.y_test = self.test_frame["success_streaming_ratio_users"]

        self.model = xgboost.XGBClassifier(
            n_estimators=n_estimators, verbosity=0, n_jobs=-1
        ).fit(self.X_train, self.y_train)

    def compute_accuracy(self):
        self.y_pred = self.model.predict(self.X_test)
        test_accuracy = round(accuracy_score(self.y_test, self.y_pred) * 100, 1)

        y_baseline_preds = np.zeros(len(self.y_test))
        baseline_accuracy = round(
            accuracy_score(self.y_test, y_baseline_preds) * 100, 1
        )
        return test_accuracy, baseline_accuracy


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
