import shap
import streamlit as st
from pycaret.classification import *

from src.constants import GENRES, MODEL_NUMERICAL_FEATURES, MODEL_CATEGORICAL_FEATURES
from src.data_transformations import classify_success
from src.model import create_holdout, ShapObject
from src.state_functions import *

st.set_option("deprecation.showPyplotGlobalUse", False)


@st.cache
def load_success_data(base_frame, users_threshold, success_threshold):
    return classify_success(
        base_frame,
        users_threshold=users_threshold,
        success_threshold=success_threshold,
    )


def main():
    # Custom functionality for ensuring changing widgets do not cause previous sections to rests
    # state = get_state()
    st.title("What Makes a Playlist Successful?")
    st.write("Creator: Alexander Wong")

    base_frame = pd.read_parquet("data/streamlit_data.parquet")

    # Sidebar Inputs -------------------------------------------------------------------------------------------------
    experiment_name = st.sidebar.text_input("Experiment name:")

    genre_options = ["ALL"] + GENRES
    genre = st.sidebar.selectbox("Select genre:", options=genre_options)
    if genre == "All":
        genre = GENRES
    else:
        genre = [genre]

    users_threshold = st.sidebar.number_input(
        "Minimum monthly number of Users:",
        min_value=10,
    )
    success_threshold = (
        st.sidebar.slider(
            "Streaming-ratio success threshold (%):",
            min_value=1,
            max_value=99,
            value=75,
        )
        / 100
    )
    holdout_fraction = (
        st.sidebar.slider("Test Size (%):", min_value=1, max_value=30, value=10) / 100
    )
    model_map = {
        "Extreme Gradient Boosting": "xgboost",
        "Decision Tree Classifier": "dt",
        "Extra Trees Classifier": "et",
        "Light Gradient Boosting Machine": "lightgbm",
        "Random Forest Classifier": "rf",
    }
    model_selection = st.sidebar.multiselect(
        "Models to train:", options=list(model_map.keys())
    )
    include_models = [model_map[x] for x in list(model_selection)]
    # ----------------------------------------------------------------------------------------------------------------
    train = st.checkbox("Click to train models")
    if train:
        genre_frame = base_frame.loc[lambda f: f["genre_1"].isin(genre)]

        labelled_frame = load_success_data(
            genre_frame, users_threshold, success_threshold
        )
        train_frame, holdout_frame = create_holdout(
            labelled_frame, holdout_fraction=holdout_fraction
        )

        # PyCaret setup to train models
        experiment = setup(
            data=train_frame,
            numeric_features=MODEL_NUMERICAL_FEATURES,
            categorical_features=MODEL_CATEGORICAL_FEATURES,
            target="success_streaming_ratio_users",
            ignore_features=["playlist_uri"],
            test_data=holdout_frame,
            session_id=123,
            ignore_low_variance=True,
            remove_outliers=True,
            fix_imbalance=True,
            remove_multicollinearity=True,
            log_experiment=True,
            log_data=True,
            fold=2,
            n_jobs=-1,
            combine_rare_levels=True,
            experiment_name=experiment_name,
            silent=True,
        )

        list_models = compare_models(
            n_select=5, round=3, cross_validation=False, include=include_models
        )
        exp = pull()
        st.dataframe(exp)

        # User selects the desired model
        dict_models = {}
        for i, model in enumerate(exp.index):
            dict_models[model] = i

        user_selected_model = st.selectbox(
            "Select model to view feature importance:", exp.index
        )

        importance = st.checkbox("Click to calculate feature importance")

        if importance:
            selected_model = list_models[dict_models[user_selected_model]]

            # Overall importance -------------------------------------------------------------------------------------
            st.header("Shapley Additive Explanations (SHAP)")
            X_train = get_config(variable="X_train")
            y_train = get_config(variable="y_train")
            view = pd.merge(
                y_train, X_train, left_index=True, right_index=True
            ).reset_index(drop=True)

            explainer = shap.TreeExplainer(selected_model)
            shap_values = explainer.shap_values(X_train.to_numpy())

            st.subheader("Feature Importance - Average")
            st.pyplot(shap.summary_plot(shap_values, X_train, plot_type="bar"))

            st.subheader("Feature Importance - All Observations")
            st.pyplot(shap.summary_plot(shap_values, X_train))

            # Individual importance -----------------------------------------------------------------------------------
            st.header("Explaining Individual Predictions")
            st.dataframe(view)
            row = int(st.number_input("Row from dataframe to inspect"))

            understand = st.checkbox("Click to understand selected observation")
            if understand:
                st.subheader("Feature Contributions to Playlist Success")
                shap_object = ShapObject(
                    base_values=explainer.expected_value,
                    values=explainer.shap_values(X_train)[row, :],
                    feature_names=X_train.columns,
                    data=X_train.iloc[row, :],
                )
                st.pyplot(shap.waterfall_plot(shap_object))
                st.write("Analysis complete!")
            else:
                st.stop()
        else:
            st.stop()
    else:
        st.stop()

    # Mandatory to avoid rollbacks with widgets
    # state.sync()


if __name__ == "__main__":
    main()
