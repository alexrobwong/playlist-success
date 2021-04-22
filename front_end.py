import base64
from io import BytesIO
import logging
from datetime import datetime

import shap
import streamlit as st
from pycaret.classification import *

from src.constants import GENRES, MODEL_NUMERICAL_FEATURES, MODEL_CATEGORICAL_FEATURES
from src.data_transformations import classify_success
from src.model import create_holdout, ShapObject
from src.state_functions import *

logging.basicConfig(level=logging.INFO)
st.set_option("deprecation.showPyplotGlobalUse", False)


def to_excel(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine="xlsxwriter")
    df.to_excel(writer, sheet_name="Sheet1")
    writer.save()
    processed_data = output.getvalue()
    return processed_data


def get_table_download_link(df):
    val = to_excel(df)
    b64 = base64.b64encode(val)
    return (
        f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="playlist_success.xlsx">Download Model '
        f"Training Data file</a> "
    )


def main():
    logging.info("Main script is refreshed...")

    # Custom functionality for ensuring changing widgets do not cause previous sections to rests
    state = get_state()
    st.title("What Makes a Playlist Successful?")
    st.write(
        "This webtool trains & evaluates playlist success classification models, "
        "and generates intuitive visualizations for analyzing feature importance (Creator: Alexander Wong)"
    )

    # Sidebar Inputs -------------------------------------------------------------------------------------------------
    experiment_name_input = st.sidebar.text_input("Experiment name:")
    experiment_name = f"{experiment_name_input}_{str(datetime.now())}"

    genre_options = GENRES
    default_ix = GENRES.index("Pop")
    selected_genre = st.sidebar.selectbox(
        "Select genre:", options=genre_options, index=default_ix
    )

    # selected genre must be a list
    genre = [selected_genre]

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

    optionals = st.sidebar.beta_expander(
        "Additional Feature Engineering Parameters", False
    )
    polynomials_box = optionals.checkbox("Feature Polynomials")
    interactions_box = optionals.checkbox("Feature Interactions")
    ratios_box = optionals.checkbox("Feature Ratios")

    if polynomials_box:
        polynomials = True
    else:
        polynomials = False

    if interactions_box:
        interactions = True
    else:
        interactions = False

    if ratios_box:
        ratios = True
    else:
        ratios = False

    # Experiment & Model Training -------------------------------------------------------------------------------------
    train = st.checkbox("Click to train models")
    if train:

        # Check that models are selected - if none are selected, all models will be trained (undesired app behavior)
        if len(include_models) == 0 or include_models is None:
            raise Exception("No models were selected. Please re-start the application")

        base_frame = pd.read_parquet("data/streamlit_data.parquet")
        state.genre_frame = base_frame.loc[lambda f: f["genre_1"].isin(genre)]
        labelled_frame = classify_success(
            state.genre_frame, users_threshold, success_threshold
        )

        train_frame, holdout_frame = create_holdout(
            labelled_frame, holdout_fraction=holdout_fraction
        )

        # PyCaret setup to train models
        if not state.experiment_complete:
            setup(
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
                feature_interaction=interactions,
                feature_ratio=ratios,
                polynomial_features=polynomials,
            )
            state.list_models = compare_models(
                n_select=5, round=3, cross_validation=False, include=include_models
            )
            state.experiment_complete = True

            state.X_train = get_config(variable="X_train")
            state.y_train = get_config(variable="y_train")
            state.view = pd.merge(
                state.y_train, state.X_train, left_index=True, right_index=True
            ).reset_index(drop=True)

        # Display model training results
        st.header("Model Training & Testing Results")
        exp = pull()
        st.dataframe(exp)

        opts = st.beta_expander("Additional Model Data", False)
        # Download the training data as an excel file
        if opts.button("Display Link to Download Model Training Data"):
            st.markdown(get_table_download_link(state.view), unsafe_allow_html=True)

        # Prompt to launch MLFlow
        if opts.button("Display Link to Spotify Model Training History"):
            link = "[MLFlow](http://localhost:5000/#/)"
            st.markdown(link, unsafe_allow_html=True)

        # Overall importance ------------------------------------------------------------------------------------------
        st.write("")  # Intentional extra blank spaces
        st.write("")
        st.header(f"{selected_genre} Playlist Shapley Values")
        dict_models = {}
        for i, model in enumerate(exp.index):
            dict_models[model] = i

        user_selected_model = st.selectbox(
            "Select model to view feature importance:", exp.index
        )
        state.importance = st.checkbox("Click to calculate feature importance")
        if state.importance and state.experiment_complete:
            state.new_selected_model = state.list_models[
                dict_models[user_selected_model]
            ]

            if state.selected_model != state.new_selected_model:
                state.selected_model = state.new_selected_model
                state.explainer = shap.TreeExplainer(state.selected_model)
                state.shap_values = state.explainer.shap_values(
                    state.X_train.to_numpy()
                )

            # Overall Feature Importance -------------------------------------------------------------------------
            st.subheader("Success Drivers - Average")
            st.pyplot(
                shap.summary_plot(state.shap_values, state.X_train, plot_type="bar")
            )

            # Violin plot and waterfall plot only available at this time for XGBoost model
            if user_selected_model == "xgboost":

                # Violin Feature Importance --------------------------------------------------------------------------
                st.subheader("Success Drivers - All Playlists")
                st.pyplot(shap.summary_plot(state.shap_values, state.X_train))

                # Individual importance -------------------------------------------------------------------------------
                st.header("Explaining Individual Predictions")

                # Display the data frame for users to visually see the row they want to analyze
                st.subheader("Model Training Data")
                st.dataframe(state.view)
                state.new_row = int(
                    st.number_input(
                        "Row from dataframe to inspect",
                        min_value=0,
                        max_value=len(state.view),
                        value=10,
                    )
                )
                if state.row != state.new_row:
                    state.row = state.new_row
                    shap_object = ShapObject(
                        base_values=state.explainer.expected_value,
                        values=state.explainer.shap_values(state.X_train)[state.row, :],
                        feature_names=state.X_train.columns,
                        data=state.X_train.iloc[state.row, :],
                    )
                    st.subheader(f"Feature Contributions to Playlist #{state.row}")
                    st.pyplot(shap.waterfall_plot(shap_object))
                    st.stop()
                else:
                    st.stop()
            else:
                st.stop()
        else:
            st.stop()
    else:
        st.stop()


if __name__ == "__main__":
    main()
