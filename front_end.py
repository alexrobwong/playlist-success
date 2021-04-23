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
    youtube_link = "[Link to recorded demo on Youtube](https://youtu.be/dPsGxb9lTUY)"
    st.markdown(youtube_link, unsafe_allow_html=True)

    # Sidebar Inputs -------------------------------------------------------------------------------------------------
    experiment_name_input = st.sidebar.text_input("Experiment name:")
    experiment_name = f"{experiment_name_input}_{str(datetime.now())}"

    genre_options = GENRES
    default_ix = GENRES.index("Dance & House")
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
            value=70,
        )
        / 100
    )
    holdout_fraction = (
        st.sidebar.slider("Test Size (%):", min_value=1, max_value=30, value=5) / 100
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

    # Application can only be run start to finish if xgboost is selected...add it to the list of options
    if "xgboost" not in include_models:
        include_models.append("xgboost")

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
        st.info("**Models were trained using default parameters**")
        st.info(
            "To improve individual model performance,"
            "please consider offline **hyperparameter tuning** techniques such as **Grid Search**. "
            "To improve overall performance, please consider advanced offline **ensembling** techniques "
            "such as **Bagging**, **Boosting**, **Stacking**"
        )

        # Model Definitions
        models_expander = st.beta_expander("Model Definitions")
        models_expander.write(
            "[**Decision Tree Classifier**](https://en.wikipedia.org/wiki/Decision_tree_learning)"
        )
        models_expander.write(
            "A Decision Tree is a simple representation for "
            "classifying examples, a form of Supervised Machine Learning where the data is "
            "continuously split according to a certain parameter. A decision tree starts with a "
            "single node, which branches into possible outcomes. Each of those outcomes "
            "leads to additional nodes, which branch off into other possibilities"
        )
        models_expander.write("")
        models_expander.write(
            "[**Random Forest Classifier**](https://en.wikipedia.org/wiki/Random_forest)"
        )
        models_expander.write(
            "An ensemble learning method"
            "that operates by constructing a multitude of decision trees at training time, "
            "where each tree is trained on a bootstrap replica of the training data and final "
            "model classification is decide via majority vote from the constituent trees"
        )
        models_expander.write("")
        models_expander.write(
            "[**Extra Trees Classifier**](https://quantdare.com/what-is-the-difference-between"
            "-extra-trees-and-random-forest/)"
        )
        models_expander.write(
            "Extremely randomized trees is similar to Random Forest, "
            "in that it builds multiple trees and splits nodes using random subsets of features, "
            "but with two key differences: it does not bootstrap observations (meaning it samples "
            "without replacement), and nodes are split on random splits, not best splits"
        )
        models_expander.write("")
        models_expander.write(
            "[**Extreme Gradient Boosting**](https://en.wikipedia.org/wiki/Gradient_boosting)"
        )
        models_expander.write(
            "Boosting is a technique which combines a learning "
            "algorithm in series to achieve a strong learner from many sequentially connected "
            "weak learners. In case of gradient boosted decision trees algorithm, "
            "the weak learners are decision trees where each tree attempts to minimize the errors "
            "of previous tree. Trees in boosting are weak learners but adding many trees in series a"
            "and each focusing on the errors from previous one make boosting a "
            "highly efficient and accurate model"
        )
        models_expander.write("")
        models_expander.write(
            "[**Light Gradient Boosting Machine**](https://lightgbm.readthedocs.io/en/latest/)"
        )
        models_expander.write(
            "A gradient boosting framework for machine "
            "learning originally developed by Microsoft. Similar to Extreme Gradient Boosting, "
            "it is based on decision tree algorithms, however unlike Extreme Gradient Boosting, "
            "the algorithm splits the tree leaf wise instead of level wise"
        )
        models_expander.write("")

        # Model Evaluation Metrics
        metrics_expander = st.beta_expander("Model Evaluation Metrics")
        metrics_expander.write("**Accuracy**")
        metrics_expander.write(
            "Accuracy is defined as the percentage of correct predictions for the test data."
            " It can be calculated easily by dividing the number of correct predictions by the "
            "number of total predictions."
        )
        metrics_expander.write("")
        metrics_expander.write("**AUC**")
        metrics_expander.write(
            "An ROC curve (receiver operating characteristic curve) is a graph showing the "
            "performance of a classification model at all classification thresholds. This curve "
            "plots the True Positive Rate (TP) and False Negative Rate (FP)"
        )
        metrics_expander.write("")
        metrics_expander.write("**Recall**")
        metrics_expander.write(
            "Recall is defined as the fraction of examples which were predicted to belong "
            "to a class with respect to all of the examples that truly belong in the class."
        )
        metrics_expander.write("")
        metrics_expander.write("**Precision**")
        metrics_expander.write(
            "Precision is defined as the fraction of relevant examples (true positives) among "
            "all of the examples which were predicted to belong in a certain class."
        )
        metrics_expander.write("")
        metrics_expander.write("**F1**")
        metrics_expander.write(
            "The traditional F-measure or balanced F-score (F1 score) is the harmonic mean "
            "of precision and recall and is calculated as --> F1 score = 2 * (Precision * Recall) / "
            "(Precision + Recall)"
        )
        metrics_expander.write("")
        metrics_expander.write("**Kappa**")
        metrics_expander.write(
            "The Kappa statistic (or value) is a metric that compares an Observed Accuracy with "
            "an Expected Accuracy (random chance). The kappa statistic is used not only to evaluate "
            "a single classifier, but also to evaluate classifiers amongst themselves. In addition, "
            "it takes into account random chance (agreement with a random classifier), which"
            " generally means it is less misleading than simply using accuracy as a metric "
            "(an Observed Accuracy of 80% is a lot less impressive with an Expected Accuracy of "
            "75% versus an Expected Accuracy of 50%)"
        )
        metrics_expander.write("")
        metrics_expander.write("**MCC**")
        metrics_expander.write(
            "Unlike the other metrics discussed above, MCC takes all the cells of the Confusion"
            " Matrix into consideration in its formula --> MCC = TP * TN – FP * FN / √ (TP +FP) * "
            "(TP + FN) * (TN + FP) * (TN + FN) .Similar to Correlation Coefficient, the range of "
            "values of MCC lie between -1 to +1. A model with a score of +1 is a perfect model "
            "and -1 is a poor model. This property is one of the key usefulness of MCC as it"
            " leads to easy interpretability."
        )
        metrics_expander.write("")

        # Additional model data
        opts = st.beta_expander("Additional Model Data", False)
        # Download the training data as an excel file
        if opts.button("Display Link to Download Model Training Data"):
            st.markdown(get_table_download_link(state.view), unsafe_allow_html=True)

        # Prompt to launch MLFlow
        if opts.button("Display Link to Spotify Model Training History"):
            st.info(
                "Note that this application uses MLFlow only when both the application and MLFlow are "
                "deployed locally"
            )

        # Overall importance ------------------------------------------------------------------------------------------
        st.write("")  # Intentional extra blank spaces
        st.write("")
        st.header(f"Success Drives for {selected_genre} Playlists")
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
            st.write("**Model parameters: **")
            st.write(state.new_selected_model)
            st.write("")
            st.write("**Generating Visualizations...**")
            bar = st.progress(0)

            if state.selected_model != state.new_selected_model:
                state.selected_model = state.new_selected_model
                state.explainer = shap.TreeExplainer(state.selected_model)
                state.shap_values = state.explainer.shap_values(
                    state.X_train.to_numpy()
                )
            bar.progress(25)

            # Overall Feature Importance -------------------------------------------------------------------------
            st.subheader("Success Drivers - Average")
            st.pyplot(
                shap.summary_plot(state.shap_values, state.X_train, plot_type="bar")
            )

            # Violin plot and waterfall plot only available at this time for XGBoost model
            if user_selected_model != "xgboost":
                st.warning(
                    "This PoC has only been configured for when **Extreme Gradient Boosting "
                    "(xgboost)** is selected for analysis"
                )
                bar.progress(100)
                st.stop()

            else:
                # Violin Feature Importance --------------------------------------------------------------------------
                st.subheader(f"Success Drivers - All {selected_genre} Playlists")
                st.pyplot(shap.summary_plot(state.shap_values, state.X_train))
                bar.progress(50)

                # Dependence plots for each of the top 3 features ----------------------------------------------------
                st.header(f"Shapley Dependence for {selected_genre} Playlists")
                vals = np.abs(state.shap_values).mean(0)
                feature_importance = pd.DataFrame(
                    list(zip(state.X_train.columns, vals)),
                    columns=["col_name", "feature_importance_vals"],
                )
                feature_importance = (
                    feature_importance.sort_values(
                        by=["feature_importance_vals"], ascending=False
                    )
                    .reset_index(drop=True)
                    .head(3)
                )

                top_features = list(feature_importance["col_name"])
                for feature in top_features:
                    index = list(state.X_train.columns).index(feature)
                    st.subheader(f"Shapley Value Dependence for {feature}")
                    st.pyplot(
                        shap.dependence_plot(
                            index,
                            state.shap_values,
                            state.X_train,
                            alpha=0.5,
                            interaction_index=None,
                        )
                    )
                bar.progress(70)

                # Individual importance -------------------------------------------------------------------------------
                st.header(f"Explaining {selected_genre} Playlist Success Prediction")

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
                    bar.progress(85)
                    st.subheader(
                        f"Feature Contributions to {selected_genre} Playlist #{state.row}"
                    )
                    st.pyplot(shap.waterfall_plot(shap_object))
                    bar.progress(100)
                    st.stop()
                else:
                    st.stop()
        else:
            st.stop()
    else:
        st.stop()


if __name__ == "__main__":
    main()
