import pandas as pd
import shap
import streamlit as st
import streamlit.components.v1 as components

from src.data_transformations import classify_success
from src.model import PlaylistSuccessPredictor, ShapObject

st.set_option('deprecation.showPyplotGlobalUse', False)


@st.cache
def load_base_data():
    return pd.read_parquet("data/streamlit_data.parquet")


@st.cache
def load_success_data(base_frame, users_threshold, success_threshold):
    return classify_success(
        base_frame,
        users_threshold=users_threshold,
        success_threshold=success_threshold / 100,
    )


@st.cache
def create_shap_object(base_values, data, values, feature_names):
    ShapObject(base_values, values, feature_names, data)
    return ShapObject(base_values, values, feature_names, data)


def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)


def main():
    st.title("What Makes a Playlist Successful?")
    st.write("Creator: Alexander Wong")

    # Loading original spotify data that has been merged with aggregated playlist acoustic feature
    base_frame = load_base_data()

    # Sidebar Inputs -------------------------------------------------------------------------------------------------
    genre_options = sorted(base_frame["genre_1"].unique())
    genre = st.sidebar.selectbox("Select genre:", options=genre_options)

    users_threshold = st.sidebar.number_input(
        "Minimum monthly number of Users:",
        min_value=10,
    )
    success_threshold = st.sidebar.slider(
        "Streaming-ratio success threshold (%):", min_value=1, max_value=99, value=75
    )
    # model_train_size = (
    #         st.sidebar.slider(
    #             "Model train-test split (%):", min_value=1, max_value=99, value=80
    #         )
    #         / 100
    # )

    # ----------------------------------------------------------------------------------------------------------------
    if st.button("Click to train model"):
        labelled_frame = load_success_data(
            base_frame, users_threshold, success_threshold
        )

        PS = PlaylistSuccessPredictor(labelled_frame, genre)
        PS.train_model(0.80, n_estimators=1000)
        test_accuracy, baseline_accuracy = PS.compute_accuracy()

        st.write(f"Model test accuracy: {test_accuracy}%")
        st.write(f"Baseline accuracy: {baseline_accuracy}%")

        # SHAP feature importance
        explainer = shap.TreeExplainer(PS.model)
        shap_values = explainer.shap_values(PS.X_train.to_numpy())

        st.header("SHAP Feature Importance")
        st.pyplot(shap.summary_plot(shap_values, PS.X_test, plot_type='bar'))

        st.pyplot(shap.summary_plot(shap_values, PS.X_train, show=True))

        st.write("DONE!")

    # Individual observation
    # row = int(st.number_input(
    #     "Observation number to inspect",
    # ))
    #
    # shap_object = ShapObject(
    #     base_values=explainer.expected_value,
    #     values=explainer.shap_values(PS.X_train)[row, :],
    #     feature_names=PS.X_train.columns,
    #     data=PS.X_train.iloc[row, :],
    # )
    #
    # st_shap(shap.waterfall_plot(shap_object))


if __name__ == "__main__":
    main()
