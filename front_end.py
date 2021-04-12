import pandas as pd
import streamlit as st

from src.data_transformations import classify_success


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


def main():
    st.title("What Makes a Playlist Successful?")
    st.write("Creator: Alexander Wong")

    # Loading original spotify data that has been merged with aggregated playlist acoustic feature
    base_frame = load_base_data()

    # Sidebar Inputs -------------------------------------------------------------------------------------------------
    users_threshold = st.sidebar.number_input(
        "Minimum monthly number of Users:",
        min_value=10,
    )
    success_threshold = st.sidebar.slider(
        "Streaming-ratio success threshold (%):", min_value=1, max_value=99, value=75
    )
    model_test_size = st.sidebar.slider(
        "Model train-test split (%):", min_value=1, max_value=99, value=80
    )

    # ----------------------------------------------------------------------------------------------------------------
    success_frame = load_success_data(
        base_frame, users_threshold, success_threshold
    )
    st.write("Data has been loaded")


if __name__ == "__main__":
    main()
