import pandas as pd
from tqdm import tqdm
import time
import logging
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

logging.basicConfig(level=logging.INFO)

client_id = "e35493012e244de9bba1f21247427a75"
client_secret = "d743787e2ce14482865e843bf06fc1c5"
client_credentials_manager = SpotifyClientCredentials(
    client_id=client_id, client_secret=client_secret
)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)


def split_dataframe(df, chunk_size):
    chunks = list()
    num_chunks = len(df) // chunk_size + 1
    for i in range(num_chunks):
        chunks.append(df[i * chunk_size : (i + 1) * chunk_size])
    return chunks


def features_extract(features):
    """
    Parse the list of dictionaries returned by spotify endpoint for track attributes and return a
    formatted dataframe
    """
    list_acousticness = []
    list_danceability = []
    list_duration = []
    list_energy = []
    list_instrumentalness = []
    list_key = []
    list_liveness = []
    list_loudness = []
    list_mode = []
    list_speechiness = []
    list_tempo = []
    list_time = []
    list_valence = []

    for i in range(len(features)):

        # Check if any features were returned
        if features[i] == None:
            list_acousticness.append(None)
            list_danceability.append(None)
            list_duration.append(None)
            list_energy.append(None)
            list_instrumentalness.append(None)
            list_key.append(None)
            list_liveness.append(None)
            list_loudness.append(None)
            list_mode.append(None)
            list_speechiness.append(None)
            list_tempo.append(None)
            list_time.append(None)
            list_valence.append(None)

        else:
            list_acousticness.append(features[i]["acousticness"])
            list_danceability.append(features[i]["danceability"])
            list_duration.append(features[i]["duration_ms"])
            list_energy.append(features[i]["energy"])
            list_instrumentalness.append(features[i]["instrumentalness"])
            list_key.append(features[i]["key"])
            list_liveness.append(features[i]["liveness"])
            list_loudness.append(features[i]["loudness"])
            list_mode.append(features[i]["mode"])
            list_speechiness.append(features[i]["speechiness"])
            list_tempo.append(features[i]["tempo"])
            list_time.append(features[i]["time_signature"])
            list_valence.append(features[i]["valence"])

    frame_features = pd.DataFrame(
        {
            "track_id": list_track_ids,
            "acousticness": list_acousticness,
            "danceability": list_danceability,
            "duration": list_duration,
            "energy": list_energy,
            "instrumentalness": list_instrumentalness,
            "key": list_key,
            "liveness": list_liveness,
            "loudness": list_loudness,
            "mode": list_mode,
            "speechiness": list_speechiness,
            "tempo": list_tempo,
            "time": list_time,
            "valence": list_valence,
        }
    )

    return frame_features


if __name__ == "__main__":
    track_frame = pd.read_parquet(
        "../data/playlist_track_ids/playlist_track_ids_part_35000.parquet"
    ).reset_index(drop=True)

    track_deduped = track_frame.drop_duplicates(subset='track_id').reset_index(drop=True)

    track_chunks = split_dataframe(track_deduped, chunk_size=100)

    all_features = []
    for i in tqdm(range(len(track_chunks))):
        list_track_ids = list(track_chunks[i]["track_id"])
        success = False
        try_number = 1

        while not success:
            try:
                f = sp.audio_features(list_track_ids)

                features_frame = (
                    features_extract(f)
#                     .assign(user_id=track_chunks[i]["user_id"])
#                     .assign(playlist_id=track_chunks[i]["playlist_id"])
                )
                all_features.append(features_frame)
                success = True

            except:
                logging.info("error...waiting 3 seconds before retrying")
                time.sleep(3)
                try_number += 1

                if try_number == 3:
                    success = True
                    logging.info(f"skipping chunk: {i}")

        # Checkpoint every 100 chunks of songs
        if i != 0 and i % 100 == 0:
            checkpoint = pd.concat(all_features)
            checkpoint.to_parquet(
                f"../data/track_features/track_features_part_{i}.parquet"
            )
            logging.info(f"checkpoint save for part {i} success")

    # Final frame to be saved
    final = pd.concat(all_features)
    final.to_parquet("../data/track_features/track_features_all.parquet")
