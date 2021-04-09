"""
This script is executed to obtain all track ids for each playlist
"""
import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from tqdm import tqdm
import logging
import time

logging.basicConfig(level=logging.INFO)

client_id = 'c809928a30d041ba9cbff2d365ff4bd7'
client_secret = 'ec8f46a3cf15429da5f7e1d9dda9854b'
client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)


def user_playlist_tracks(spotify_connection, user_id, playlist_id):
    """
    https://developer.spotify.com/documentation/web-api/reference/playlists/get-playlists-tracks/
    """
    # Maximum of 100 tracks are return each time
    response = spotify_connection.user_playlist_tracks(user_id, playlist_id, limit=100)
    results = response["items"]

    # Subsequently call the endpoint for the same user_id and playlist_id until all songs in playlist are obtained
    while len(results) < response["total"]:
        response = spotify_connection.user_playlist_tracks(
            user_id, playlist_id, limit=100, offset=len(results)
        )
        results.extend(response["items"])

    # extract only the specific track features
    track_ids = []
    popularity = []
    for result in results:
        track_ids.append(result["track"]["uri"].split(":")[-1])
        popularity.append(result["track"]["popularity"])

    frame = pd.DataFrame(
        {'track_id': track_ids,
         'popularity': popularity
         }).assign(user_id=user_id).assign(playlist_id=playlist_id)

    return frame


if __name__ == "__main__":
    filepath = "../data/playlist_data.parquet"
    logging.info(f"loading file from: {filepath}")
    data = pd.read_parquet(filepath)
    all_playlists = []

    for i in tqdm(range(len(data))):
        success = False
        user = data["user_id"][i]
        playlist = data["playlist_id"][i]
        try_number = 1

        while not success:
            try:
                playlist_frame = user_playlist_tracks(spotify_connection=sp, user_id=user, playlist_id=playlist)
                all_playlists.append(playlist_frame)
                success = True

            except:
                logging.info("timeout error...waiting 3 seconds before retrying")
                time.sleep(3)
                try_number += 1

                if try_number == 2:
                    success = True
                    logging.info(f"skipping user: {user} and playlist: {playlist}")

        # Checkpoint every 100 users
        if i != 0 and i % 100 == 0:
            checkpoint = pd.concat(all_playlists)
            checkpoint.to_parquet(f"../data/playlist_track_ids/playlist_track_ids_part_{i}.parquet")
            logging.info(f"checkpoint save for part {i} success")

    # Final frame to be saved
    final = pd.concat(all_playlists)
    final.to_parquet("../data/playlist_track_ids/playlist_track_ids_all.parquet")
