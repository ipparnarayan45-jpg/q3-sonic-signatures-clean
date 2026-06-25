from collections import defaultdict
def generate_hashes(peaks,tar_t=3,fan_value=10):
    peaks=sorted(peaks,key=lambda x:x[0])

    hashes=[]
    for i,anchor in enumerate(peaks):
        t1,f1=anchor
        for j in range(i+1,min(i+fan_value+1,len(peaks))):
            t2,f2=peaks[j]
            dt=t2-t1

            if dt>tar_t:
                break

            hash_key=(int(f1//10),int(f2//10),int(dt*100))
            hashes.append((hash_key,round(t1,2)))
    return hashes
from scipy.ndimage import maximum_filter

def extract_peaks(f,t,S,percentile=95,size=15):
    S_db=10*np.log10(S+1e-10)
    local_max=S_db==maximum_filter(S_db,size=size)
    threshold=np.percentile(S_db,percentile)
    freq,time=np.where(local_max & (S_db>threshold))
    # print(f.shape,t.shape)
    # print(freq.shape,time.shape)
    freq_a=f[freq]
    time_a=t[time]
    return list(zip(time_a,freq_a))
from collections import defaultdict
import tempfile
import numpy as np
import matplotlib.pyplot as plt
import librosa
from scipy import signal
def song_spectrogram(path,window=4096):
    song,fs=librosa.load(path,sr=None)

    f,t,S=signal.spectrogram(song,fs,
    nperseg=window,
    noverlap=window//2,
    )
    return f,t,S
import streamlit as st
import pickle

st.title("Song Identifier")

import os
import gdown

DATABASE_PATH = "database/hash_database.pkl"

@st.cache_resource
def load_database():
    if not os.path.exists(DATABASE_PATH):
        os.makedirs("database", exist_ok=True)

        file_id = "119ZZGO7Rb2TofaKx1qZiuMhUf_DrDKiV"
        url = f"https://drive.google.com/uc?id={file_id}"

        gdown.download(url, DATABASE_PATH, quiet=False)

    with open(DATABASE_PATH, "rb") as f:
        return pickle.load(f)

database = load_database()

st.success("Database loaded successfully!")
st.write(f"Total fingerprints loaded: {len(database)}")

uploaded_file = st.file_uploader(
    "Upload an audio file",
    type=["wav","mp3"]
)

if uploaded_file is not None:
    st.audio(uploaded_file)
    st.success("Audio uploaded successfully!")
    import tempfile

if uploaded_file is not None:
    st.audio(uploaded_file)
    st.success("Audio uploaded successfully!")

    import tempfile

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_path = tmp_file.name

    st.write(temp_path)

    f, t, S = song_spectrogram(temp_path)

    peaks = extract_peaks(f, t, S)

    query_hashes = generate_hashes(peaks)

    st.write("Peaks found:", len(peaks))
    st.write("Hashes generated:", len(query_hashes))
    def identify_song(q_hashes, database):
        votes = defaultdict(int)

        for hash_key, q_time in q_hashes:
            if hash_key not in database:
                continue

            for song_name, time_db in database[hash_key]:
                offset = round(time_db - q_time, 1)
                votes[(song_name, offset)] += 1

        if len(votes) == 0:
            return None

        top = sorted(
            votes.items(),
            key=lambda x: x[1],
            reverse=True
    )

        return top[0]
    result = identify_song(query_hashes, database)

    if result is not None:
        (song_name, offset), score = result

        st.success("Song Identified!")
        st.write("Song:", song_name)
        st.write("Offset:", offset)
        st.write("Votes:", score)
    else:
        st.error("No matching song found")