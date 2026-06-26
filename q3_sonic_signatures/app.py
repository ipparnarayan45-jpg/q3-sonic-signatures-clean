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
    song, fs = librosa.load(
    path,
    sr=None,
    mono=True,
    duration=30      # only first 30 seconds
)

    f,t,S=signal.spectrogram(song,fs,
    nperseg=window,
    noverlap=window//2,
    )
    return f,t,S
from collections import defaultdict
import os
import pickle
import tempfile
import zipfile
import pandas as pd
import streamlit as st
import gdown

# ---------------- DATABASE ----------------

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

database = {}

st.success("Dummy database loaded")

st.title("Song Identifier")
st.success("Database loaded successfully!")
st.write(f"Total fingerprints loaded: {len(database)}")

# ---------------- IDENTIFICATION ----------------

def identify_song(q_hashes, database):
    votes = defaultdict(int)

    for hash_key, q_time in q_hashes:
        if hash_key not in database:
            continue

        for song_name, time_db in database[hash_key]:
            offset = round(time_db - q_time, 1)
            votes[(song_name, offset)] += 1

    if not votes:
        return None

    return sorted(votes.items(), key=lambda x: x[1], reverse=True)[0]


def process_audio(path):
    f, t, S = song_spectrogram(path)
    fig, ax = plt.subplots(figsize=(10,4))
    ax.pcolormesh(t, f, 10*np.log10(S + 1e-10), shading="gouraud")
    ax.set_title("Spectrogram")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    plt.colorbar(ax.collections[0], ax=ax)
    st.subheader("1. Spectrogram")
    st.pyplot(fig)
    peaks = extract_peaks(f, t, S)
    fig2, ax2 = plt.subplots(figsize=(10,4))

    times = [p[0] for p in peaks]
    freqs = [p[1] for p in peaks]

    ax2.scatter(times, freqs, s=4, color="red")
    ax2.set_title("Peak Constellation")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Frequency (Hz)")

    st.subheader("2. Peak Constellation")
    st.pyplot(fig2)
    hashes = generate_hashes(peaks)
    fig3, ax3 = plt.subplots(figsize=(10,4))

    step = max(1, len(hashes)//3000)

    for hash_key, t1 in hashes[::step]:
        f1 = hash_key[0] * 10
        f2 = hash_key[1] * 10
        dt = hash_key[2] / 100

        ax3.plot(
            [t1, t1 + dt],
            [f1, f2],
            color="blue",
            alpha=0.05
    )

    ax3.set_title("Fingerprint Hash Connections")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Frequency (Hz)")

    st.subheader("3. Fingerprint Hash Connections")
    st.pyplot(fig3)
    result = identify_song(hashes, database)

    if result is None:
        return {
            "Song": "No Match",
            "Offset": "-",
            "Votes": 0,
            "Peaks": len(peaks),
            "Hashes": len(hashes),
        }

    (song, offset), score = result
    return {
        "Song": song,
        "Offset": offset,
        "Votes": score,
        "Peaks": len(peaks),
        "Hashes": len(hashes),
    }

mode = st.radio("Mode", ["Single Clip", "Batch (ZIP)"])

if mode == "Single Clip":

    uploaded = st.file_uploader(
        "Upload an audio file",
        type=["wav", "mp3"],
        key="single",
    )

    if uploaded:
        st.audio(uploaded)

        suffix = "." + uploaded.name.split(".")[-1]

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.read())
            path = tmp.name

        result = process_audio(path)

        st.write("Peaks found:", result["Peaks"])
        st.write("Hashes generated:", result["Hashes"])

        if result["Song"] == "No Match":
            st.error("No matching song found")
        else:
            st.success("Song Identified!")
            st.write("Song:", result["Song"])
            st.write("Offset:", result["Offset"])
            st.write("Votes:", result["Votes"])

else:

    uploaded_zip = st.file_uploader(
        "Upload ZIP containing .wav/.mp3 files",
        type="zip",
        key="zip",
    )

    if uploaded_zip:

        with tempfile.TemporaryDirectory() as temp_dir:

            zip_path = os.path.join(temp_dir, "songs.zip")

            with open(zip_path, "wb") as f:
                f.write(uploaded_zip.read())

            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(temp_dir)

            rows = []

            for root, _, files in os.walk(temp_dir):
                for file in files:
                    if file.lower().endswith((".wav", ".mp3")):

                        full_path = os.path.join(root, file)
                        result = process_audio(full_path)

                        rows.append({
                            "File": file,
                            "Predicted Song": result["Song"],
                            "Votes": result["Votes"],
                            "Offset": result["Offset"],
                            "Peaks": result["Peaks"],
                            "Hashes": result["Hashes"],
                        })

            if rows:
                df = pd.DataFrame(rows)
                st.dataframe(df, use_container_width=True)

                csv = df.to_csv(index=False).encode("utf-8")

                st.download_button(
                    "Download Results CSV",
                    csv,
                    "results.csv",
                    "text/csv",
                )
            else:
                st.warning("No .wav or .mp3 files found in the ZIP.")
