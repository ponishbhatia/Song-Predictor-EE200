# app.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import librosa
import pickle
import os
import glob
import tempfile
from collections import defaultdict
from scipy.signal import spectrogram as sp
from scipy.ndimage import maximum_filter
from fingerprint import get_peaks, generate_hashes, match_from_array

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Zapp-tain America 🎵",
    page_icon="🎵",
    layout="wide"
)

# ─────────────────────────────────────────
# LOAD DATABASE & CONSTELLATIONS
# ─────────────────────────────────────────
song_files = glob.glob('songs/*.mp3')

@st.cache_resource
def load_database():
    with open('database.pkl', 'rb') as f:
        return pickle.load(f)

@st.cache_data
def get_all_constellations(_song_files, sr=22050):
    constellations = {}
    for song_path in _song_files:
        song_name = os.path.basename(song_path).replace('.mp3', '')
        y, _ = librosa.load(song_path, sr=sr, mono=True, duration=60)
        peaks_list, f, t = get_peaks(y, sr)
        constellations[song_name] = (peaks_list, f, t)
    return constellations

database = load_database()
constellations = get_all_constellations(song_files)

# ─────────────────────────────────────────
# APP TITLE
# ─────────────────────────────────────────
st.title("🎵 Zapp-tain America")
st.markdown("**Song identifier using audio fingerprinting**")
st.markdown(f"Database loaded: **{len(database)} unique hashes**")

# ─────────────────────────────────────────
# MODE SELECTION
# ─────────────────────────────────────────
mode = st.sidebar.radio("Select Mode", ["Single Clip", "Batch Mode", "Library"])

# ─────────────────────────────────────────
# SINGLE CLIP MODE
# ─────────────────────────────────────────
if mode == "Single Clip":
    st.header("🎧 Single Clip Identification")

    # ── SAMPLE SONGS ──
    st.subheader("🎵 Try a Sample Song")
    sample_songs = [os.path.basename(s).replace('.mp3', '')
                    for s in glob.glob('songs/*.mp3')][:5]

    selected_sample = st.selectbox("Pick a sample from the database",
                                   ["-- Select a sample --"] + sample_songs)

    if selected_sample != "-- Select a sample --":
        sample_path = os.path.join('songs', selected_sample + '.mp3')
        st.audio(sample_path)
        run_sample = st.button("Identify Sample")
    else:
        run_sample = False

    st.markdown("---")
    st.subheader("📤 Or Upload Your Own Clip")
    uploaded = st.file_uploader("Upload a query audio clip", type=['mp3', 'wav'])

    # ── DETERMINE INPUT SOURCE ──
    tmp_path = None
    source_name = None
    cleanup = False

    if run_sample and selected_sample != "-- Select a sample --":
        tmp_path = os.path.join('songs', selected_sample + '.mp3')
        source_name = selected_sample
        y, sr = librosa.load(tmp_path, sr=22050, mono=True)

    elif uploaded is not None:
        with tempfile.NamedTemporaryFile(delete=False,
                                         suffix=os.path.splitext(uploaded.name)[1]) as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name
        source_name = uploaded.name
        y, sr = librosa.load(tmp_path, sr=22050, mono=True)
        st.audio(uploaded)
        cleanup = True

    # ── RUN ANALYSIS IF INPUT EXISTS ──
    if tmp_path is not None and source_name is not None:
        with st.spinner('Fingerprinting...'):
            peaks_list, f, t = get_peaks(y, sr)
            query_hashes = generate_hashes(peaks_list, f, t)
            matched_song, ranked, offset_counts = match_from_array(y, sr, database)

        # ── MATCH RESULT BOX ──
        top_score = ranked[0][1] if ranked else 0
        runner_up_score = ranked[1][1] if len(ranked) > 1 else 0
        ratio = round(top_score / runner_up_score, 1) if runner_up_score > 0 else '∞'

        st.markdown(f"""
        <div style='background:#0d1117; border:1px solid #30363d;
                    border-radius:10px; padding:20px; margin-bottom:20px'>
            <p style='color:#58a6ff; font-size:12px;
                      letter-spacing:2px; margin:0'>MATCH FOUND</p>
            <h1 style='color:white; margin:5px 0'>{matched_song}</h1>
            <p style='color:#8b949e; margin:0'>
                cluster score <b style='color:white'>{top_score}</b> ·
                <b style='color:white'>{ratio}×</b> the runner-up
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.subheader("Top Matches")
        for name, score in ranked[:5]:
            st.write(f"**{name}** — score: {score}")

        # ── STEP 1: FEATURE EXTRACTION ──
        st.markdown("---")
        st.markdown("### STEP 1 · FEATURE EXTRACTION")
        st.markdown("#### From spectrogram to constellation")
        st.markdown(
            f"The clip was converted into a time-frequency map. "
            f"From that image, only the **{len(peaks_list)} most prominent peaks** "
            f"were kept. Discarding amplitude and phase makes the fingerprint robust "
            f"to EQ, volume changes, and noise."
        )

        col1, col2 = st.columns(2)

        with col1:
            f_s, t_s, Sxx = sp(y, fs=sr, nperseg=2048, noverlap=1024)
            Sxx_db = 10 * np.log10(Sxx + 1e-10)

            fig1, ax1 = plt.subplots(figsize=(6, 4))
            fig1.patch.set_facecolor('#0d1117')
            ax1.set_facecolor('#0d1117')
            ax1.imshow(Sxx_db, aspect='auto', origin='lower',
                       extent=[t_s[0], t_s[-1], f_s[0], f_s[-1]],
                       cmap='inferno', vmin=-80, vmax=0)
            ax1.set_ylim(0, 4000)
            ax1.set_xlabel('time (s)', color='white')
            ax1.set_ylabel('freq (Hz)', color='white')
            ax1.tick_params(colors='white')
            for spine in ax1.spines.values():
                spine.set_edgecolor('#30363d')
            plt.tight_layout()
            st.pyplot(fig1)
            plt.close()

        with col2:
            peak_freq_idx = [p[1] for p in peaks_list]
            peak_time_idx = [p[0] for p in peaks_list]
            peak_freqs = f[peak_freq_idx]
            peak_times = t[peak_time_idx]

            fig2, ax2 = plt.subplots(figsize=(6, 4))
            fig2.patch.set_facecolor('#0d1117')
            ax2.set_facecolor('#0d1117')
            ax2.scatter(peak_times, peak_freqs, c='cyan', s=3, alpha=0.7)
            ax2.set_xlim(t_s[0], t_s[-1])
            ax2.set_ylim(0, 4000)
            ax2.set_xlabel('time (s)', color='white')
            ax2.set_ylabel('freq (Hz)', color='white')
            ax2.tick_params(colors='white')
            for spine in ax2.spines.values():
                spine.set_edgecolor('#30363d')
            ax2.annotate(f'{len(peaks_list)} peaks',
                         xy=(0.98, 0.98), xycoords='axes fraction',
                         color='cyan', fontsize=10,
                         ha='right', va='top')
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close()

        # ── STEP 2: DATABASE SEARCH ──
        st.markdown("---")
        st.markdown("### STEP 2 · DATABASE SEARCH")
        st.markdown("#### Where in the song?")
        st.markdown(
            f"The **{len(query_hashes)} fingerprint hashes** were looked up against "
            f"every indexed track. Below is the full fingerprint of "
            f"**{matched_song}** reconstructed from the database. "
            f"The highlighted window is exactly where the query clip sits inside the full song."
        )

        if matched_song in constellations:
            db_peaks, db_f, db_t = constellations[matched_song]
            db_freq_idx = [p[1] for p in db_peaks]
            db_time_idx = [p[0] for p in db_peaks]

            best_offset = max(offset_counts[matched_song],
                              key=offset_counts[matched_song].get) \
                          if offset_counts[matched_song] else 0

            fig3, ax3 = plt.subplots(figsize=(12, 5))
            fig3.patch.set_facecolor('#0d1117')
            ax3.set_facecolor('#0d1117')
            ax3.scatter(db_time_idx, db_freq_idx, c='cyan', s=1, alpha=0.5)
            ax3.axvspan(best_offset, best_offset + len(t),
                        alpha=0.15, color='orange', label='Query window')
            ax3.axvline(best_offset, color='orange', linewidth=1.5)
            ax3.axvline(best_offset + len(t), color='orange', linewidth=1.5)
            ax3.set_xlabel('time (frames)', color='white')
            ax3.set_ylabel('freq bin', color='white')
            ax3.tick_params(colors='white')
            for spine in ax3.spines.values():
                spine.set_edgecolor('#30363d')
            ax3.legend(facecolor='#0d1117', labelcolor='white')
            plt.tight_layout()
            st.pyplot(fig3)
            plt.close()

        # ── STEP 3: THE PROOF ──
        st.markdown("---")
        st.markdown("### STEP 3 · THE PROOF")
        st.markdown("#### The alignment spike")
        st.markdown(
            f"Every matched hash votes for a time offset (database frame minus query frame). "
            f"Chance matches scatter votes randomly, forming a flat noise floor. "
            f"A genuine match makes them converge: "
            f"**{top_score} hashes agreed on a single offset**. "
            f"That spike cannot be a coincidence."
        )

        offsets_match = offset_counts.get(matched_song, {})
        if offsets_match:
            best_offset_val = max(offsets_match, key=offsets_match.get)
            best_count = offsets_match[best_offset_val]

            fig4, ax4 = plt.subplots(figsize=(12, 4))
            fig4.patch.set_facecolor('#0d1117')
            ax4.set_facecolor('#0d1117')
            ax4.bar(list(offsets_match.keys()),
                    list(offsets_match.values()),
                    width=2, color='cyan', alpha=0.4)
            ax4.bar([best_offset_val], [best_count],
                    width=2, color='orange')
            ax4.annotate(f'{best_count} hashes\nalign here',
                         xy=(best_offset_val, best_count),
                         xytext=(best_offset_val + 50, best_count * 0.8),
                         color='orange', fontsize=10,
                         arrowprops=dict(arrowstyle='->', color='orange'))
            ax4.set_xlabel('time offset (frames)', color='white')
            ax4.set_ylabel('hashes', color='white')
            ax4.tick_params(colors='white')
            for spine in ax4.spines.values():
                spine.set_edgecolor('#30363d')
            plt.tight_layout()
            st.pyplot(fig4)
            plt.close()

        if cleanup:
            os.unlink(tmp_path)

# ─────────────────────────────────────────
# BATCH MODE
# ─────────────────────────────────────────
elif mode == "Batch Mode":
    st.header("📂 Batch Identification")
    st.markdown("Upload multiple audio clips — get `results.csv` back")

    uploaded_files = st.file_uploader("Upload query clips",
                                       type=['mp3', 'wav'],
                                       accept_multiple_files=True)

    if uploaded_files and st.button("Run Batch"):
        results = []
        progress = st.progress(0)
        status = st.empty()

        for i, uploaded in enumerate(uploaded_files):
            status.text(f"Processing {uploaded.name}...")

            with tempfile.NamedTemporaryFile(delete=False,
                                              suffix=os.path.splitext(uploaded.name)[1]) as tmp:
                tmp.write(uploaded.read())
                tmp_path = tmp.name

            y, sr = librosa.load(tmp_path, sr=22050, mono=True)
            matched_song, ranked, _ = match_from_array(y, sr, database)

            filename_no_ext = os.path.splitext(uploaded.name)[0]
            results.append((filename_no_ext, matched_song))

            os.unlink(tmp_path)
            progress.progress((i + 1) / len(uploaded_files))

        st.subheader("Results")
        for filename, prediction in results:
            st.write(f"**{filename}** → {prediction}")

        csv_content = "filename,prediction\n"
        for filename, prediction in results:
            csv_content += f"{filename},{prediction}\n"

        st.download_button(
            label="📥 Download results.csv",
            data=csv_content,
            file_name="results.csv",
            mime="text/csv"
        )
        status.text("Done!")

# ─────────────────────────────────────────
# LIBRARY PAGE
# ─────────────────────────────────────────
elif mode == "Library":
    st.header("🎵 Song Library")
    st.markdown(f"**{len(song_files)} songs indexed in database**")

    song_hash_counts = defaultdict(int)
    for entries in database.values():
        for song_name, _ in entries:
            song_hash_counts[song_name] += 1

    cols_per_row = 4
    song_names = sorted(song_hash_counts.keys())

    for row_start in range(0, len(song_names), cols_per_row):
        cols = st.columns(cols_per_row)

        for col_idx, song_name in enumerate(song_names[row_start:row_start + cols_per_row]):
            with cols[col_idx]:
                try:
                    peaks_list, f_lib, t_lib = constellations[song_name]

                    fig, ax = plt.subplots(figsize=(4, 3))
                    fig.patch.set_facecolor('black')
                    ax.set_facecolor('black')

                    if peaks_list:
                        peak_freq_idx = [p[1] for p in peaks_list]
                        peak_time_idx = [p[0] for p in peaks_list]
                        peak_freqs = f_lib[peak_freq_idx]
                        peak_times = t_lib[peak_time_idx]
                        ax.scatter(peak_times, peak_freqs,
                                   c=peak_freqs, cmap='plasma',
                                   s=2, alpha=0.7)

                    ax.set_xlim(t_lib[0], t_lib[-1])
                    ax.set_ylim(0, 4000)
                    ax.axis('off')
                    plt.tight_layout(pad=0)
                    st.pyplot(fig)
                    plt.close()

                except Exception as e:
                    st.error(f"Could not load {song_name}")

                st.markdown(f"**{song_name}**")
                st.markdown(f"`{song_hash_counts[song_name]:,} hashes`")