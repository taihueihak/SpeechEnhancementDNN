import os
import glob
import numpy as np
import soundfile as sf
import pickle
from scipy.signal import stft

# PATHS
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "generated_noisy")

clean_path = os.path.join(BASE_DIR, "Clean", "volunteers")

noisy_files = glob.glob(os.path.join(DATA_PATH, "*.wav"))

print("Noisy files:", len(noisy_files))

# PARAMETERS (FROM PAPER STYLE)
N_FFT = 512
HOP = 256

X = []  # noisy features
Y = []  # clean features

# LIMIT (IMPORTANT FOR MEMORY)
noisy_files = noisy_files[:20000]   # safe limit

for i, noisy_file in enumerate(noisy_files):

    noisy, sr = sf.read(noisy_file)

    # find matching clean file
    clean_file = noisy_file.replace("generated_noisy", "Clean").split("_snr")[0] + ".wav"

    if not os.path.exists(clean_file):
        continue

    clean, _ = sf.read(clean_file)

    # STFT
    f, t, noisy_spec = stft(noisy, fs=sr, nperseg=N_FFT, noverlap=HOP)
    _, _, clean_spec = stft(clean, fs=sr, nperseg=N_FFT, noverlap=HOP)

    # magnitude + log power
    noisy_mag = np.log1p(np.abs(noisy_spec))
    clean_mag = np.log1p(np.abs(clean_spec))

    X.append(noisy_mag.T)
    Y.append(clean_mag.T)

    if i % 500 == 0:
        print(f"Processed {i}")

# SAVE
with open("noisy.pkl", "wb") as f:
    pickle.dump(X, f)

with open("clean.pkl", "wb") as f:
    pickle.dump(Y, f)

print("STEP 3 DONE ✔")