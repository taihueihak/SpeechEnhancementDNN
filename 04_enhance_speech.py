import numpy as np
import soundfile as sf
from scipy.signal import stft, istft
import os
import glob
from keras.models import load_model
import pickle

# =========================
# LOAD MODEL
# =========================
model = load_model("dnn_se_paper_model.h5", compile=False)
print("Model loaded")

# =========================
# LOAD TRAIN NORMALIZATION (IMPORTANT)
# =========================
with open("processed_data/X_train.pkl", "rb") as f:
    X_train = np.array(pickle.load(f), dtype=np.float32)

train_mean = np.mean(X_train)
train_std = np.std(X_train)

# =========================
# PARAMETERS
# =========================
TAU = 5
N_FFT = 512
HOP = 256

# =========================
# FOLDER
# =========================
test_folder = "generated_noisy"
output_folder = "enhanced_output"
os.makedirs(output_folder, exist_ok=True)

# 🔥 LIMIT FOR SPEED (IMPORTANT FOR 32K FILES)
noisy_files = glob.glob(test_folder + "/*.wav")
noisy_files = noisy_files[:600]   # change to 600 for full test set

print("Files used:", len(noisy_files))

# =========================
# BUILD CONTEXT
# =========================
def build_context(frames):
    X = []
    for t in range(TAU, len(frames) - TAU):
        context = frames[t-TAU:t+TAU+1]
        X.append(context.flatten())
    return np.array(X)

# =========================
# PROCESS
# =========================
for file in noisy_files:

    noisy, sr = sf.read(file)

    _, _, Zxx = stft(noisy, fs=sr, nperseg=N_FFT, noverlap=HOP)

    magnitude = np.log1p(np.abs(Zxx)).T
    phase = np.angle(Zxx).T

    if len(magnitude) < 2 * TAU + 1:
        continue

    X = build_context(magnitude)

    # ✅ USE TRAIN NORMALIZATION (PAPER CORRECT)
    X = (X - train_mean) / (train_std + 1e-8)

    # prediction
    Y_pred = model.predict(X, verbose=0)

    # reconstruction
    enhanced_mag = np.expm1(Y_pred)

    full_mag = np.zeros_like(magnitude)

    for i in range(len(Y_pred)):
        full_mag[i + TAU] = enhanced_mag[i]

    full_mag[:TAU] = full_mag[TAU]
    full_mag[-TAU:] = full_mag[-TAU-1]

    full_mag = full_mag.T
    phase = phase.T

    Z = full_mag * np.exp(1j * phase)
    _, enhanced_audio = istft(Z, fs=sr, nperseg=N_FFT, noverlap=HOP)

    out_path = os.path.join(output_folder, os.path.basename(file))
    sf.write(out_path, enhanced_audio, sr)

    print("Saved:", out_path)

print("STEP 4 DONE ✔")