import numpy as np
import soundfile as sf
import glob
import os
from pesq import pesq
from pystoi import stoi
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

# =========================
# FOLDERS
# =========================
clean_root = "Clean"
noisy_folder = "generated_noisy"
enhanced_folder = "enhanced_output"

# =========================
# LOAD ALL CLEAN FILES
# =========================
clean_files = glob.glob(os.path.join(clean_root, "**", "*.wav"), recursive=True)
enhanced_files = sorted(glob.glob(os.path.join(enhanced_folder, "*.wav")))

print("Clean files found:", len(clean_files))
print("Enhanced files found:", len(enhanced_files))

# filename -> full path
clean_dict = {}

for f in clean_files:
    name = os.path.basename(f)
    clean_dict[name] = f

# =========================
# STORAGE
# =========================
pesq_noisy_list = []
pesq_enh_list = []

stoi_noisy_list = []
stoi_enh_list = []

matched = 0

sample_clean = None
sample_noisy = None
sample_enh = None
sample_sr = None

# =========================
# EVALUATION
# =========================
for enh_path in enhanced_files:

    enh_name = os.path.basename(enh_path)

    # Example:
    # si2261_snr10_mix123.wav
    # -> si2261.wav
    clean_name = enh_name.split("_snr")[0] + ".wav"

    if clean_name not in clean_dict:
        continue

    clean_path = clean_dict[clean_name]

    noisy_path = os.path.join(noisy_folder, enh_name)

    if not os.path.exists(noisy_path):
        continue

    clean, sr = sf.read(clean_path)
    noisy, _ = sf.read(noisy_path)
    enh, _ = sf.read(enh_path)

    # PESQ requires 16 kHz
    if sr != 16000:
        continue

    # align length
    min_len = min(len(clean), len(noisy), len(enh))

    clean = clean[:min_len].astype(np.float32)
    noisy = noisy[:min_len].astype(np.float32)
    enh = enh[:min_len].astype(np.float32)

    try:
        pesq_noisy = pesq(sr, clean, noisy, "wb")
        pesq_enh = pesq(sr, clean, enh, "wb")
    except:
        continue

    stoi_noisy = stoi(clean, noisy, sr, extended=False)
    stoi_enh = stoi(clean, enh, sr, extended=False)

    pesq_noisy_list.append(pesq_noisy)
    pesq_enh_list.append(pesq_enh)

    stoi_noisy_list.append(stoi_noisy)
    stoi_enh_list.append(stoi_enh)

    matched += 1

    # save one sample for spectrogram
    if sample_clean is None:
        sample_clean = clean
        sample_noisy = noisy
        sample_enh = enh
        sample_sr = sr

print("\nMatched files:", matched)

# =========================
# RESULTS
# =========================
print("\nRESULTS")
print("----------------------------")
print("Average PESQ (Noisy)    :", np.mean(pesq_noisy_list))
print("Average PESQ (Enhanced) :", np.mean(pesq_enh_list))
print("Average STOI (Noisy)    :", np.mean(stoi_noisy_list))
print("Average STOI (Enhanced) :", np.mean(stoi_enh_list))

# =========================
# BAR CHART
# =========================
labels = ["Noisy", "Enhanced"]

pesq_scores = [
    np.mean(pesq_noisy_list),
    np.mean(pesq_enh_list)
]

stoi_scores = [
    np.mean(stoi_noisy_list),
    np.mean(stoi_enh_list)
]

x = np.arange(len(labels))

plt.figure(figsize=(6,4))

plt.bar(x - 0.2, pesq_scores, 0.4, label="PESQ")
plt.bar(x + 0.2, stoi_scores, 0.4, label="STOI")

plt.xticks(x, labels)
plt.ylabel("Score")
plt.title("Speech Enhancement Performance")
plt.legend()

plt.tight_layout()
plt.show()

# =========================
# SPECTROGRAM
# =========================
def plot_spec(signal, sr, title):

    f, t, Sxx = spectrogram(signal, sr)

    plt.figure(figsize=(7,4))

    plt.pcolormesh(
        t,
        f,
        10*np.log10(Sxx + 1e-10),
        shading="gouraud"
    )

    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.colorbar(label="dB")

    plt.tight_layout()
    plt.show()

if sample_clean is not None:

    plot_spec(sample_clean, sample_sr, "Clean Speech")
    plot_spec(sample_noisy, sample_sr, "Noisy Speech")
    plot_spec(sample_enh, sample_sr, "Enhanced Speech")

# ======================================================
# SAVE EVALUATION RESULTS
# ======================================================

np.save("pesq_noisy.npy", np.array(pesq_noisy_list))
np.save("pesq_enhanced.npy", np.array(pesq_enh_list))

np.save("stoi_noisy.npy", np.array(stoi_noisy_list))
np.save("stoi_enhanced.npy", np.array(stoi_enh_list))

print("Evaluation results saved.")

print("\nSTEP 5 DONE ✔")