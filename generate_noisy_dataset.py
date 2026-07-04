import os
import glob
import numpy as np
import random
import soundfile as sf

# =========================
# PATH SETUP
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CLEAN_PATH = os.path.join(BASE_DIR, "Clean", "volunteers")
NOISE_PATH = os.path.join(BASE_DIR, "Noise")
OUTPUT_PATH = os.path.join(BASE_DIR, "generated_noisy")

os.makedirs(OUTPUT_PATH, exist_ok=True)

# =========================
# PARAMETERS (FROM PAPER)
# =========================
SNR_LEVELS = [-5, 0, 5, 10, 15, 20]

# =========================
# LOAD FILES
# =========================
clean_files = glob.glob(os.path.join(CLEAN_PATH, "**", "*.wav"), recursive=True)
noise_files = glob.glob(os.path.join(NOISE_PATH, "*.wav"))

print(f"Clean files: {len(clean_files)}")
print(f"Noise files: {len(noise_files)}")

# =========================
# FUNCTIONS
# =========================

def normalize_audio(audio):
    return audio / (np.max(np.abs(audio)) + 1e-8)


def mix_noises(noise_list, target_length):
    """Mix 1-4 random noises"""
    n = random.randint(1, 4)
    selected = random.sample(noise_list, n)

    mix = np.zeros(target_length)

    for noise_path in selected:
        noise, sr = sf.read(noise_path)

        # repeat or trim noise to match length
        if len(noise) < target_length:
            repeats = int(np.ceil(target_length / len(noise)))
            noise = np.tile(noise, repeats)

        noise = noise[:target_length]
        mix += noise

    return mix


def adjust_snr(clean, noise, snr_db):
    """Apply correct SNR scaling"""
    clean = clean.astype(np.float32)
    noise = noise.astype(np.float32)

    clean_power = np.mean(clean ** 2)
    noise_power = np.mean(noise ** 2)

    if noise_power == 0:
        return clean

    # compute scaling factor
    scale = np.sqrt(clean_power / (10 ** (snr_db / 10)) / noise_power)
    noise = noise * scale

    return clean + noise


# =========================
# MAIN PROCESS
# =========================

count = 0

for clean_path in clean_files:

    clean, sr = sf.read(clean_path)

    # skip empty files
    if len(clean) < 1000:
        continue

    for snr in SNR_LEVELS:

        # STEP 1: mix 1–4 noises
        noise_mix = mix_noises(noise_files, len(clean))

        # STEP 2: apply SNR
        noisy = adjust_snr(clean, noise_mix, snr)

        # STEP 3: normalize
        noisy = normalize_audio(noisy)

        # STEP 4: save file
        filename = os.path.basename(clean_path).replace(".wav", "")
        out_name = f"{filename}_snr{snr}_mix{count}.wav"

        out_path = os.path.join(OUTPUT_PATH, out_name)

        sf.write(out_path, noisy, sr)

        count += 1

        if count % 500 == 0:
            print(f"Generated {count} files...")

print("DONE ✔ Dataset generated successfully!")
print(f"Total files: {count}")