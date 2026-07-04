import os
import pickle
from sklearn.model_selection import train_test_split

# =====================================================
# LOAD DATASET
# =====================================================

with open("noisy.pkl", "rb") as f:
    noisy = pickle.load(f)

with open("clean.pkl", "rb") as f:
    clean = pickle.load(f)

print("Dataset Loaded Successfully")
print("Number of noisy samples :", len(noisy))
print("Number of clean samples :", len(clean))

assert len(noisy) == len(clean), "Mismatch between noisy and clean datasets!"

# =====================================================
# SPLIT DATASET (80% TRAIN / 20% TEST)
# =====================================================

train_noisy, test_noisy, train_clean, test_clean = train_test_split(
    noisy,
    clean,
    test_size=0.20,
    random_state=42,
    shuffle=True
)

# =====================================================
# SHOW RESULT
# =====================================================

print("\nDataset Split")
print("-------------------------")
print("Training :", len(train_noisy))
print("Testing  :", len(test_noisy))

# =====================================================
# SAVE SPLIT DATA
# =====================================================

SAVE_DIR = "split_data"
os.makedirs(SAVE_DIR, exist_ok=True)

with open(os.path.join(SAVE_DIR, "train_noisy.pkl"), "wb") as f:
    pickle.dump(train_noisy, f)

with open(os.path.join(SAVE_DIR, "train_clean.pkl"), "wb") as f:
    pickle.dump(train_clean, f)

with open(os.path.join(SAVE_DIR, "test_noisy.pkl"), "wb") as f:
    pickle.dump(test_noisy, f)

with open(os.path.join(SAVE_DIR, "test_clean.pkl"), "wb") as f:
    pickle.dump(test_clean, f)

print("\nDataset saved successfully!")
print("Location:", SAVE_DIR)