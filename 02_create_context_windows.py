import pickle
import numpy as np
import os

# =====================================================
# LOAD SPLIT DATA (FROM STEP 1)
# =====================================================

with open("split_data/train_noisy.pkl", "rb") as f:
    train_noisy = pickle.load(f)

with open("split_data/train_clean.pkl", "rb") as f:
    train_clean = pickle.load(f)

with open("split_data/test_noisy.pkl", "rb") as f:
    test_noisy = pickle.load(f)

with open("split_data/test_clean.pkl", "rb") as f:
    test_clean = pickle.load(f)

print("Data loaded successfully")

# =====================================================
# PARAMETERS (FROM PAPER)
# =====================================================

TAU = 5  # context window size (t-5 to t+5)
MAX_SAMPLES = 200000   # ✔ memory-safe limit

# =====================================================
# FUNCTION: CREATE CONTEXT WINDOWS
# =====================================================

def create_context(noisy_set, clean_set):

    X = []
    Y = []
    sample_count = 0

    for utt_idx in range(len(noisy_set)):

        noisy = np.array(noisy_set[utt_idx])  # (T, 257)
        clean = np.array(clean_set[utt_idx])  # (T, 257)

        T = noisy.shape[0]

        if T < (2 * TAU + 1):
            continue

        for t in range(TAU, T - TAU):

            # ✔ STOP IF LIMIT REACHED
            if sample_count >= MAX_SAMPLES:
                return np.array(X), np.array(Y)

            # =================================================
            # CONTEXT WINDOW (t-5 to t+5)
            # =================================================
            context = noisy[t - TAU : t + TAU + 1]  # (11, 257)

            # flatten → 2827 features
            context_flat = context.flatten()

            # target clean frame
            target = clean[t]

            X.append(context_flat)
            Y.append(target)

            sample_count += 1

    return np.array(X), np.array(Y)

# =====================================================
# BUILD TRAIN / TEST DATA
# =====================================================

print("Creating TRAIN data...")
X_train, Y_train = create_context(train_noisy, train_clean)

print("Creating TEST data...")
X_test, Y_test = create_context(test_noisy, test_clean)

# =====================================================
# OUTPUT SHAPES
# =====================================================

print("\nFINAL SHAPES")
print("----------------------")
print("X_train:", X_train.shape)
print("Y_train:", Y_train.shape)
print("X_test :", X_test.shape)
print("Y_test :", Y_test.shape)

# =====================================================
# SAVE PROCESSED DATA
# =====================================================

os.makedirs("processed_data", exist_ok=True)

with open("processed_data/X_train.pkl", "wb") as f:
    pickle.dump(X_train, f)

with open("processed_data/Y_train.pkl", "wb") as f:
    pickle.dump(Y_train, f)

with open("processed_data/X_test.pkl", "wb") as f:
    pickle.dump(X_test, f)

with open("processed_data/Y_test.pkl", "wb") as f:
    pickle.dump(Y_test, f)

print("\nContext windows created and saved successfully!")