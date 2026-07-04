import pickle
import numpy as np
import keras
from keras import layers, models, regularizers

# =========================
# LOAD DATA
# =========================
with open("processed_data/X_train.pkl", "rb") as f:
    X_train = np.array(pickle.load(f), dtype=np.float32)

with open("processed_data/Y_train.pkl", "rb") as f:
    Y_train = np.array(pickle.load(f), dtype=np.float32)
    

with open("processed_data/X_test.pkl", "rb") as f:
    X_test = np.array(pickle.load(f), dtype=np.float32)

with open("processed_data/Y_test.pkl", "rb") as f:
    Y_test = np.array(pickle.load(f), dtype=np.float32)

print("Data loaded")

# =========================
# NORMALIZATION
# =========================
mean = np.mean(X_train)
std = np.std(X_train)

X_train = (X_train - mean) / (std + 1e-8)
X_test  = (X_test - mean) / (std + 1e-8)

# =========================
# PAPER DNN MODEL
# =========================
model = models.Sequential([
    layers.Input(shape=(2827,)),

    layers.Dense(
        2048,
        activation='relu',
        kernel_regularizer=regularizers.l2(1e-5)  # λ||W||² (paper Eq 2)
    ),

    layers.Dense(
        2048,
        activation='relu',
        kernel_regularizer=regularizers.l2(1e-5)
    ),

    layers.Dense(
        2048,
        activation='relu',
        kernel_regularizer=regularizers.l2(1e-5)
    ),

    layers.Dense(257, activation='linear')
])

# =========================
# COMPILE (MSE = paper Eq 2)
# =========================
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='mse'
)

model.summary()

# =========================
# TRAIN (FAST VERSION)
# =========================
history = model.fit(
    X_train,
    Y_train,
    validation_split=0.1,
    epochs=10,
    batch_size=256,
    shuffle=True
)

# =========================
# TEST
# =========================
loss = model.evaluate(X_test, Y_test)
print("Test Loss:", loss)

# =========================
# SAVE
# =========================
model.save("dnn_se_paper_model.h5")

print("DONE")