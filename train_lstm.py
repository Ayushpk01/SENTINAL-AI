import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# 1. Load preprocessed data
# Make sure you ran 'prepare_data.py' first!
try:
    X_train = np.load("X_train.npy")
    X_test = np.load("X_test.npy")
    y_train = np.load("y_train.npy")
    y_test = np.load("y_test.npy")
    classes = np.load("label_encoder_classes.npy", allow_pickle=True)
except FileNotFoundError:
    print("❌ Error: .npy files not found. Run 'prepare_data.py' first.")
    exit()

print(f"Loaded Data Shape: {X_train.shape}")
print(f"Classes found: {classes}")

# 2. Dynamic Configuration
# FIX: Don't hardcode 3. Calculate it from the data.
num_classes = len(classes) 
print(f"Number of classes detected: {num_classes}")

# Convert labels to one-hot encoding
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

# 3. Build LSTM model
model = Sequential()
# Input shape comes directly from X_train (Sequence_Length, Features)
model.add(LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=False))
model.add(Dropout(0.2)) # Added Dropout to prevent overfitting
model.add(Dense(32, activation='relu'))
model.add(Dense(num_classes, activation='softmax')) # Output layer matches num_classes (4)

# 4. Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 5. Train model
print("🚀 Starting Training...")
history = model.fit(
    X_train, y_train, 
    epochs=50,  # Increased epochs for better convergence
    batch_size=32, 
    validation_data=(X_test, y_test)
)

# 6. Save the model
model.save("lstm_crowd_behavior.h5")
print("✅ Model saved as 'lstm_crowd_behavior.h5'")

# 7. Plot training history
plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title(f"LSTM Training Accuracy ({num_classes} Behaviors)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()