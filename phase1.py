import os
import numpy as np
import struct

def load_mnist(path, kind='train'):
    """Load MNIST data using the exact naming shown in your screenshot."""
    # Use a dot '.' before idx instead of a hyphen '-'
    labels_path = os.path.join(path, f'{kind}-labels.idx1-ubyte')
    images_path = os.path.join(path, f'{kind}-images.idx3-ubyte')

    print(f"Attempting to open: {labels_path}") # Debug line

    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
        # Flattening 28x28 images into 784-dimensional vectors [cite: 94]
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)

    return images, labels

# --- Execution ---
try:
    # 1. Load Data
    X_train, y_train = load_mnist('MINSET', kind='train')
    X_test, y_test = load_mnist('MINSET', kind='t10k')

    # 2. Preprocessing: Normalization (Scale pixels to 0-1 range)
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0

    # 3. Create Binary Subset for Part A (Digits 0 and 1)
    mask = (y_train == 0) | (y_train == 1)
    X_train_bin = X_train[mask]
    y_train_bin = y_train[mask]

    print("✅ Phase 1 Successful!")
    print(f"Full Train Shape: {X_train.shape}")
    print(f"Binary (0 vs 1) Train Shape: {X_train_bin.shape}")

except FileNotFoundError:
    print("❌ Error: Could not find the 'MINSET' folder. Check your path!")