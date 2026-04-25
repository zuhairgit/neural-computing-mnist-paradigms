import os
import numpy as np
import struct
import matplotlib.pyplot as plt

# --- 1. DATA LOADING (From Phase 1) ---
def load_mnist(path, kind='train'):
    labels_path = os.path.join(path, f'{kind}-labels.idx1-ubyte')
    images_path = os.path.join(path, f'{kind}-images.idx3-ubyte')
    
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)
    return images, labels

# --- 2. PERCEPTRON CLASS ---
class Perceptron:
    def __init__(self, input_size, learning_rate=0.01, epochs=10):
        self.weights = np.zeros(input_size)
        self.bias = 0
        self.eta = learning_rate
        self.epochs = epochs
        self.history = {'misclassifications': []}

    def activation(self, z):
        # Threshold activation [cite: 35]
        return 1 if z >= 0 else 0

    def predict(self, x):
        z = np.dot(x, self.weights) + self.bias
        return self.activation(z)

    def train(self, X, d):
        print(f"Starting Training: Learning Rate={self.eta}, Epochs={self.epochs}")
        for epoch in range(self.epochs):
            errors = 0
            for i in range(len(X)):
                y = self.predict(X[i])
                # Update rule: w(t+1) = w(t) + eta * (d - y) * x [cite: 45]
                update = self.eta * (d[i] - y)
                self.weights += update * X[i]
                self.bias += update
                if update != 0:
                    errors += 1
            self.history['misclassifications'].append(errors)
            print(f"Epoch {epoch+1}: Misclassifications = {errors}")
            if errors == 0:
                print("Converged successfully!")
                break

# --- 3. EXECUTION BLOCK ---
if __name__ == "__main__":
    # Load Data
    try:
        X_train, y_train = load_mnist('MINSET', kind='train')
        
        # Create Binary Subset (Digits 0 and 1) [cite: 44]
        mask = (y_train == 0) | (y_train == 1)
        X_train_bin = X_train[mask].astype('float32') / 255.0
        y_train_bin = y_train[mask].astype(np.int8)

        # Initialize and Train
        # input_size=784 for flattened 28x28 images [cite: 94]
        model = Perceptron(input_size=784, learning_rate=0.1, epochs=5)
        model.train(X_train_bin, y_train_bin)

        # Plot Misclassifications 
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, len(model.history['misclassifications']) + 1), 
                 model.history['misclassifications'], marker='o', color='b')
        plt.title('Perceptron Convergence (Digits 0 vs 1)')
        plt.xlabel('Epochs')
        plt.ylabel('Errors')
        plt.grid(True)
        plt.show()

    except Exception as e:
        print(f"An error occurred: {e}")