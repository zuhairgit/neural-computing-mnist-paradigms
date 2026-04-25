import os
import numpy as np
import struct
import matplotlib.pyplot as plt

# --- 1. DATA LOADING (Same as Part A) ---
def load_mnist(path, kind='train'):
    labels_path = os.path.join(path, f'{kind}-labels.idx1-ubyte')
    images_path = os.path.join(path, f'{kind}-images.idx3-ubyte')
    with open(labels_path, 'rb') as lbpath:
        struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)
    with open(images_path, 'rb') as imgpath:
        struct.unpack('>IIII', imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)
    return images, labels

# --- 2. ADALINE CLASS (Widrow-Hoff Learning) ---
class Adaline:
    def __init__(self, input_size, learning_rate=0.0001, epochs=50):
        # Initialize weights and bias
        self.weights = np.zeros(input_size)
        self.bias = 0
        self.eta = learning_rate
        self.epochs = epochs
        self.cost = []

    def net_input(self, X):
        # Linear summation: z = w·x + b [cite: 52]
        return np.dot(X, self.weights) + self.bias

    def train(self, X, d):
        for epoch in range(self.epochs):
            # Adaline uses continuous linear output for training [cite: 52]
            output = self.net_input(X)
            errors = (d - output)
            
            # Delta Rule update: Δw = η * (d - y) * x [cite: 56]
            self.weights += self.eta * X.T.dot(errors)
            self.bias += self.eta * errors.sum()
            
            # MSE Loss function: E = 1/2 * Σ(d - y)^2 [cite: 55]
            cost = (errors**2).sum() / 2.0
            self.cost.append(cost)
        return self

# --- 3. EXPERIMENT EXECUTION ---
if __name__ == "__main__":
    # Load and Preprocess Data
    X_train, y_train = load_mnist('MINSET', kind='train')
    mask = (y_train == 0) | (y_train == 1)
    X_train_bin = X_train[mask].astype('float32') / 255.0
    y_train_bin = y_train[mask].astype('float32')

    # Required Learning Rates 
    rates = [0.0001, 0.001, 0.01, 0.1]
    plt.figure(figsize=(10, 6))

    for lr in rates:
        print(f"Training Adaline with η = {lr}...")
        ada = Adaline(input_size=784, learning_rate=lr, epochs=20)
        ada.train(X_train_bin, y_train_bin)
        
        # Plotting the cost for each rate
        plt.plot(range(1, len(ada.cost) + 1), np.log10(ada.cost), label=f'η = {lr}')

    plt.title('Adaline Convergence - Learning Rate Comparison')
    plt.xlabel('Epochs')
    plt.ylabel('log10(Sum-squared-error)')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()