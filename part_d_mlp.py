import os
import numpy as np
import struct
import matplotlib.pyplot as plt

# --- 1. DATA LOADING ---
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

# --- 2. MLP CLASS ---
class MLP:
    def __init__(self, layers, lr=0.1, activation='relu', l2_lambda=0.0001):
        self.lr = lr
        self.l2_lambda = l2_lambda # Part D-iv: L2 Regularization [cite: 100]
        self.activation_type = activation
        self.weights = [np.random.randn(layers[i], layers[i+1]) * np.sqrt(2/layers[i]) for i in range(len(layers)-1)]
        self.biases = [np.zeros((1, layers[i+1])) for i in range(len(layers)-1)]

    def _activate(self, x):
        if self.activation_type == 'relu': return np.maximum(0, x)
        return 1 / (1 + np.exp(-x)) # Sigmoid choice for comparison [cite: 96]

    def _activate_derivative(self, x):
        if self.activation_type == 'relu': return (x > 0).astype(float)
        s = 1 / (1 + np.exp(-x))
        return s * (1 - s)

    def _softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward(self, x):
        self.activations = [x]
        self.zs = []
        for i in range(len(self.weights)):
            z = np.dot(self.activations[-1], self.weights[i]) + self.biases[i]
            self.zs.append(z)
            if i == len(self.weights) - 1: # Output layer uses Softmax 
                self.activations.append(self._softmax(z))
            else:
                self.activations.append(self._activate(z))
        return self.activations[-1]

    def backward(self, x, y):
        m = x.shape[0]
        # Cross-Entropy Gradient: (Pred - Actual) [cite: 91, 94]
        delta = (self.activations[-1] - y) 
        
        for i in reversed(range(len(self.weights))):
            # Weight update with L2 Regularization [cite: 100]
            dw = (np.dot(self.activations[i].T, delta) / m) + (self.l2_lambda * self.weights[i])
            db = np.sum(delta, axis=0, keepdims=True) / m
            
            if i > 0: # Backpropagate using Chain Rule [cite: 87, 91]
                delta = np.dot(delta, self.weights[i].T) * self._activate_derivative(self.zs[i-1])
            
            self.weights[i] -= self.lr * dw
            self.biases[i] -= self.lr * db

# --- 3. EXECUTION ---
if __name__ == "__main__":
    X_train, y_train = load_mnist('MINSET', kind='train')
    X_test, y_test = load_mnist('MINSET', kind='t10k')

    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    y_train_oh = np.eye(10)[y_train]

    nn = MLP([784, 128, 64, 10], lr=0.01, activation='relu')

    batch_size = 64
    epochs = 10
    history = []

    print("🚀 Training MLP with Mini-batches...")
    for epoch in range(epochs):
        # Shuffle data each epoch
        permutation = np.random.permutation(X_train.shape[0])
        X_shuffled = X_train[permutation]
        y_shuffled = y_train_oh[permutation]

        for i in range(0, X_train.shape[0], batch_size):
            x_batch = X_shuffled[i:i + batch_size]
            y_batch = y_shuffled[i:i + batch_size]
            
            nn.forward(x_batch)
            nn.backward(x_batch, y_batch)
        
        # Check Test Accuracy
        preds = np.argmax(nn.forward(X_test), axis=1)
        acc = np.mean(preds == y_test)
        history.append(acc)
        print(f"Epoch {epoch+1}/{epochs} - Test Accuracy: {acc*100:.2f}%")

    # Real Progress Plot
    plt.plot(range(1, epochs + 1), history, marker='o')
    plt.title("MLP Accuracy Improvement")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.show()