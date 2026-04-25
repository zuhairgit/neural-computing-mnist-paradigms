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

# --- 2. SOM CLASS ---
class KohonenSOM:
    def __init__(self, grid_size, input_dim, eta0=0.5, sigma0=None):
        self.grid_size = grid_size  # (rows, cols)
        self.eta0 = eta0
        self.sigma0 = sigma0 if sigma0 else max(grid_size) / 2
        # Initialize weights randomly [cite: 71]
        self.weights = np.random.uniform(0, 1, (grid_size[0], grid_size[1], input_dim))
        
    def get_bmu(self, x):
        # Euclidean distance between input x and all neuron weights [cite: 72]
        dist = np.linalg.norm(self.weights - x, axis=2)
        return np.unravel_index(np.argmin(dist), dist.shape)

    def train(self, data, epochs):
        n_samples = len(data)
        time_constant = epochs / np.log(self.sigma0)
        
        print(f"Training SOM on {n_samples} samples for {epochs} epochs...")
        for t in range(epochs):
            # Decay learning rate and sigma 
            eta_t = self.eta0 * np.exp(-t / epochs)
            sigma_t = self.sigma0 * np.exp(-t / time_constant)
            
            for x in data:
                bmu = self.get_bmu(x)
                
                # Update weights for BMU and neighbors [cite: 75]
                for r in range(self.grid_size[0]):
                    for c in range(self.grid_size[1]):
                        dist_sq = (r - bmu[0])**2 + (c - bmu[1])**2
                        if dist_sq < (sigma_t**2) * 4: # Optimization: check radius
                            # Gaussian neighborhood function [cite: 71]
                            h = np.exp(-dist_sq / (2 * sigma_t**2))
                            self.weights[r, c] += eta_t * h * (x - self.weights[r, c])
            if (t+1) % 5 == 0:
                print(f"Epoch {t+1}/{epochs} completed.")

    def get_u_matrix(self):
        # Calculate average distance between neighbors [cite: 77]
        u_matrix = np.zeros(self.grid_size)
        for r in range(self.grid_size[0]):
            for c in range(self.grid_size[1]):
                sum_dist = 0
                count = 0
                for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                    if 0 <= r+dr < self.grid_size[0] and 0 <= c+dc < self.grid_size[1]:
                        sum_dist += np.linalg.norm(self.weights[r,c] - self.weights[r+dr, c+dc])
                        count += 1
                u_matrix[r,c] = sum_dist / count
        return u_matrix

# --- 3. EXECUTION ---
if __name__ == "__main__":
    X_train, y_train = load_mnist('MINSET', kind='train')
    
    # Use a small subset (e.g., 1000 samples) to speed up training for today's deadline
    subset_size = 1000
    X_subset = X_train[:subset_size].astype('float32') / 255.0
    y_subset = y_train[:subset_size]

    # Initialize 10x10 SOM [cite: 67, 71]
    som = KohonenSOM(grid_size=(10, 10), input_dim=784, eta0=0.1)
    som.train(X_subset, epochs=20)

    # Visualization 1: U-Matrix [cite: 77]
    plt.figure(figsize=(6, 6))
    plt.imshow(som.get_u_matrix(), cmap='gray')
    plt.title("U-Matrix (Neuron Distances)")
    plt.colorbar()
    plt.show()

    # Visualization 2: Weight Maps (28x28 digit images) [cite: 77]
    fig, axes = plt.subplots(10, 10, figsize=(10, 10))
    for i in range(10):
        for j in range(10):
            axes[i, j].imshow(som.weights[i, j].reshape(28, 28), cmap='gray')
            axes[i, j].axis('off')
    plt.suptitle("SOM Weight Maps (Learned Prototypes)")
    plt.show()