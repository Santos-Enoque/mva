import numpy as np
import matplotlib.pyplot as plt

def split_train_test(data, test_ratio=0.2):
    num_people = data.shape[5]
    num_test = int(num_people * test_ratio)
    test_data = data[:, :, :, :, :, :num_test]
    train_data = data[:, :, :, :, :, num_test:]
    return train_data, test_data

def reshape_data(data, n_samples):
    data = data.reshape(24, 24, 60*7, 30, n_samples)
    data = np.transpose(data, (0, 1, 4, 2, 3))
    data = np.reshape(data, (24, 24, 420 * n_samples, 30))
    return data

def split_and_reshape(data):
    data = np.split(data, 30, axis=3)
    data = [arr.squeeze(axis=3) for arr in data]
    return data

def flatten_and_transpose(data):
    data = [arr.reshape(24 * 24, arr.shape[2]) for arr in data]
    data = [arr.T for arr in data]
    return data

def plot_first_images(first_images):
    grid_rows = 6
    grid_cols = 5
    fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(12, 12))

    for i, img in enumerate(first_images):
        row = i // grid_cols
        col = i % grid_cols
        ax = axes[row, col]
        ax.imshow(img, cmap='gray')
        ax.set_title(f'Class {i+1}')
        ax.axis('off')

    plt.subplots_adjust(hspace=0.5, wspace=0.5)
    plt.show()