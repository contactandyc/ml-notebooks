import matplotlib.pyplot as plt
import torch
import numpy as np
import random

# Set random seed for reproducibility
def set_random_seed(SEED):
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    np.random.seed(SEED)
    random.seed(SEED)

def draw_dnn(layers):
    fig, ax = plt.subplots()
    for i, n_neurons in enumerate(layers):
        y_offset = -n_neurons / 2.0 + 0.5
        for j in range(n_neurons):
            plt.scatter(i, j + y_offset, s=500, zorder=2)
            if i == 0:
                plt.text(i, j + y_offset, 'Input', verticalalignment='center', horizontalalignment='right')
            elif i == len(layers) - 1:
                plt.text(i, j + y_offset, 'Output', verticalalignment='center', horizontalalignment='right')
            else:
                plt.text(i, j + y_offset, f'Hidden{i}', verticalalignment='center', horizontalalignment='right')
    
    for i in range(len(layers) - 1):
        y_offset_src = -layers[i] / 2.0 + 0.5
        y_offset_dst = -layers[i+1] / 2.0 + 0.5
        for j in range(layers[i]):
            for k in range(layers[i+1]):
                plt.plot([i, i+1], [j + y_offset_src, k + y_offset_dst], c='black', zorder=1)
    
    plt.axis('off')
    plt.show()


