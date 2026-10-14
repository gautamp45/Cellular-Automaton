import matplotlib.pyplot as plt
import numpy as np

class Visuals:
    def __init__(self):
        pass
         
    def visualize_all(self, bin_string, initial_state, final_state, evol, encrypted_bin):
        data = [initial_state, final_state, 
            evol, bin_string, 
            encrypted_bin]
    
        titles = ['Initial State', 'Final State', 
              'Evolution', 'Binary Representation of Text', 
              'Encrypted Binary Text']

        fig, axes = plt.subplots(3, 2, figsize=(12, 12))
        axes = axes.flatten()

        for ax, d, title in zip(axes, data, titles):
            if title == 'Evolution':
                matrix = evol
                ax.imshow(matrix, cmap='binary', aspect='equal')
                ax.set_title(title)
                ax.set_xticks([])
                ax.set_yticks([])
        
            else:
                matrix = np.array([list(map(int, d))])
                ax.imshow(matrix, cmap='binary', aspect='equal')
                ax.set_title(title)
                ax.set_xticks([])
                ax.set_yticks([])
        
                for i in range(len(d)):
                    ax.axvline(x=i + 0.5, color='grey', linewidth=1)
                    ax.axhline(y=0.5, color='grey', linewidth=1)

    
        plt.tight_layout()

        plt.show()
