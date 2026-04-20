import numpy as np
import matplotlib.pyplot as plt
from CA import TwoDimensionalCA

class CAVisualizer:
    def __init__(self, ca):
        self.ca = ca
        self.generation = 0
        self.population_history = []
        plt.style.use('dark_background')
        
    def update(self):
        self.generation += 1
        population = np.sum(self.ca.grid)
        self.population_history.append(population)
        
    def plot_current_state(self, mode='standard'):
        plt.clf()
        
        if mode == 'standard':
            self._plot_standard()
        elif mode == 'detailed':
            self._plot_detailed()
        elif mode == 'heatmap':
            self._plot_heatmap()
            
        plt.tight_layout()
        plt.pause(0.01)
        
    def _plot_standard(self):
        # Main grid
        plt.subplot(1, 1, 1)
        plt.imshow(self.ca.grid, cmap='viridis')
        plt.title(f'Generation: {self.generation}')
        plt.axis('off')
        
    def _plot_detailed(self):
        # Main grid
        plt.subplot(2, 2, 1)
        plt.imshow(self.ca.grid, cmap='viridis')
        plt.title(f'Current State (Gen: {self.generation})')
        plt.axis('off')
        
        # Population history
        plt.subplot(2, 2, 2)
        plt.plot(self.population_history, 'g-')
        plt.title('Population History')
        plt.xlabel('Generation')
        plt.ylabel('Live Cells')
        
        # Density heatmap
        plt.subplot(2, 2, 3)
        density = self._calculate_density()
        plt.imshow(density, cmap='hot')
        plt.title('Cell Density')
        plt.colorbar()
        plt.axis('off')
        
        # Statistics
        plt.subplot(2, 2, 4)
        self._plot_statistics()
        
    def _plot_heatmap(self):
        density = self._calculate_density()
        plt.imshow(density, cmap='hot')
        plt.title(f'Cell Density Heatmap (Gen: {self.generation})')
        plt.colorbar(label='Cell Density')
        plt.axis('off')
        
    def _calculate_density(self, window_size=5):
        density = np.zeros_like(self.ca.grid, dtype=float)
        padded = np.pad(self.ca.grid, window_size//2, mode='constant')
        
        for i in range(self.ca.height):
            for j in range(self.ca.width):
                window = padded[i:i+window_size, j:j+window_size]
                density[i, j] = np.mean(window)
        return density
        
    def _plot_statistics(self):
        plt.axis('off')
        stats = f"""Statistics:
Generation: {self.generation}
Population: {self.population_history[-1]}
Density: {self.population_history[-1]/(self.ca.width*self.ca.height):.2%}
Peak Pop: {max(self.population_history)}
Min Pop: {min(self.population_history)}"""
        plt.text(0.1, 0.5, stats, fontsize=8, family='monospace')

def visualize_evolution(initial_state, birth_rule, survival_rule, generations=100, mode='detailed'):
    # Create CA instance
    ca = TwoDimensionalCA(width=initial_state.shape[1], 
                         height=initial_state.shape[0],
                         birth_rule=birth_rule, 
                         survival_rule=survival_rule)
    ca.grid = initial_state.copy()
    
    # Create visualizer
    visualizer = CAVisualizer(ca)
    
    # Setup interactive plotting
    plt.ion()
    
    # Run evolution
    for _ in range(generations):
        visualizer.plot_current_state(mode=mode)
        ca.evolve()
        visualizer.update()
        
    plt.ioff()
    plt.show()