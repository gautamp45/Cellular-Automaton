import numpy as np

class TwoDimensionalCA:
    def __init__(self, width, height, birth_rule, survival_rule):
        self.height = height
        self.width = width
        self.birth_rule = birth_rule
        self.survival_rule = survival_rule
        self.grid = np.zeros((height, width), dtype=int)

    def get_cell_state(self, row, col):
        return self.grid[row][col]

    def set_cell_state(self, row, col, state):
        self.grid[row][col] = state

    def evolve(self):
        new_grid = np.zeros((self.height, self.width), dtype=int)

        for x in range(self.height):
            for y in range(self.width):
                new_state = self.calculate_new_state(x, y)
                new_grid[x][y] = new_state

        self.grid = new_grid

        return self.grid

    def calculate_new_state(self, row, col):
        current_state = self.get_cell_state(row, col)
        neighbors = self.get_moore_neighborhood(row, col)
        live_neighbors = sum(neighbors)
        is_alive = current_state == 1

        if is_alive and str(live_neighbors) in self.survival_rule:
            return 1  # Cell survives
        elif not is_alive and str(live_neighbors) in self.birth_rule:
            return 1  # Cell is born
        else:
            return 0

    def get_moore_neighborhood(self, y, x):
        neighbors = []

        for dy in range(-1, 2):
            for dx in range(-1, 2):
                if dx == 0 and dy == 0:
                    continue  # Exclude the current cell
                nx = x + dx
                ny = y + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    neighbors.append(self.get_cell_state(ny, nx))
                else:
                    neighbors.append(0)  # Consider out-of-bounds cells as dead

        return neighbors
