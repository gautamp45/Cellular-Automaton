import matplotlib.pyplot as plt
import numpy as np

class OneDimensionalCA:
    def __init__(self, rule_number, width, generations):
        self.rule_number = rule_number
        self.rule = self.generate_rule(rule_number)
        self.width = width
        self.generations = generations
        self.grid = np.zeros((generations, width), dtype=np.int32)

    def initialize(self, initial_state):
        self.grid[0] = np.array(list(initial_state), dtype=np.int32)

    def apply_rule(self, current_row, next_row):
        for i in range(1, self.width - 1):
            pattern = tuple(self.grid[current_row, i - 1 : i + 2])
            next_row[i] = self.rule[pattern]
            
    def generate_rule(self, rule_number):
        rule_dict = {}
        binary_repr = format(rule_number, "08b")
        for i in range(7, -1, -1):
            pattern = tuple(int(bit) for bit in format(i, "03b"))
            rule_dict[pattern] = int(binary_repr[7 - i])
        return rule_dict

    def evolve(self):
        for i in range(1, self.generations):
            self.apply_rule(i - 1, self.grid[i])
            
        final_state = "".join(str(bit) for bit in self.grid[-1])
        return final_state

    def visualize(self):
        plt.imshow(self.grid, cmap="binary")
        plt.title("1D Cellular Automaton")
        plt.xlabel("Cell")
        plt.ylabel("Generation")
        plt.show()
