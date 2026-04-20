import numpy as np
import pygame
import csv
from ca_2d import TwoDimensionalCA
from drawing import DrawingBoard

RULES = [
    ("B3/S23", "B3/S23"),
    ("B1/S1", "B1/S1"),
]
DATASET_SIZE = 100
CSV_FILE = "cellular_automaton_dataset.csv"

pygame.init()

def build_and_save_dataset():
    dataset = []

    for rule in RULES:
        initial_state = np.random.randint(2, size=(GRID_HEIGHT, GRID_WIDTH))

        ca = TwoDimensionalCA(GRID_WIDTH, GRID_HEIGHT, birth_rule=rule[0], survival_rule=rule[1])
        ca.grid = initial_state

        for _ in range(10):
            ca.evolve()

        final_state = ca.grid

        dataset.append((rule[0], rule[1], initial_state, final_state))

    with open(CSV_FILE, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["Birth Rule", "Survival Rule", "Initial State", "Final State"])
        for example in dataset:
            birth_rule, survival_rule, initial_state, final_state = example
            csv_writer.writerow([birth_rule, survival_rule, initial_state, final_state])

if __name__ == "__main__":
    GRID_WIDTH = 80
    GRID_HEIGHT = 60

    build_and_save_dataset()

    pygame.quit()
