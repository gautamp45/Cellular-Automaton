import pygame
import numpy as np
from ca_2d import TwoDimensionalCA
from game_board import GameBoard
from grapher import visualize_evolution

pygame.init()

birth_rule = '08'
survival_rule = '4'

birth_rule = [digit for digit in birth_rule]
survival_rule = [digit for digit in survival_rule]

ca = TwoDimensionalCA(100, 70, birth_rule, survival_rule)

# B0123478/S01234678  AntiLife
# B01356/S012345      Wickstretcher And The Parasites

gb = GameBoard(ca)
gb.run_simulation()
