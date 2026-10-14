import numpy as np
import matplotlib.pyplot as plt
from CA import OneDimensionalCA
from RSRG import RandomStateRuleGen
import cv2
from converter import Conv
from visualizer_2 import Visuals
from enc_dec import *

e = Encrypt()
d = Decrypt()
v = Visuals()
convt = Conv()


plaintext = "Encryption"
bin_string = convt.binary_string(plaintext)

generator = RandomStateRuleGen()
rule_number = generator.RandomR()
generations = generator.RandomG()

initial_state = bin_string[:]

ca = OneDimensionalCA(rule_number, len(initial_state), generations)
ca.initialize(initial_state)
encrypted_binary = ca.evolve()
final_state = encrypted_binary

encrypted_text = convt.convert_binary_to_text(encrypted_binary)

print("Plain Text: "+ plaintext)
print("Binary Representation of Text: "+ bin_string)
print("Encrypted Binary: "+ encrypted_binary)
print("Encrypted Text: "+ str(encrypted_text))
print("Rule: ",rule_number)
print("Generations: ",generations)
v.visualize_all(bin_string, initial_state, final_state, ca.grid, encrypted_binary)
