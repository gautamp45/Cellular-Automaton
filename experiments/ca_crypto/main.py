from CA import OneDimensionalCA
from RSRG import RandomStateRuleGen
from converter import Conv
from visualiser import Visuals
from enc_dec import *

e = Encrypt()
d = Decrypt()
v = Visuals()
convt = Conv()


plaintext = "The quick brown fox"
bin_string = convt.binary_string(plaintext)

generator = RandomStateRuleGen()
rule_number = generator.RandomR()
generations = generator.RandomG()
initial_state = generator.RandomS(len(bin_string))

ca = OneDimensionalCA(rule_number, len(initial_state), generations)
ca.initialize(initial_state)
final_state = ca.evolve()

encrypted_binary = e.encrypt_text_with_final_state(final_state, bin_string)
encrypted_text = convt.convert_binary_to_text(encrypted_binary)

decrypted_binary = d.decrypt_text_with_final_state(final_state, encrypted_binary)
decrypted_text = convt.convert_binary_to_text(decrypted_binary)

print("Plain Text: "+ plaintext)
print("Binary Representation of Text: "+ bin_string)
print("Final State: "+ final_state)
print("Encrypted Binary: "+ encrypted_binary)
print("Encrypted Text: "+ str(encrypted_text))
print("Decrypted Binary: "+ decrypted_binary)
print("Decrypted Text: "+ str(decrypted_text))
v.visualize_all(bin_string, initial_state, final_state, 
              ca.grid, encrypted_binary, decrypted_binary)


