class Encrypt:
    def __init__(self):
        pass
    
    def encrypt_text_with_final_state(self,final_state, plaintext_binary):
        # Ensure that the lengths of the final state and plaintext binary are the same
        if len(final_state) != len(plaintext_binary):
            raise ValueError("Lengths of final state and plaintext binary do not match")

        # Encrypt the binary text by performing bitwise XOR with the final state
        encrypted_binary = ''.join(str(int(final_state[i]) ^ int(plaintext_binary[i])) for i in range(len(final_state)))

        return encrypted_binary
    
class Decrypt:
    def __init__(self):
        pass
    
    def decrypt_text_with_final_state(self,final_state, encrypted_binary):
        # Ensure that the lengths of the final state and encrypted binary match
        if len(final_state) != len(encrypted_binary):
            raise ValueError("Lengths of final state and encrypted binary do not match")

        # Decrypt the binary text by performing bitwise XOR with the final state
        decrypted_binary = ''.join(str(int(final_state[i]) ^ int(encrypted_binary[i])) for i in range(len(final_state)))

        return decrypted_binary