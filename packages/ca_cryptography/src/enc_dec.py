class Encrypt:
    def __init__(self):
        pass

    def encrypt_text_with_final_state(self, final_state, plaintext_binary):
        if len(final_state) != len(plaintext_binary):
            raise ValueError("Lengths of final state and plaintext binary do not match")

        encrypted_binary = ''.join(str(int(final_state[i]) ^ int(plaintext_binary[i])) for i in range(len(final_state)))

        return encrypted_binary


class Decrypt:
    def __init__(self):
        pass

    def decrypt_text_with_final_state(self, final_state, encrypted_binary):
        if len(final_state) != len(encrypted_binary):
            raise ValueError("Lengths of final state and encrypted binary do not match")

        decrypted_binary = ''.join(str(int(final_state[i]) ^ int(encrypted_binary[i])) for i in range(len(final_state)))

        return decrypted_binary
