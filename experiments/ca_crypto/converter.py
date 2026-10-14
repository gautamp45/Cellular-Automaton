class Conv:
    def __init__(self):
        self.plaintext = ""
        self.binary_text = ""
        
    
    def binary_string(self,plain):
        binary_str = ''.join(format(ord(char), '08b') for char in plain)
        return(binary_str)
    
    def convert_binary_to_text(self, binary_text):
        binary_string = binary_text.replace(' ', '')  # Remove any whitespaces in the binary string
        binary_string = binary_string.strip()  # Remove leading and trailing whitespaces

        try:
            text = ''.join(chr(int(binary_string[i:i+8], 2)) 
                           for i in range(0, len(binary_string), 8))
            return text
        
        except ValueError:
            print("Conversion failed")  # Print a message indicating the conversion failed
            return None

