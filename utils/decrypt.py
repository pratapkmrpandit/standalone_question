import os

from cryptography.fernet import Fernet


class Encryption_Decryption:
    def __init__(self,encryption_key):
        self.encryption_key = encryption_key
        self.cipher = Fernet(self.encryption_key)

    def encrypt_value(self):
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(BASE_DIR, ".env"),"r") as file:
            lines = file.readlines()

        encrypted_lines = []
        for line in lines:
            if "=" in line and not line.startswith("#") and (("password" in line.lower()) or ("key" in line.lower())):  # Ignore comments
                key, value = line.strip().split("=", 1)
                encrypted_value = self.cipher.encrypt(value.encode()).decode()
                encrypted_lines.append(f"{key}={encrypted_value}")
            else:
                encrypted_lines.append(line.strip())  # Keep comments and blank lines

    # Save encrypted file in text mode
        with open(os.path.join(BASE_DIR, ".env"), "w") as enc_file:
            for enc_line in encrypted_lines:
                enc_file.write(enc_line + "\n")



    def decrypt_value(self,encrypted_value: str) -> str:
        """Decrypt an encrypted environment variable value."""
        try:

            return self.cipher.decrypt(encrypted_value.encode()).decode()
        except Exception as e:
            raise ValueError(f"Error decrypting value: {e}")
