from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend
import base64
import pickle


# Generate private and public keys for the server

def generate_keys():
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
    )
    public_key = private_key.public_key()
    return private_key, public_key


def generate_session_key(password, salt):
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
        backend=default_backend()
    )
    return base64.urlsafe_b64encode(kdf.derive(password))

# Serialize public key for transmission
def serialize_public_key(public_key):
    serialized_public_key = public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo
    )
    return serialized_public_key


# Deserialize public key after transmission
def deserialize_public_key(serialized_public_key):
    public_key = serialization.load_pem_public_key(serialized_public_key)
    return public_key


# Serialize public key for transmission
def serialize_private_key(private_key):
    private_key_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption()
    )
    return private_key_pem


# Deserialize public key after transmission
def deserialize_private_key(serialized_private_key):
    private_key  = serialization.load_pem_private_key(serialized_private_key, password=None)
    return private_key


# Generate and encrypt a session key using the public key
def encrypt_session_key(public_key):
    session_key = Fernet.generate_key()

    encrypted_session_key = public_key.encrypt(
        session_key,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None
        )
    )
    return session_key, encrypted_session_key


# Decrypt the encrypted session key using the private key
def decrypt_session_key(private_key, encrypted_session_key):
    session_key = private_key.decrypt(
        encrypted_session_key,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None
        )
    )
    return session_key



import zipfile
import os

def public_encrypt(pky, plainText):
    public_key = deserialize_public_key(pky)
    cipherText = public_key.encrypt(
        plainText,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None
        )
    )
    return cipherText

def private_decrypt(pky, cipherText):
    private_key = deserialize_private_key(pky)

    plainText = private_key.decrypt(
        cipherText,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None
        )
    )
    return plainText


def sess_encrypt(key, plainText):
    fernet_key = Fernet(key)
    return fernet_key.encrypt(plainText)

def sess_decrypt(key, cipherText):
    fernet_key = Fernet(key)
    return fernet_key.encrypt(cipherText)


def encrypt_file(input_file, output_file, key):
    with open(input_file, 'rb') as f:
        data = f.read()

    fernet = Fernet(key)
    encrypted_data = fernet.encrypt(data)

    with open(output_file, 'wb') as f:
        f.write(encrypted_data)


def decrypt_file(input_file, output_file, key):
    with open(input_file, 'rb') as f:
        encrypted_data = f.read()

    fernet = Fernet(key)
    decrypted_data = fernet.decrypt(encrypted_data)

    with open(output_file, 'wb') as f:
        f.write(decrypted_data)


def encrypt_large_zip(input_zip_path, output_zip_path, key):

    fernet_key = Fernet(key)
    # Open the input and output ZIP archives
    with zipfile.ZipFile(input_zip_path, 'r') as input_zip, \
            zipfile.ZipFile(output_zip_path, 'w') as output_zip:

        # Loop through each file in the input ZIP archive
        for file_info in input_zip.infolist():
            file_name = file_info.filename

            with input_zip.open(file_name, 'r') as input_file:
                file_contents = input_file.read()


            encrypted_file_contents = fernet_key.encrypt(file_contents)

            output_zip.writestr(file_name, encrypted_file_contents)

    return os.path.getsize(input_zip_path)


def decrypt_large_zip(input_zip_path, output_zip_path, key):

    fernet_key = Fernet(key)
    output_dir_path = os.path.dirname(output_zip_path)

    # Create the output directory if it does not exist
    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)

    # Open the input ZIP archive
    with zipfile.ZipFile(input_zip_path, 'r') as input_zip, \
            zipfile.ZipFile(output_zip_path, 'w') as zip_file:

        for file_info in input_zip.infolist():
            file_name = file_info.filename
            with input_zip.open(file_name, 'r') as input_file:
                encrypted_file_contents = input_file.read()

            decrypted_file_contents = fernet_key.decrypt(encrypted_file_contents)

            dest_file_path = os.path.join(os.path.dirname(output_dir_path), file_name)

            zip_file.writestr(dest_file_path, decrypted_file_contents)
