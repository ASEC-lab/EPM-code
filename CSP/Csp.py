from Pyfhel import PyCtxt, Pyfhel, PyPtxt
import multiprocessing
import numpy as np


class Csp:
    __public_key = None
    __private_key = None
    __pyfhelCtxt = None
    __n = None



    def __init__(self):
        # generate the public/private keys upon init
        self.__gen_enc_keys()
        self.__n = 2 ** 14

    def decrypt_arr(self, arr):
        # not for general use, only for debug and test
        # should be removed in final implementation
        return self.__pyfhelCtxt.decryptInt(arr)

    def decode_arr(self, arr):
        return self.__pyfhelCtxt.decode(arr)

    def encode_array(self, arr: np.ndarray):
        assert arr.ndim == 1, "Only 1 dimensional arrays are supported"
        arr_encoded = self.__pyfhelCtxt.encodeInt(arr)
        return arr_encoded

    def encrypt_array(self, arr: np.ndarray):
        assert arr.ndim == 1, "Only 1 dimensional arrays are supported"
        arr_encoded = self.__pyfhelCtxt.encodeInt(arr)
        arr_encrypted = self.__pyfhelCtxt.encryptPtxt(arr_encoded)
        return arr_encrypted

    def recrypt_array(self, arr: np.ndarray):
        dec_arr = self.__pyfhelCtxt.decryptInt(arr)
        recrypted_arr = self.encrypt_array(dec_arr)
        return recrypted_arr

    def sum_array(self, enc_arr):
        dec_arr = self.__pyfhelCtxt.decryptInt(enc_arr)
        arr_sum = np.array([np.sum(dec_arr)])
        enc_arr_sum = self.encrypt_array(arr_sum)
        return enc_arr_sum

    def get_enc_n(self):
        return self.__n

    def get_noise_level(self, enc_ctxt):
        return self.__pyfhelCtxt.noise_level(enc_ctxt)

    def __gen_enc_keys(self):
        self.__pyfhelCtxt = Pyfhel()
        # self.__pyfhelCtxt.contextGen("bfv", n=2 ** 13, t_bits=48, sec=128)
        self.__pyfhelCtxt.contextGen("bfv", n=2 ** 14, t_bits=60, sec=128)  # t=65537, sec=128)
        # self.__pyfhelCtxt.contextGen("ckks", n=2 ** 14, scale=2**30, qi=[60,30 ,30, 30, 60])
        self.__pyfhelCtxt.keyGen()
        self.__pyfhelCtxt.rotateKeyGen()
        self.__pyfhelCtxt.relinKeyGen()
        self.__pyfhelCtxt.save_public_key("../pubkey.pk")
        self.__pyfhelCtxt.save_rotate_key("../rotkey.pk")
        self.__pyfhelCtxt.save_context("../context.con")

def generate_P() -> Pyfhel:
    P = Pyfhel()
    bfv_params = {
        'scheme': 'BFV',  # can also be 'bfv'
        'n': 2 ** 14,  # Polynomial modulus degree, the num. of slots per plaintext,
        #  of elements to be encoded in a single ciphertext in a
        #  2 by n/2 rectangular matrix (mind this shape for rotations!)
        #  Typ. 2^D for D in [10, 16]
        't': 65537,  # Plaintext modulus. Encrypted operations happen modulo t
        #  Must be prime such that t-1 be divisible by 2^N.
        't_bits': 20,  # Number of bits in t. Used to generate a suitable value
        #  for t. Overrides t if specified.
        'sec': 128,  # Security parameter. The equivalent length of AES key in bits.
        #  Sets the ciphertext modulus q, can be one of {128, 192, 256}
        #  More means more security but also slower computation.
    }
    P.contextGen(**bfv_params)  # Generate context for bfv scheme
    P.keyGen()  # Key Generation: generates a pair of public/secret keys
    P.rotateKeyGen()  # Rotate key generation --> Allows rotation/shifting
    P.relinKeyGen()  # Relinearization key generation
    return P

def main():
    P = Pyfhel()
    bfv_params = {
        'scheme': 'BFV',  # can also be 'bfv'
        'n': 2 ** 14,  # Polynomial modulus degree, the num. of slots per plaintext,
        #  of elements to be encoded in a single ciphertext in a
        #  2 by n/2 rectangular matrix (mind this shape for rotations!)
        #  Typ. 2^D for D in [10, 16]
        't': 65537,  # Plaintext modulus. Encrypted operations happen modulo t
        #  Must be prime such that t-1 be divisible by 2^N.
        't_bits': 20,  # Number of bits in t. Used to generate a suitable value
        #  for t. Overrides t if specified.
        'sec': 128,  # Security parameter. The equivalent length of AES key in bits.
        #  Sets the ciphertext modulus q, can be one of {128, 192, 256}
        #  More means more security but also slower computation.
    }
    P.contextGen(**bfv_params)  # Generate context for bfv scheme
    P.keyGen()  # Key Generation: generates a pair of public/secret keys
    P.rotateKeyGen()  # Rotate key generation --> Allows rotation/shifting
    P.relinKeyGen()  # Relinearization key generation

    A = np.arange(10, dtype=np.int64)
    B = np.arange(10, dtype=np.int64) + 10

    p5 = np.asarray([5], dtype=np.int64)
    p3 = np.asarray([3], dtype=np.int64)

    p5e = P.encryptInt(p5)
    p3e = P.encryptInt(p3)

    result = P.mod(p5e, p3e)
    x=2


    p_5 = P.encodeInt(A)
    Ae = P.encryptInt(A)
    ctxt_mod = P.mod(Ae, 2)
    ptxt_mod = P.decrypt(ctxt_mod)

    Be = P.encryptInt(B)
    Ce = np.add(Ae,Be)
    C = P.decrypt(Ce)


    x = 2
    M = np.arange(100).reshape(10, 10).astype(np.int64)
    CSP = Csp()
    B = CSP.encode_array(arr=A)
    # D = CSP.encode_array(arr=M)
    print(CSP.decode_arr(B))

    C = CSP.encrypt_array(arr=A)
    print(CSP.decrypt_arr(arr=C))


if __name__ == '__main__':
    main()
