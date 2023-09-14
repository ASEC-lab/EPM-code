from Pyfhel import PyCtxt, Pyfhel, PyPtxt
import multiprocessing
import numpy as np

'''
Crypto Service Provider (CSP) implementation
The CSP assists in private key distribution and calculations on encrypted data
note that encryption is not done via this class as the CSP should not see data that is not
encrypted and masked
'''


class CSP:

    def __init__(self, prime, enc_n):
        self.__public_key = None
        self.__private_key = None
        #self.__pyfhelCtxt = None
        self.__prime = prime
        self.__n = enc_n
        # generate the public/private keys upon init
        self.__gen_enc_keys()


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

    def recrypt_array(self, arr:np.ndarray):
        dec_arr = self.__pyfhelCtxt.decrypt(arr)
        recrypted_arr = self.encrypt_array(dec_arr)
        return recrypted_arr

    def get_enc_n(self):
        return self.__n

    def get_noise_level(self, enc_ctxt):
        return self.__pyfhelCtxt.noise_level(enc_ctxt)

    def sum_arr(self, enc_ctxt, size):
        return self.__pyfhelCtxt.cumul_add(enc_ctxt, True, size)

    def __gen_enc_keys(self):
        self.__pyfhelCtxt = Pyfhel()
        #self.__pyfhelCtxt.contextGen("bfv", n=2 ** 13, t=self.__prime, sec=128)
        self.__pyfhelCtxt.contextGen("bfv", n=self.__n, t=self.__prime, sec=128)
        self.__pyfhelCtxt.keyGen()
        self.__pyfhelCtxt.rotateKeyGen()
        self.__pyfhelCtxt.relinKeyGen()
        #self.__pyfhelCtxt.save_public_key("../pubkey.pk")
        #self.__pyfhelCtxt.save_rotate_key("../rotkey.pk")
        #self.__pyfhelCtxt.save_context("../context.con")



