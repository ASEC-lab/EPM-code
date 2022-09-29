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
    __public_key = None
    __private_key = None
    __pyfhelCtxt = None
    __n = None

    def __init__(self):
        # generate the public/private keys upon init
        self.__gen_enc_keys()
        self.__n = 2 ** 13

    def decrypt_arr(self, arr):
        # not for general use, only for debug and test
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
        dec_arr = self.__pyfhelCtxt.decryptInt(arr)
        recrypted_arr = self.encrypt_array(dec_arr)
        return recrypted_arr

    def sum_array(self, enc_arr):
        dec_arr = self.__pyfhelCtxt.decryptInt(enc_arr)
        arr_sum = np.array([np.sum(dec_arr)])
        enc_arr_sum = self.encrypt_array(arr_sum)
        return enc_arr_sum

    '''
    def dec_array_old(self, arr: np.ndarray) -> np.ndarray:
        dec_arr = []
        for row in arr:
            dec_row = []
            for num in row:
                dec_row.append(self.pyfhelCtxt.decrypt(num, decode=True))
            dec_arr.append(dec_row)
        return dec_arr
    '''

    def get_enc_n(self):
        return self.__n

    def get_noise_level(self, enc_ctxt):
        return self.__pyfhelCtxt.noise_level(enc_ctxt)


    def __gen_enc_keys(self):
        self.__pyfhelCtxt = Pyfhel()
        self.__pyfhelCtxt.contextGen("bfv", n=2 ** 13, t=65537, sec=128)
        self.__pyfhelCtxt.keyGen()
        self.__pyfhelCtxt.rotateKeyGen()
        self.__pyfhelCtxt.relinKeyGen()
        self.__pyfhelCtxt.save_public_key("../pubkey.pk")
        self.__pyfhelCtxt.save_rotate_key("../rotkey.pk")
        self.__pyfhelCtxt.save_context("../context.con")



