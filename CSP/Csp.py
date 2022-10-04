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
        self.__n = 2 ** 14

    def decrypt_arr(self, arr):
        # not for general use, only for debug and test
        # return self.__pyfhelCtxt.decryptInt(arr)
        return self.__pyfhelCtxt.decryptFrac(arr)

    def decode_arr(self, arr):
        return self.__pyfhelCtxt.decode(arr)

    def ckks_align_mod_and_scale(self, this, other):
        return self.__pyfhelCtxt.align_mod_n_scale(this, other)

    def ckks_rescale_to_next(self, ciphertext):
        self.__pyfhelCtxt.rescale_to_next(ciphertext)

    def ckks_mod_switch_to_next(self, ciphertext):
        self.__pyfhelCtxt.mod_switch_to_next(ciphertext)

    def encode_array(self, arr: np.ndarray):
        assert arr.ndim == 1, "Only 1 dimensional arrays are supported"
        # arr_encoded = self.__pyfhelCtxt.encodeInt(arr)
        arr_encoded = self.__pyfhelCtxt.encodeFrac(arr)
        return arr_encoded

    def encrypt_array(self, arr: np.ndarray):
        assert arr.ndim == 1, "Only 1 dimensional arrays are supported"
        # arr_encoded = self.__pyfhelCtxt.encodeInt(arr)
        arr_encoded = self.__pyfhelCtxt.encodeFrac(arr)
        arr_encrypted = self.__pyfhelCtxt.encryptPtxt(arr_encoded)
        return arr_encrypted

    def recrypt_array(self, arr:np.ndarray):
       #  dec_arr = self.__pyfhelCtxt.decryptInt(arr)
        dec_arr = self.__pyfhelCtxt.decryptFrac(arr)
        recrypted_arr = self.encrypt_array(dec_arr)
        return recrypted_arr

    def sum_array(self, enc_arr):
        # dec_arr = self.__pyfhelCtxt.decryptInt(enc_arr)
        dec_arr = self.__pyfhelCtxt.decryptFrac(enc_arr)
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
        #self.__pyfhelCtxt.contextGen("bfv", n=2 ** 13, t_bits=48, sec=128)
        #self.__pyfhelCtxt.contextGen("bfv", n=2 ** 14, t_bits=60, sec=128)  # t=65537, sec=128)
        self.__pyfhelCtxt.contextGen("ckks", n=2 ** 14, scale=2**30, qi=[60, 30, 30, 30, 30, 30, 30, 30, 30, 60])
        #self.__pyfhelCtxt.contextGen("ckks", n=2 ** 15, scale=2**60, qi=[90, 60, 60, 60, 90])

        self.__pyfhelCtxt.keyGen()
        self.__pyfhelCtxt.rotateKeyGen()
        self.__pyfhelCtxt.relinKeyGen()
        self.__pyfhelCtxt.save_public_key("../pubkey.pk")
        self.__pyfhelCtxt.save_rotate_key("../rotkey.pk")
        self.__pyfhelCtxt.save_context("../context.con")



