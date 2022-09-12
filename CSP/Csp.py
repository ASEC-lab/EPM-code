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

    def __init__(self):
        # generate the public/private keys upon init
        self.__gen_enc_keys()

    def dec_array(self, arr: np.ndarray) -> np.ndarray:
        dec_arr = []
        for row in arr:
            dec_row = []
            for num in row:
                dec_row.append(self.pyfhelCtxt.decrypt(num, decode=True))
            dec_arr.append(dec_row)
        return dec_arr

    def __gen_enc_keys(self):
        self.pyfhelCtxt = Pyfhel()
        self.pyfhelCtxt.contextGen("bfv", n=16384, t=65537)
        self.pyfhelCtxt.keyGen()
        self.pyfhelCtxt.save_public_key("../pubkey.pk")
        self.pyfhelCtxt.save_context("../context.con")

    def __dec_array(self, arr: np.ndarray) -> np.ndarray:
        """
        decrypt a given numpy array
        note that this is a private function which should not be exposed to external parties
        @param arr: the array to decrypt
        @return: the decrypted array
        """
        assert arr.ndim <= 2, "Only 1D and 2D arrays are supported for decryption"
        arr_as_list = arr.tolist()  # faster for processing than numpy array?
        if arr.ndim == 2:
            decrypted_arr_as_list = [[self.__private_key.decrypt(x) for x in row] for row in arr_as_list]
        else:
            decrypted_arr_as_list = [self.__private_key.decrypt(x) for x in arr_as_list]
        decrypted_numpy_arr = np.array(decrypted_arr_as_list)
        return decrypted_numpy_arr

