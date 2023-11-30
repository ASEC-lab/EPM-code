from Pyfhel import Pyfhel
from DataHandler.DataFormat import format_array_for_dec
from CRT.CrtVector import CrtVector
from CRT.CrtUtils import CrtUtils
import numpy as np
import time

'''
Crypto Service Provider (CSP) implementation
The CSP assists in private key distribution, decryption and publishing of the calculation results
'''


class CSP:

    def __init__(self, enc_n):
        self.__public_keys = {}
        self.__private_keys = {}
        self.__pyfhelCtxts = []
        self.__primes = []
        self.__n = enc_n

    def gen_keys(self, primes: list):
        offset = len(self.__primes)
        self.__primes.extend(primes)
        # generate the public/private keys upon init
        prime_indices = self.__gen_enc_keys(primes, offset)
        return prime_indices

    def get_num_of_slots(self):
        return self.__n // 2

    def get_poly_modulus_degree(self):
        return self.__n

    def decrypt_arr(self, prime_index, arr):
        # not for general use, only for debug and test
        # should be removed in final implementation
        return self.__pyfhelCtxts[prime_index].decryptInt(arr)

    def encode_array(self, prime_index: int, arr: np.ndarray):
        assert arr.ndim == 1, "Only 1 dimensional arrays are supported"
        arr_encoded = self.__pyfhelCtxts[prime_index].encodeInt(arr)
        return arr_encoded

    def encrypt_array(self, arr: np.ndarray, prime_index: int):
        assert arr.ndim == 1, "Only 1 dimensional arrays are supported"

        arr_encoded = self.__pyfhelCtxts[prime_index].encodeInt(arr)
        arr_encrypted = self.__pyfhelCtxts[prime_index].encryptPtxt(arr_encoded)
        return arr_encrypted

    def recrypt_array(self, prime_index: int, arr: np.ndarray):
        dec_arr = self.__pyfhelCtxts[prime_index].decrypt(arr)
        recrypted_arr = self.encrypt_array(dec_arr, prime_index)
        return recrypted_arr

    def get_enc_n(self):
        return self.__n

    def get_noise_level(self, prime_index, enc_ctxt):
        return self.__pyfhelCtxts[prime_index].noise_level(enc_ctxt)

    def sum_arr(self, prime_index, enc_ctxt, size):
        return self.__pyfhelCtxts[prime_index].cumul_add(enc_ctxt, True, size)

    def decrypt_crt_vector(self, crt_vector, m ,n):
        t_num_list = []
        t_denom_list = []
        rates_list = []
        s0_vals_list = []

        for i, crt_set in enumerate(crt_vector.get_vector()):
            assert i == crt_set.prime_index, "mismatch between CSP and crt_vector order"
            prime_index = crt_set.prime_index
            crt_set.t_num = self.__pyfhelCtxts[prime_index].decryptInt(crt_set.t_num)[:m]
            crt_set.t_denom = self.__pyfhelCtxts[prime_index].decryptInt(crt_set.t_denom)[:1]
            crt_set.rates = self.__pyfhelCtxts[prime_index].decryptInt(crt_set.rates)[:n]
            crt_set.s0_vals = self.__pyfhelCtxts[prime_index].decryptInt(crt_set.s0_vals)[:n]

            t_num_list.append(crt_set.t_num)
            t_denom_list.append(crt_set.t_denom)
            rates_list.append(crt_set.rates)
            s0_vals_list.append(crt_set.s0_vals)

        return t_num_list, t_denom_list, rates_list, s0_vals_list

    def decrypt_and_publish_results(self, crt_vector: CrtVector, m: int, n:int):

        file_timestamp = time.strftime("%Y%m%d-%H%M%S")
        crt_utils = CrtUtils()
        t_num_list, t_denom_list, rates_list, s0_vals_list = self.decrypt_crt_vector(crt_vector,m ,n)
        ages, max_age = crt_utils.calc_final_ages_crt(self.__primes, t_num_list, t_denom_list)
        ages = format_array_for_dec(ages)

        filename = 'ages_'+file_timestamp+'.log'
        with open(filename, 'w') as fp:
            fp.write("ages:\n")
            for age in ages:
                fp.write(f"{age}\n")
            fp.write("max age numerator:\n")
            fp.write(f"{max_age}\n")
            fp.write("max age numerator length: ")
            fp.write(f"{len(str(max_age))}\n")

        print("Results have been saved to: ", filename)


    def __gen_enc_keys(self, primes, offset):
        prime_indices = []
        # in case keys were previously added, add these in addition
        for i, prime in enumerate(primes):
            idx = offset + i
            print("CSP: generating keys for prime index: {} n: {} prime: {}".format(idx, self.__n, prime))
            ctxt = Pyfhel()
            self.__pyfhelCtxts.append(ctxt)
            self.__pyfhelCtxts[idx].contextGen("bfv", n=self.__n, t=prime, sec=128)
            self.__pyfhelCtxts[idx].keyGen()
            self.__pyfhelCtxts[idx].rotateKeyGen()
            self.__pyfhelCtxts[idx].relinKeyGen()

            #self.__pyfhelCtxt.save_public_key("../pubkey.pk")
            #self.__pyfhelCtxt.save_rotate_key("../rotkey.pk")
            #self.__pyfhelCtxt.save_context("../context.con")

            prime_indices.append(idx)

        return prime_indices


