from unittest import TestCase
from CSP.Csp import CSP
from MLE.Mle import MLE
from DataHandler.DataFormat import enc_array
import numpy as np
from Pyfhel import Pyfhel


class TestMLE(TestCase):

    def test_calc_dec_array_sum(self):
        csp = CSP()
        mle = MLE(csp)
        arr = np.array([1, 2, 3, 4, 5, 6])
        #arr = np.ones(500, dtype=np.int64)
        enc_arr = csp.encrypt_array(arr)
        sum = mle.calc_encrypted_array_sum(enc_arr, len(arr))
        dec_sum = csp.decrypt_arr(sum)
        print(dec_sum)
        assert dec_sum[0] == arr.sum()
