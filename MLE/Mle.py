import math

import numpy as np
import time
from Pyfhel import Pyfhel, PyCtxt
from inspect import getframeinfo, stack
import os

'''
Machine Learning Engine (MLE) implementation
The MLE receives the encrypted input data from the DO and calculates the model
'''

# Global Definitions
# number of times to try and generate the random matrix R
RANDOM_MATRIX_GEN_THRESHOLD = 100
# maximum value to use in random generated matrix
# not too high in order to meet the requirements of the homomorphic encryption/decryption
RANDOM_MATRIX_MAX_VAL = 100



class MLE:

    def __init__(self, csp):

        self.ages = None
        self.meth_val_list = None
        self.transposed_meth_val_list = None
        self.transposed_test_meth_val_list = None
        self.m = None
        self.n = None
        self.test_m = None
        self.recrypt_count = 0
        self.FILE_COUNTER = 0
        self.csp = csp
        self.rounds = 2
        rand_mask = np.random.randint(1, high=RANDOM_MATRIX_MAX_VAL, size=(csp.get_enc_n()//2))
        self.rand_mask = csp.encrypt_array(rand_mask)

    def get_data_from_DO(self, meth_val_list, enc_transposed_meth_val_list, enc_transposed_test_meth_vals,
                         ages, m, n, test_m, rounds):

        self.meth_val_list = meth_val_list
        self.transposed_meth_val_list = enc_transposed_meth_val_list
        self.transposed_test_meth_val_list = enc_transposed_test_meth_vals
        self.ages = ages
        self.m = m
        self.n = n
        self.test_m = test_m
        self.rounds = rounds

    def check_noise_lvl(self, ctxt):
        # check the noise level (number of remaining mult operations)
        # as the data needs to be decrypted, mask it first
        #ctxt += self.rand_mask
        lvl = self.csp.get_noise_level(ctxt)
        #assert lvl > 0, "reached 0 noised budget"

        if lvl < 20:
            ctxt = self.csp.recrypt_array(ctxt)
            self.recrypt_count += 1
            print("Recrypting. New noise level: ", self.csp.get_noise_level(ctxt))
            caller = getframeinfo(stack()[3][0])
            print("%s:%d - %s" % (caller.filename, caller.lineno, " called this"))

        #ctxt -= self.rand_mask
        return ctxt

    def __safe_math_run_op__(self, ctxt1, ctxt2, operation):

        match operation:
            case '+':
                result = ctxt1 + ctxt2
            case '-':
                result = ctxt1 - ctxt2
            case '*':
                result = ctxt1 * ctxt2
                result = ~result
            case _:
                assert False, "Operation " + operation + " not supported"

        return result

    def safe_math(self, ctxt1, ctxt2, operation):

        result = self.__safe_math_run_op__(ctxt1, ctxt2, operation)
        if isinstance(result, PyCtxt):
            lvl = self.csp.get_noise_level(result)
            if lvl == 0:
                if isinstance(ctxt1, PyCtxt):
                    #print("Noise level of ctxt1 before recrypt: ", self.csp.get_noise_level(ctxt1))
                    ctxt1 = self.csp.recrypt_array(ctxt1)
                    #print("Noise level of ctxt1 after recrypt: ", self.csp.get_noise_level(ctxt1))
                if isinstance(ctxt2, PyCtxt):
                    #print("Noise level of ctxt2 before recrypt: ", self.csp.get_noise_level(ctxt2))
                    ctxt2 = self.csp.recrypt_array(ctxt2)
                    #print("Noise level of ctxt2 after recrypt: ", self.csp.get_noise_level(ctxt2))

                result = self.__safe_math_run_op__(ctxt1, ctxt2, operation)
                lvl = self.csp.get_noise_level(result)

                assert lvl > 0, "Noise level is 0 even after recrypt"

        return result

    def safe_mul(self, ctxt1, ctxt2):
        """
        Safely multiply 2 ciphertext
        After multiplication it is important to relinearize the array in order to reduce the polynom size
        failing to do this will result in inability to shift and maybe other bad things which I have not yet discovered
        In addition, if the noise level reaches 0, it will be impossible to decrypt the data
        In this case, need to perform a recrypt in order to add some noise level.
        @param ctxt1: first context to multiply
        @param ctxt2: second context to multiply
        @return: the multiplication result
        """

        # mask before sending to CSP for noise level check - doesn't work with BGV
        if isinstance(ctxt1, PyCtxt):
            ctxt1 = self.check_noise_lvl(ctxt1)
        if isinstance(ctxt2, PyCtxt):
            ctxt2 = self.check_noise_lvl(ctxt2)

        result = ctxt1 * ctxt2
        if self.csp.get_noise_level(result) == 0:
            if isinstance(ctxt1, PyCtxt):
                ctxt1 = self.csp.recrypt_array(ctxt1)
            if isinstance(ctxt2, PyCtxt):
                ctxt2 = self.csp.recrypt_array(ctxt2)
            result = ctxt1 * ctxt2
            self.recrypt_count += 2

            assert self.csp.get_noise_level(result) > 0, "noise level is zero in multiplication after recrypt. We are doomed"

        result = ~result
        return result

    def calc_encrypted_array_sum(self, arr, arr_len: int):

        # Fast algorithm for summing encrypted arrays
        # The general idea here is to split the array into 2 using shifts
        # and then sum the 2 arrays (original and shifted).
        # The only caveat here is cases where there is a remainder of the array_size/2.
        # In this case we need to add an extra number which is always located at shift_val*2 + 1
        # so if we shift the array again by the same value and add the first number, we will get
        # what we are looking for
        # A the end of the loop, add these numbers to the total sum

        # for debug
        #summed_arr = self.csp.sum_array(arr)
        #dec_arr = self.csp.decrypt_arr(summed_arr)
        #print("dec_arr old algorithm: ", dec_arr[0])

        summed_arr = arr + 0
        temp_add_arr = 0
        remainder = False

        while arr_len > 1:
            if (arr_len % 2) > 0:
                remainder = True
            arr_len = arr_len // 2
            shift_val = arr_len
            #print("arr_len: ", arr_len)
            shifted = summed_arr << shift_val
            #dec_arr = self.csp.decrypt_arr(summed_arr)
            #print("summed: ", dec_arr[0:20])
            #dec_arr = self.csp.decrypt_arr(shifted)
            #print("shiftd: ", dec_arr[0:20])
            summed_arr = self.safe_math(summed_arr, shifted, '+')
            #dec_arr = self.csp.decrypt_arr(summed_arr)
            #print("---------------------------------------")
            #print("summed: ", dec_arr[0:20])
            #print("---------------------------------------")
            if remainder:
                temp_add_arr = self.safe_math(temp_add_arr, (shifted << arr_len), '+')
                remainder = False

        summed_arr = self.safe_math(summed_arr, temp_add_arr, '+')
        #dec_arr = self.csp.decrypt_arr(summed_arr)
        #print("last addition: ", dec_arr[0:20])

        mask_arr = np.array([1])
        encoded_mask_arr = self.csp.encode_array(mask_arr)
        #new_sum = self.safe_mul(summed_arr, encoded_mask_arr)
        new_sum = self.safe_math(summed_arr, encoded_mask_arr, "*")

        #dec_arr = self.csp.decrypt_arr(new_sum)
        #print("dec_arr new algorithm: ", dec_arr[0])

        return new_sum

    def enc_array_same_num(self, enc_num, size):
        """
        create an encrypted array where all cells contain the same number
        @param enc_num: the number to duplicate
        @param size: array size
        @return: encrypted array
        """
        i = 1
        num_array = enc_num + 0
        next_array_size = 2
        while (i < size) and (next_array_size <= size):
            num_array = self.safe_math(num_array, num_array >> i, '+')
            i *= 2
            next_array_size *= 2
        # if the size is a power of 2, we're done. If not, need to add to the missing cells
        if size != 0 and (size & (size-1) == 0):
            return num_array
        else:
            while i < size:
                num_array = self.safe_math(num_array, enc_num >> i, '+')
                i += 1

        return num_array

    def safe_power_of(self, enc_arr, power):
        result = enc_arr ** power
        if self.csp.get_noise_level(result) == 0:
            enc_arr = self.csp.recrypt_array(enc_arr)
            result = enc_arr ** power
        ~result # re-liniarize after power as it seems this does not happen automatically
        return result

    def noise_level_assert(self, arr):
        lvl = self.csp.get_noise_level(arr)
        assert lvl > 0, "noise level is 0"

    def adapted_site_step(self, ages, meth_vals_list, sum_ri_square):
        """
        The EPM site step algorithm. This step calculates beta = (XtX)^-1 XtY using the conclusions from
        lemma 3 / corollary 1 in https://bmcgenomics.biomedcentral.com/articles/10.1186/s12864-020-6606-0
        The lemma defines 2 nXmn expanded diagonal matrices
        assuming m=2 and n=3, a matching nXmn expanded  diagonal matrix would look as following:
        x_1 x_2   0    0    0    0
        0    0   x_1  x_2   0    0
        0    0    0    0   x_1  x_2


        @return: the model parameters: s0 and rates
        """

        print("Process", os.getpid(), "is executing the site step")
        # create empty encrypted arrays to store the s0 and rate values
        dummy_zero = np.zeros(1, dtype=np.int64)
        rates = self.csp.encrypt_array(dummy_zero)
        s0_vals = self.csp.encrypt_array(dummy_zero)

        sum_ri_square_arr = self.enc_array_same_num(sum_ri_square, self.m)

        #print("calc sigma_t starting at: ", time.perf_counter())
        sigma_t = self.calc_encrypted_array_sum(ages, self.m)
        
        #print("calc sigma_t ended at: ", time.perf_counter())
        square_ages = self.safe_power_of(ages, 2)
        sigma_t_square = self.calc_encrypted_array_sum(square_ages, self.m)
        
        gamma_denom = self.safe_power_of(sigma_t, 2)
        gamma_denom -= self.safe_mul(self.m, sigma_t_square)

        minus_m_arr = self.csp.encrypt_array(np.array([-1*self.m], dtype=np.int64))
        minus_m_arr_enc = self.enc_array_same_num(minus_m_arr, self.m)
        
        rates_assist_arr = self.safe_mul(minus_m_arr_enc, ages)
        
        s0_assist_arr = ages

        all_sigma_t_arr = self.enc_array_same_num(sigma_t, self.m)
        
        all_sigma_t_square_arr = self.enc_array_same_num(sigma_t_square, self.m)
        
        # in order to avoid the need to build the expanded diagonal matrices
        # we create a vector with the x_0....x_m values and multiply the Y vector by n copies of this vector
        rates_assist_arr = self.safe_math(rates_assist_arr, all_sigma_t_arr, '+')
        
        rates_assist_arr = self.safe_mul(rates_assist_arr, sum_ri_square_arr)
        
        s0_assist_arr = self.safe_mul(s0_assist_arr, all_sigma_t_arr)
        s0_assist_arr = self.safe_math(s0_assist_arr, all_sigma_t_square_arr, '-')

        enc_array_size = self.csp.get_enc_n() // 2
        elements_in_vector = enc_array_size // self.m
        elements_in_vector = min(self.n, elements_in_vector)

        meth_vals_vector_num = 0
        for meth_vals in meth_vals_list:
            for i in range(0, elements_in_vector):
                shifted_vals = meth_vals << (i*self.m)
                r_mult_assist = self.safe_mul(rates_assist_arr, shifted_vals)
                rate = self.calc_encrypted_array_sum(r_mult_assist, self.m)
                rates = self.safe_math(rates, (rate >> (i+meth_vals_vector_num*elements_in_vector)), '+')
                s0_mult_assist = self.safe_mul(s0_assist_arr, shifted_vals)
                s0 = self.calc_encrypted_array_sum(s0_mult_assist, self.m)
                s0_vals = self.safe_math(s0_vals, (s0 >> (i+meth_vals_vector_num*elements_in_vector)), '+')
                
            meth_vals_vector_num += 1

        return rates, s0_vals, gamma_denom

    def adapted_time_step_transposed(self, rates, s0_vals, gamma, inference=False):
        print("Process", os.getpid(), "is executing the time step")
        if inference:
            meth_val_list = self.transposed_test_meth_val_list
            m = self.test_m
        else:
            meth_val_list = self.transposed_meth_val_list
            m = self.m

        dummy_zero = np.zeros(1, dtype=np.int64)
        ages = self.csp.encrypt_array(dummy_zero)
        enc_array_size = self.csp.get_enc_n() // 2
        calc_assist_s0 = s0_vals + 0
        calc_assist_rates = rates + 0
        num_of_ages_in_table = min(enc_array_size // self.n, m)
        gamma_array = self.enc_array_same_num(gamma, enc_array_size)

        # create an array of s0 and rate values times the number of individuals that can fit in an encrypted array
        # this will make calculations quicker
        for i in range(1, num_of_ages_in_table):
            calc_assist_s0 = self.safe_math(calc_assist_s0, (s0_vals >> (i*self.n)), '+')
            calc_assist_rates = self.safe_math(calc_assist_rates, (rates >> (i*self.n)), '+')

        age_index = 0
        for enc_meth_vals in meth_val_list:
            temp_meth_vals = self.safe_math(enc_meth_vals, gamma_array, '*')
            temp_meth_vals = self.safe_math(temp_meth_vals, calc_assist_s0, '-')
            temp_meth_vals = self.safe_math(temp_meth_vals, calc_assist_rates, '*')
            for j in range(num_of_ages_in_table):
                age_sum = self.calc_encrypted_array_sum(temp_meth_vals << (j*self.n), self.n)
                ages = self.safe_math(ages, age_sum >> age_index, '+')
                age_index += 1

            if age_index == m:
                break

        # now we just need to calculate the denominator for the site step
        # which is sum(r_i^2)
        ri_squared = self.safe_power_of(rates, 2)
        sum_ri_squared = self.calc_encrypted_array_sum(ri_squared, self.n)

        return ages, sum_ri_squared

    def calc_model(self):
        """
        model calculation
        @return: ages, rates and s0 values calculated by the 2 steps
        """
        sum_ri_squared = 1
        rates = 0
        s0_vals = 0
        gamma_denom = 0
        predicted_ages = 0
        for i in range(self.rounds):
            rates, s0_vals, gamma_denom = self.adapted_site_step(self.ages, self.meth_val_list, sum_ri_squared)
            new_ages, sum_ri_squared = self.adapted_time_step_transposed(rates, s0_vals, gamma_denom)
            self.ages = new_ages

        # now perform the inference on the test data
        if self.transposed_test_meth_val_list is not None:
            predicted_ages, _ = self.adapted_time_step_transposed(rates, s0_vals, gamma_denom, inference=True)
        return self.ages, sum_ri_squared, rates, s0_vals, gamma_denom, predicted_ages

