import math
from multiprocessing import cpu_count, Queue, Process
import numpy as np
from CRT.CrtVector import CrtVector
from Pyfhel import PyCtxt
import os
import queue
'''
Machine Learning Engine (MLE) implementation
The MLE receives the encrypted input data from the DO and calculates the ages
'''


class MLE:

    prime_index = None
    def __init__(self, csp, rounds):

        self.m = 0
        self.n = 0
        self.csp = csp
        self.rounds = rounds
        self.crt_vector = CrtVector()

    def get_data_from_DO(self, crt_vector: CrtVector, m: int, n: int):

        self.crt_vector.add_vector(crt_vector)
        self.m += m
        self.n += n
        print("MLE: got data from DO")

    def __safe_math_run_op__(self, ctxt1, ctxt2, operation):

        match operation:
            case '+':
                result = ctxt1 + ctxt2
            case '-':
                result = ctxt1 - ctxt2
            case '*':
                result = ctxt1 * ctxt2
                result = ~result
            case '**':
                result = ctxt1 ** ctxt2
                result = ~result
            case _:
                assert False, "Operation " + operation + " not supported"

        return result

    def safe_math(self, ctxt1, ctxt2, operation):
        result = self.__safe_math_run_op__(ctxt1, ctxt2, operation)
        if isinstance(result, PyCtxt):
            lvl = self.csp.get_noise_level(self.prime_index, result)
            if lvl == 0:
                if isinstance(ctxt1, PyCtxt):
                    ctxt1 = self.csp.recrypt_array(self.prime_index, ctxt1)
                if isinstance(ctxt2, PyCtxt):
                    ctxt2 = self.csp.recrypt_array(self.prime_index, ctxt2)

                result = self.__safe_math_run_op__(ctxt1, ctxt2, operation)
                lvl = self.csp.get_noise_level(self.prime_index, result)

                assert lvl > 0, "Noise level is 0 even after recrypt"

        return result

    def calc_encrypted_array_sum(self, arr, arr_len: int):

        # Fast algorithm for summing encrypted arrays
        # The general idea here is to split the array into 2 using shifts
        # and then sum the 2 arrays (original and shifted).
        # The only caveat here is cases where there is a remainder of the array_size/2.
        # In this case we need to add an extra number which is always located at shift_val*2 + 1
        # so if we shift the array again by the same value and add the first number, we will get
        # the required number
        # At the end of the loop, add these numbers to the total sum


        summed_arr = arr + 0
        temp_add_arr = 0
        remainder = False

        while arr_len > 1:
            if (arr_len % 2) > 0:
                remainder = True
            arr_len = arr_len // 2
            shift_val = arr_len
            shifted = summed_arr << shift_val
            summed_arr = self.safe_math(summed_arr, shifted, '+')

            if remainder:
                temp_add_arr = self.safe_math(temp_add_arr, (shifted << arr_len), '+')
                remainder = False

        summed_arr = self.safe_math(summed_arr, temp_add_arr, '+')

        mask_arr = np.array([1])
        encoded_mask_arr = self.csp.encode_array(self.prime_index, mask_arr)
        new_sum = self.safe_math(summed_arr, encoded_mask_arr, "*")

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

    def noise_level_assert(self, arr):
        lvl = self.csp.get_noise_level(self.prime_index, arr)
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
        print("Process", os.getpid(), "is executing the site step for prime index: ", self.prime_index)

         # create empty encrypted arrays to store the s0 and rate values
        dummy_zero = np.zeros(1, dtype=np.int64)
        rates = self.csp.encrypt_array(dummy_zero, self.prime_index)
        s0_vals = self.csp.encrypt_array(dummy_zero, self.prime_index)

        sum_ri_square_arr = self.enc_array_same_num(sum_ri_square, self.m)

        sigma_t = self.calc_encrypted_array_sum(ages, self.m)

        square_ages = self.safe_math(ages, 2, "**")
        sigma_t_square = self.calc_encrypted_array_sum(square_ages, self.m)

        lambda_inv = self.safe_math(sigma_t, 2, "**")
        lambda_inv -= self.safe_math(self.m, sigma_t_square, '*')

        minus_m_arr = self.csp.encrypt_array(np.array([-1*self.m], dtype=np.int64), self.prime_index)
        minus_m_arr_enc = self.enc_array_same_num(minus_m_arr, self.m)
        
        rates_assist_arr = self.safe_math(minus_m_arr_enc, ages, '*')
        
        s0_assist_arr = ages
        all_sigma_t_arr = self.enc_array_same_num(sigma_t, self.m)       
        all_sigma_t_square_arr = self.enc_array_same_num(sigma_t_square, self.m)
        
        # in order to avoid the need to build the expanded diagonal matrices
        # we create a vector with the x_0....x_m values and multiply the Y vector by n copies of this vector
        rates_assist_arr = self.safe_math(rates_assist_arr, all_sigma_t_arr, '+')
        
        rates_assist_arr = self.safe_math(rates_assist_arr, sum_ri_square_arr, '*')
        
        s0_assist_arr = self.safe_math(s0_assist_arr, all_sigma_t_arr, '*')

        s0_assist_arr = self.safe_math(s0_assist_arr, all_sigma_t_square_arr, '-')

        enc_array_size = self.csp.get_enc_n() // 2
        elements_in_vector = enc_array_size // self.m
        elements_in_vector = min(self.n, elements_in_vector)

        meth_vals_vector_num = 0
        for meth_vals in meth_vals_list:
            for i in range(0, elements_in_vector):
                shifted_vals = meth_vals << (i*self.m)
                r_mult_assist = self.safe_math(rates_assist_arr, shifted_vals, '*')
                rate = self.calc_encrypted_array_sum(r_mult_assist, self.m)
                rates = self.safe_math(rates, (rate >> (i+meth_vals_vector_num*elements_in_vector)), '+')
                s0_mult_assist = self.safe_math(s0_assist_arr, shifted_vals, '*')
                s0 = self.calc_encrypted_array_sum(s0_mult_assist, self.m)
                s0_vals = self.safe_math(s0_vals, (s0 >> (i+meth_vals_vector_num*elements_in_vector)), '+')
                
            meth_vals_vector_num += 1

        return rates, s0_vals, lambda_inv

    def adapted_time_step_transposed(self, transposed_meth_val_list, rates, s0_vals, lambda_denom):
        print("Process", os.getpid(), "is executing the time step for prime index: ", self.prime_index)

        dummy_zero = np.zeros(1, dtype=np.int64)
        ages = self.csp.encrypt_array(dummy_zero, self.prime_index)
        enc_array_size = self.csp.get_enc_n() // 2
        calc_assist_s0 = s0_vals + 0
        calc_assist_rates = rates + 0
        num_of_ages_in_table = min(enc_array_size // self.n, self.m)
        lambda_denom_array = self.enc_array_same_num(lambda_denom, enc_array_size)

        # create an array of s0 and rate values times the number of individuals that can fit in an encrypted array
        # this will make calculations quicker
        for i in range(1, num_of_ages_in_table):
            calc_assist_s0 = self.safe_math(calc_assist_s0, (s0_vals >> (i*self.n)), '+')
            calc_assist_rates = self.safe_math(calc_assist_rates, (rates >> (i*self.n)), '+')

        age_index = 0
        for enc_meth_vals in transposed_meth_val_list:
            temp_meth_vals = self.safe_math(enc_meth_vals, lambda_denom_array, '*')
            temp_meth_vals = self.safe_math(temp_meth_vals, calc_assist_s0, '-')
            temp_meth_vals = self.safe_math(temp_meth_vals, calc_assist_rates, '*')

            for j in range(num_of_ages_in_table):
                age_sum = self.calc_encrypted_array_sum(temp_meth_vals << (j*self.n), self.n)
                ages = self.safe_math(ages, age_sum >> age_index, '+')
                age_index += 1

            if age_index == self.m:
                break

        # now we just need to calculate the denominator for the site step
        # which is sum(r_i^2)
        ri_squared = self.safe_math(rates, 2, "**")
        sum_ri_squared = self.calc_encrypted_array_sum(ri_squared, self.n)

        # 2 recrypt opretaion to handle multiplication depth issues
        ages = self.csp.recrypt_array(self.prime_index, ages)
        sum_ri_squared = self.csp.recrypt_array(self.prime_index, sum_ri_squared)
        return ages, sum_ri_squared

    def calc_process(self, calc_per_prime_queue, results_queue):
        while True:
            try:
                rates = None
                s0_vals = None
                #crt_set = calc_per_prime_queue.get_nowait()
                crt_vec_index = calc_per_prime_queue.get_nowait()
                crt_set = self.crt_vector.get(crt_vec_index)

                prime_index = crt_set.prime_index
                self.prime_index = prime_index
                ages = crt_set.enc_ages

                meth_val_list = crt_set.enc_meth_val_list
                transposed_meth_val_list = crt_set.enc_transposed_meth_val_list
                sum_ri_squared = 1
                for i in range(self.rounds):
                    rates, s0_vals, lambda_inv = self.adapted_site_step(ages, meth_val_list, sum_ri_squared)
                    ages, sum_ri_squared = self.adapted_time_step_transposed(transposed_meth_val_list,
                                                                             rates, s0_vals, lambda_inv)
                    print("Process:", os.getpid(), " MLE: iteration ", i, "is done")

                crt_set.t_num = ages
                crt_set.t_denom = sum_ri_squared
                crt_set.rates = rates
                crt_set.s0_vals = s0_vals
                results_queue.put(crt_set)

            except queue.Empty:
                break

    def calc_model_multi_process(self):

        calc_per_prime_queue = Queue()
        results_queue = Queue()
        processes = []
        num_of_cores = cpu_count()
        num_of_crt_elements = len(self.crt_vector)

        for i in range(num_of_crt_elements):
            #calc_per_prime_queue.put(self.crt_vector.get(i))
            calc_per_prime_queue.put(i)

        # set number of processes according to the minimum between the number of cores
        # or the number of elements in the crt vector.
        # need to leave one core available for the result collection process
        num_of_processes = min(num_of_cores - 1, num_of_crt_elements)

        print("MLE: spawning processes")
        for process in range(num_of_processes):
            p = Process(target=self.calc_process, args=[calc_per_prime_queue, results_queue])
            processes.append(p)
            p.start()
            print("MLE: process started")

        # now get the results
        i = 0
        while i < num_of_crt_elements:
            if not results_queue.empty():
                crt_set = results_queue.get()
                self.crt_vector.copy_calc_results(crt_set)
                i += 1

        # verify that all processes completed
        for p in processes:
            p.join()

        print("MLE: Calculation complete")


