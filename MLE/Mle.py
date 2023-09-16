import math

import numpy as np
import time
from Pyfhel import Pyfhel, PyCtxt
from inspect import getframeinfo, stack

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
        self.m = None
        self.n = None
        self.recrypt_count = 0
        self.FILE_COUNTER = 0
        self.csp = csp
        rand_mask = np.random.randint(1, high=RANDOM_MATRIX_MAX_VAL, size=(csp.get_enc_n()//2))
        self.rand_mask = csp.encrypt_array(rand_mask)

    def get_data_from_DO(self, meth_val_list, ages, m, n):

        self.meth_val_list = meth_val_list
        self.ages = ages
        self.m = m
        self.n = n

        '''
        DO_to_MLE_file = open("network/do_to_mle_{}.bin".format(self.FILE_COUNTER), "wb")
        A.tofile(DO_to_MLE_file, sep=',')
        B.tofile(DO_to_MLE_file, sep=',')
        meth_vals.tofile(DO_to_MLE_file, sep=',')
        DO_to_MLE_file.write(bytearray(rank_A))
        DO_to_MLE_file.close()
        self.FILE_COUNTER += 1
        '''

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
        new_sum = self.safe_mul(summed_arr, encoded_mask_arr)

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

        print("Executing site step")
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
        
        # dec_rates_assist_arr = self.csp.decrypt_arr(rates_assist_arr)
        # dec_ages = self.csp.decrypt_arr(ages)
        # dec_ages_m = self.csp.decrypt_arr(self.m * ages)

        s0_assist_arr = ages

        tic = time.perf_counter()
        #print("calc all_sigma arrays starting at: ", tic)
        all_sigma_t_arr = self.enc_array_same_num(sigma_t, self.m)
        
        all_sigma_t_square_arr = self.enc_array_same_num(sigma_t_square, self.m)
        
        # for debug: dec_sigma_t = self.csp.decrypt_arr(all_sigma_t_arr)
        # for debug: dec_sigma_t_s = self.csp.decrypt_arr(all_sigma_t_square_arr)

        # in order to avoid the need to build the expanded diagonal matrices
        # we create a vector with the x_0....x_m values and multiply the Y vector by n copies of this vector
        rates_assist_arr = self.safe_math(rates_assist_arr, all_sigma_t_arr, '+')
        
        # for debug: dec_rates_assist = self.csp.decrypt_arr(rates_assist_arr)
        # for debug: dec_all_sigma_t_arr = self.csp.decrypt_arr(all_sigma_t_arr)
        rates_assist_arr = self.safe_mul(rates_assist_arr, sum_ri_square_arr)
        
        #dec_rates_assist_arr = self.csp.decrypt_arr(rates_assist_arr)
        #print("meir: ", dec_rates_assist_arr)

        s0_assist_arr = self.safe_mul(s0_assist_arr, all_sigma_t_arr)
        s0_assist_arr = self.safe_math(s0_assist_arr, all_sigma_t_square_arr, '-')
        
        #print("calc rates and s0 values starting at: ", time.perf_counter())
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


    def adapted_time_step(self, rates, s0_vals, meth_vals_list, gamma):

        # calc the matrix  S = (S_ij - s0_i)*r_i
        # the main issue here is that S_ij is represented as long vectors which cannot be transposed as
        # the encrypted vectors do not support transposing
        # this means that each n elements in S_ij need to be multiplied by the same r_i and the same s0_i needs to be
        # reduced from it

        # in order to achieve this, we need to create 2 table sets: one for the s0 and one for the rate
        # each table set consists of one or more tables. The number of tables is determined according to the
        # value of n*m. If n*m is larger than the table size provided by the encryption, more than a single table
        # is required to hold the values.
        # Each table set will contain m copies of each of the n s0/rate values
        # This will make it easier to perform the subtraction and multiplication operations with relation to Sij

        print("Executing time step")

        enc_array_size = self.csp.get_enc_n() // 2 # the array is represented
        mask = np.zeros(enc_array_size, dtype=np.int64)
        enc_zeros = self.csp.encrypt_array(mask)
        mask[0] = 1
        encoded_mask = self.csp.encode_array(mask)

        calc_assist_s0 = []
        calc_assist_rates = []
        #table_size = self.n * self.m
        num_of_assist_tables = len(meth_vals_list)
        print("num of assist tables: ", num_of_assist_tables)
        iterations = min(self.m, (enc_array_size // self.m))
        # the addition of 0 causes the creation of a new encrypted array.
        # without it, the variable on the left side of the = will just be a pointer to the one on the right

        for i in range(num_of_assist_tables):
            calc_assist_s0.append(enc_zeros + 0)
            calc_assist_rates.append(enc_zeros + 0)

        # at the end of this loop the calc_assist_s0 and calc_assist_rate table sets should have an S0/rate value
        # in each i*m cell
        set_index = 0
        iter_num = 0
        for i in range(0, self.n):
            assert set_index < num_of_assist_tables, "Table set index is {} higher than number of tables in the set {}".format(set_index, num_of_assist_tables)

            temp_s0_vals = self.safe_math(s0_vals << i, encoded_mask, '*')
            temp_rates = self.safe_math(rates << i, encoded_mask, '*')
            calc_assist_s0[set_index] = self.safe_math(calc_assist_s0[set_index], (temp_s0_vals >> (iter_num*self.m)), '+')
            calc_assist_rates[set_index] = self.safe_math(calc_assist_rates[set_index], (temp_rates >> (iter_num*self.m)), '+')
            iter_num += 1
            if iter_num == iterations:
                iter_num = 0
                set_index += 1

        #now duplicate it m times
        for i in range(num_of_assist_tables):
            temp_s0 = calc_assist_s0[i] + 0
            temp_rates = calc_assist_rates[i] + 0
            for j in range(1, self.m):
                calc_assist_s0[i] = self.safe_math(calc_assist_s0[i], (temp_s0 >> j), '+')
                calc_assist_rates[i] = self.safe_math(calc_assist_rates[i], (temp_rates >> j), '+')

        ''' validation check
        all_s0 = self.csp.decrypt_arr(s0_vals)
        set_index = 0
        iter_index = 0
        decrypted = self.csp.decrypt_arr(calc_assist_s0[set_index])

        for i in range(self.n):
            val = all_s0[i]
            print("now checking: ", val)
            for j in range(self.m):
                if decrypted[j+iter_index*self.m] != val:
                    print("val is ", val, " while dec is: ", decrypted[j+iter_index*self.m])
            iter_index += 1
            if iter_index == iterations:
                iter_index = 0
                set_index += 1
                if set_index < num_of_assist_tables:
                    decrypted = self.csp.decrypt_arr(calc_assist_s0[set_index])

        print("We're done!!")
        '''

        # create an array full of gamma values in order to easily multiply each Sij by gamma
        gamma_array = self.enc_array_same_num(gamma, enc_array_size)
        

        tic = time.perf_counter()
        #print("calc new ages starting at: ", tic)
        new_ages = []
        for i, meth_vals in enumerate(meth_vals_list):
            meth_vals_gamma = self.safe_mul(meth_vals, gamma_array)

            
            # now we need to calc the numerator of the time step.
            # which is: sum(r_i(meth_vals_gamma - s^0_i))
            # first lets calculate (meth_vals_gamma - s^0_i)
            p = self.safe_math(meth_vals_gamma, calc_assist_s0[i], '-')
            
            # now lets multiply by r_i
            r_p = self.safe_mul(p, calc_assist_rates[i])
            
            #dec_r_p = self.csp.decrypt_arr(r_p)
            # now the tricky part - calculate the sums for each tj
            # we will use the mask to sum each r_p_j*m
            # this should give us the required sum for each tj

            new_ages.append(r_p + 0)

            for j in range(1, iterations):
                new_ages[i] = self.safe_math(new_ages[i], (r_p << j*self.m), '+')

            
            mask = np.ones(self.m, dtype=np.int64)
            encoded_mask = self.csp.encode_array(mask)
            new_ages[i] = self.safe_mul(new_ages[i], encoded_mask)
            
            #encrypted_mask = self.csp.encrypt_array(mask)
            #new_ages = self.csp.sum_array(self.safe_mul(r_p, encrypted_mask))

        #print("calc new ages took: ", time.perf_counter()-tic)

        # now we just need to calculate the denominator for the site step
        # which is sum(r_i^2)
        ri_squared = self.safe_power_of(rates, 2)
        
        sum_ri_squared = self.calc_encrypted_array_sum(ri_squared, self.n)
        
        #dec_sum_ri_squared = self.csp.decrypt_arr(sum_ri_squared)[0]
        #print("dec sum ri squared 1: ", dec_sum_ri_squared)

        for i in range(1, len(new_ages)):
            new_ages[0] = self.safe_math(new_ages[0], new_ages[i], '+')
        return new_ages[0], sum_ri_squared


    def calc_model(self):
        """
        model calculation
        runs the site step and time step once
        @return: ages, rates and s0 values calculated by the 2 steps
        """
        iter = 2
        sum_ri_squared = 1
        for i in range(iter):
            rates, s0_vals, gamma_denom = self.adapted_site_step(self.ages, self.meth_val_list, sum_ri_squared)
            #rates = self.csp.encrypt_array(np.array([1]))
            #s0_vals = self.csp.encrypt_array(np.array([1]))
            #gamma_denom = self.csp.encrypt_array(np.array([1]))
            new_ages, sum_ri_squared = self.adapted_time_step(rates, s0_vals, self.meth_val_list, gamma_denom)
            #dec_ages = self.csp.decrypt_arr(new_ages)
            #dec_sum_ri_squared = self.csp.decrypt_arr(sum_ri_squared)
            self.ages = new_ages

        return self.ages, sum_ri_squared

