import numpy as np
import time
from Pyfhel import Pyfhel, PyCtxt

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

    csp = None
    ages = None
    meth_val_list = None
    m = None
    n = None
    rand_mask = None
    recrypt_count = 0

    FILE_COUNTER = 0

    def __init__(self, csp):
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
        ctxt += self.rand_mask
        lvl = self.csp.get_noise_level(ctxt)
        assert lvl > 0, "reached 0 noised budget"
        '''
        if lvl == 0:
            print("Recrypting")
            ctxt = self.csp.recrypt_array(ctxt)
            self.recrypt_count += 1
        '''
        ctxt -= self.rand_mask
        return ctxt

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

        # mask before sending to CSP for noise level check
        if isinstance(ctxt1, PyCtxt):
            ctxt1 = self.check_noise_lvl(ctxt1)
        if isinstance(ctxt2, PyCtxt):
            ctxt2 = self.check_noise_lvl(ctxt2)

        result = ctxt1 * ctxt2
        result = ~result
        return result

    def calc_encrypted_array_sum(self, arr: np.ndarray, arr_len: int):
        """
        sum an encrypted array
        @param arr: the array to sum
        @param arr_len:  the length of the array
        @return: an encrypted array with the sum in the first cell
        """
        sum_arr = arr
        dec_arr = self.csp.decrypt_arr(arr)
        shift = arr_len
        while shift > 1:
            shift = shift // 2
            add_arr = sum_arr << shift
            sum_arr = sum_arr + add_arr

        mask_arr = np.array([1])
        encoded_mask_arr = self.csp.encode_array(mask_arr)
        new_sum = self.safe_mul(sum_arr, encoded_mask_arr)
        return new_sum

    def enc_array_same_num1(self, enc_num, size):
        '''
        an inefficient implementation. use the one below for faster performance
        This one can probably be deleted
        '''
        expected_copies = 1
        num_array = enc_num + 0
        arr_to_duplicate = enc_num + 0
        i = 1
        while True:
            num_array += arr_to_duplicate >> i
            arr_to_duplicate = num_array
            expected_copies += i
            i *= 2
            if expected_copies > size:
                break



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
            num_array += num_array >> i
            i *= 2
            next_array_size *= 2
        # if the size is a power of 2, we're done. If not, need to add to the missing cells
        if size != 0 and (size & (size-1) == 0):
            return num_array
        else:
            while i < size:
                num_array += enc_num >> i
                i += 1

        return num_array

    def adapted_site_step(self, m: int, n: int, ages, meth_vals_list, sum_ri_square):
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

        # create empty encrypted arrays to store the s0 and rate values
        dummy_zero = np.zeros(1, dtype=np.int64)
        rates = self.csp.encrypt_array(dummy_zero)
        s0_vals = self.csp.encrypt_array(dummy_zero)
        all_sigma_t_arr = self.csp.encrypt_array(dummy_zero)
        all_sigma_t_square_arr = self.csp.encrypt_array(dummy_zero)

        sum_ri_square_arr = self.enc_array_same_num(sum_ri_square, n)

        print("calc sigma_t starting at: ", time.perf_counter())
        # sigma_t = self.calc_encrypted_array_sum(ages, m)
        sigma_t = self.csp.sum_array(ages)
        # sigma_t_dec = self.csp.decrypt_arr(sigma_t)
        print("calc sigma_t ended at: ", time.perf_counter())
        square_ages = ages ** 2
        ~square_ages # re-liniarize after power as it seems this does not happen automatically
        # sigma_t_square = self.calc_encrypted_array_sum(square_ages, m)
        sigma_t_square = self.csp.sum_array(square_ages)
        print("calc sigma_t_square ended at: ", time.perf_counter())
        gamma_denom = sigma_t ** 2
        gamma_denom = ~gamma_denom
        gamma_denom -= self.safe_mul(m, sigma_t_square)
        rates_assist_arr = -1 * m * ages
        s0_assist_arr = ages

        tic = time.perf_counter()
        print("calc all_sigma arrays starting at: ", tic)
        all_sigma_t_arr = self.enc_array_same_num(sigma_t, m)
        all_sigma_t_square_arr = self.enc_array_same_num(sigma_t_square, m)

        # for debug: dec_sigma_t = self.csp.decrypt_arr(all_sigma_t_arr)
        # for debug: dec_sigma_t_s = self.csp.decrypt_arr(all_sigma_t_square_arr)

        #for i in range(m):
        #    all_sigma_t_arr += (sigma_t >> i)
        #    all_sigma_t_square_arr += (sigma_t_square >> i)
        print("calc all_sigma arrays took: ", time.perf_counter() - tic)

        # in order to avoid the need to build the expanded diagonal matrices
        # we create a vector with the x_0....x_m values and multiply the Y vector by n copies of this vector
        rates_assist_arr += all_sigma_t_arr
        # for debug: dec_rates_assist = self.csp.decrypt_arr(rates_assist_arr)
        # for debug: dec_all_sigma_t_arr = self.csp.decrypt_arr(all_sigma_t_arr)
        rates_assist_arr = self.safe_mul(rates_assist_arr, sum_ri_square_arr)

        s0_assist_arr = self.safe_mul(s0_assist_arr, all_sigma_t_arr)
        s0_assist_arr -= all_sigma_t_square_arr

        print("calc rates and s0 values starting at: ", time.perf_counter())
        # calculate the rate and s0 values
        enc_array_size = self.csp.get_enc_n() // 2
        elements_in_vector = enc_array_size // m

        meth_val_count = 0
        for meth_vals in meth_vals_list:
            for i in range(0, elements_in_vector):
                shifted_vals = meth_vals << (i*m)
                shift_dec =  self.csp.decrypt_arr(shifted_vals)
                mult_assist = self.safe_mul(rates_assist_arr, shifted_vals)
                # for debug: dec_mult_assist = self.csp.decrypt_arr(mult_assist)
                # r_val = self.calc_encrypted_array_sum(mult_assist, m)
                r_val = self.csp.sum_array(mult_assist)
                # for debug: dec_r_val = self.csp.decrypt_arr(r_val)
                #print("r_val: ", self.csp.decrypt_arr(r_val))
                mult_assist = self.safe_mul(s0_assist_arr, shifted_vals)
                # s0_val = self.calc_encrypted_array_sum(mult_assist, m)
                s0_val = self.csp.sum_array(mult_assist)
                #now need to add each value to the rates
                rate_s0_shift = (meth_val_count * elements_in_vector + i)
                rates = rates + (r_val >> rate_s0_shift)
                s0_vals += (s0_val >> rate_s0_shift)
            meth_val_count += 1
        print("calc rates and s0 values ending at: ", time.perf_counter())


        ''' for debug 
        dec_rates = self.csp.decrypt_arr(rates)
        dec_s0 = self.csp.decrypt_arr(s0_vals)
        # return dec_rates, dec_s0
        '''
        return rates, s0_vals, gamma_denom

    def adapted_time_step(self, rates, s0_vals, meth_vals_list, n, m, gamma):

        # calc the matrix  S = (S_ij - s0_i)*r_i
        # the main issue here is that S_ij is represented as long vectors which cannot be transposed as
        # the encrypted vectors do not support transposing
        # this means that each n elements in S_ij need to be multiplied by the same r_i

        enc_array_size = self.csp.get_enc_n() // 2 # the array is represented
        # the addition of 0 causes the creation of a new encrypted array.
        # without it, the variable on the left side of the = will just be a pointer to the one on the right
        calc_assist_s0 = s0_vals + 0
        calc_assist_rates = rates + 0
        iterations = min(m, (enc_array_size // m))-1

        for i in range(1, iterations):
            calc_assist_s0 += (s0_vals >> (i*m - 1))
            calc_assist_rates += (rates >> (i*m - 1))

        mask = np.zeros(enc_array_size, dtype=np.int64)
        for i in range(iterations):
            mask[i*m] = 1
        encoded_mask = self.csp.encode_array(mask)
        calc_assist_s0 = self.safe_mul(calc_assist_s0, encoded_mask)
        calc_assist_rates = self.safe_mul(calc_assist_rates, encoded_mask)

        separated_s0 = calc_assist_s0 + 0
        separated_rates = calc_assist_rates + 0

        for i in range(1, m):
            calc_assist_s0 += (separated_s0 >> i)
            calc_assist_rates += (separated_rates >> i)

        # create an array full of gamma values in order to easily multiply each Sij by gamma
        gamma_array = self.enc_array_same_num(gamma, enc_array_size)
        #dec_gamma_array = self.csp.decrypt_arr(gamma_array)

        tic = time.perf_counter()
        print("calc new ages starting at: ", tic)
        for meth_vals in meth_vals_list:
            meth_vals_gamma = self.safe_mul(meth_vals, gamma_array)

            # now we need to calc the numerator of the time step.
            # which is: sum(r_i(meth_vals_gamma - s^0_i))
            # first lets calculate (meth_vals_gamma - s^0_i)
            p = meth_vals_gamma - calc_assist_s0
            # now lets multiply by r_i
            r_p = self.safe_mul(p, calc_assist_rates)
            # now the tricky part - calculate the sums for each tj
            # we will use the mask to sum each r_p_j*m
            # this should give us the required sum for each tj

            new_ages = r_p + 0

            for i in range(1, iterations):
                new_ages += (r_p << i*m)

            mask = np.ones(m, dtype=np.int64)
            encoded_mask = self.csp.encode_array(mask)
            new_ages = self.safe_mul(new_ages, encoded_mask)
            #encrypted_mask = self.csp.encrypt_array(mask)
            #new_ages = self.csp.sum_array(self.safe_mul(r_p, encrypted_mask))
            '''
            
            for i in range(1, m):
                # new_ages += ((self.csp.sum_array(self.safe_mul(r_p, (encrypted_mask >> i)))) >> i)
                new_ages += ((self.calc_encrypted_array_sum(self.safe_mul(r_p, (encrypted_mask >> i)), enc_array_size)) >> i)
            '''
        print("calc new ages took: ", time.perf_counter()-tic)

        # now we just need to calculate the denominator for the site step
        # which is sum(r_i^2)
        ri_squared = rates**2
        ri_squared = ~ri_squared
        sum_ri_squared = self.csp.sum_array(ri_squared)

        dec_new_ages = self.csp.decrypt_arr(new_ages)

        return new_ages, sum_ri_squared


    def calc_model(self):
        """
        model calculation
        runs the site step and time step once
        @return: ages, rates and s0 values calculated by the 2 steps
        """

        iter = 1
        sum_ri_squared = 1
        for i in range(iter):
            rates, s0_vals, gamma_denom = self.adapted_site_step(self.m, self.n, self.ages, self.meth_val_list, sum_ri_squared)
            new_ages, sum_ri_squared = self.adapted_time_step(rates, s0_vals, self.meth_val_list, self.n, self.m, gamma_denom)
            self.ages = new_ages

        return self.ages, sum_ri_squared

