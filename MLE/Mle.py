import numpy as np
import time
from Pyfhel import Pyfhel, PyCtxt

'''
Machine Learning Engine (MLE) implementation
The MLE receives the encrypted input data from the DO and calculates the model
with assistance from the CSP 
'''

# Global Definitions
# number of times to try and generate the random matrix R
RANDOM_MATRIX_GEN_THRESHOLD = 100
# maximum value to use in random generated matrix
# not too high in order to meet the requirements of the homomorphic encryption/decryption
RANDOM_MATRIX_MAX_VAL = 100



class MLE:

    csp = None
    A = None
    B = None
    meth_vals = None
    rand_mask = None
    recrypt_count = 0

    FILE_COUNTER = 0

    def __init__(self, csp):
        self.csp = csp
        rand_mask = np.random.randint(1, high=RANDOM_MATRIX_MAX_VAL, size=(csp.get_enc_n()//2))
        self.rand_mask = csp.encrypt_array(rand_mask)

    def get_data_from_DO(self, A: np.ndarray, B: np.ndarray, meth_vals: np.ndarray, rank_A: int):
        """
        Called by the DO to pass the input data to the MLE
        @param A:
        @param B:
        @param meth_vals:
        @param rank_A:
        @return:
        """
        self.A = A
        self.B = B
        self.meth_vals = meth_vals
        self.rank_A = rank_A

        DO_to_MLE_file = open("network/do_to_mle_{}.bin".format(self.FILE_COUNTER), "wb")
        A.tofile(DO_to_MLE_file, sep=',')
        B.tofile(DO_to_MLE_file, sep=',')
        meth_vals.tofile(DO_to_MLE_file, sep=',')
        DO_to_MLE_file.write(bytearray(rank_A))
        DO_to_MLE_file.close()
        self.FILE_COUNTER += 1

    def check_noise_lvl(self, ctxt):
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
        digits = arr_len
        shift = arr_len
        while digits > 0:
            shift = (shift // 2) if ((shift // 2) > 0) else 1
            digits -= shift
            mask_digits = shift
            dec_sum = self.csp.decrypt_arr(sum_arr)
            mask_arr = np.ones(mask_digits, dtype=np.int64)
            encoded_mask_arr = self.csp.encode_array(mask_arr)
            # shift to the next set of numbers to be added
            temp_arr = sum_arr << shift
            # leave only the elements relevant to the addition in the sum array
            #sum_arr = sum_arr * encoded_mask_arr
            sum_arr = self.safe_mul(sum_arr, encoded_mask_arr)
            dec_sum = self.csp.decrypt_arr(sum_arr)
            dec_temp_arr = self.csp.decrypt_arr(temp_arr)
            sum_arr += temp_arr
            dec_sum = self.csp.decrypt_arr(sum_arr)

        # as it seems that the encryption "remembers" the shifted numbers
        # ie. if we shift back later, we will get them instead of zeros,
        # multiply by a new array which contains only a single 1 in the first cell
        mask_arr = np.array([1])
        encrypted_mask_arr = self.csp.encrypt_array(mask_arr)
        new_sum = self.safe_mul(sum_arr, encrypted_mask_arr)
        return new_sum

    def calc_beta_corollary1(self, m: int, n: int, age_vals: np.ndarray, Y: np.ndarray):
        """
        calc beta using corollary1
        from the article https://bmcgenomics.biomedcentral.com/articles/10.1186/s12864-020-6606-0
        @return: rates and s0 values
        """

        mult_val_list = []
        result = np.zeros(2*n) # here will be the results of the final calculation
        sigma_t = np.sum(age_vals)
        sigma_t_square = np.sum(age_vals ** 2)
        #gamma = 1 / (sigma_t ** 2 - m * sigma_t_square)
        gamma = 1

        # prepare the multiply values
        for i in range(m):
            mult_val_list.append((-1*m*age_vals[i] + sigma_t) * gamma)

        for i in range(m):
            mult_val_list.append((age_vals[i] * sigma_t - sigma_t_square) * gamma)

        list_len_div2 = int(len(mult_val_list)/2)

        #Y = self.meth_vals.flatten().transpose().astype(float)

        # as described in the corollary, we should be able to calculate (XtX)^-1XtY
        # without using heavy linear algebra calculations
        # due to the structure of the matrix (XtX)^-1Xt which is made up of 2 expanded-diagonal matrices
        # we can calculate the multiplication values for each of these matrices
        # and then multiply the respective values in Y.
        # for the upper part of the result: each i*m value in Y will be multiplied by the i'th value
        # for the lower part of the result: each i*m value in Y will be multiplied by the i+list_len_div2 value

        i = 0
        temp_result_upper = Y.copy()
        temp_result_lower = Y.copy()
        while(i<list_len_div2):
            temp_result_upper[i::list_len_div2] *= mult_val_list[i]
            temp_result_lower[i::list_len_div2] *= mult_val_list[i + list_len_div2]
            i += 1

        # for the final step, each m values from the upper and lower matrices will be summed
        # (in order to "emulate" matrix multiplication) and put into its respective location in the result
        for j in range(n):
            result[j] = np.sum(temp_result_upper[j * m:((j+1)*m)])
            result[n + j] = np.sum(temp_result_lower[j * m:((j + 1) * m)])

        rates = result[0:n]
        s0 = result[n:]
        return rates, s0

    def enc_array_same_num(self, enc_num, size):
        """
        create an encrypted array where all cells contain the same number
        @param enc_num: the number to duplicate
        @param size: array size
        @return: encrypted array
        """
        i = 1
        num_array = enc_num + 0
        while i < size:
            num_array += num_array >> i
            i *= 2
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
        print("calc sigma_t ended at: ", time.perf_counter())
        square_ages = ages ** 2
        ~square_ages # re-liniarize after power as it seems this does not happen automatically
        # sigma_t_square = self.calc_encrypted_array_sum(square_ages, m)
        sigma_t_square = self.csp.sum_array(square_ages)
        print("calc sigma_t_square ended at: ", time.perf_counter())
        gamma_denom = (sigma_t ** 2 - m * sigma_t_square)
        ~gamma_denom # re-linearize
        rates_assist_arr = -1 * m * ages
        s0_assist_arr = ages

        print("calc all_sigma arrays starting at: ", time.perf_counter())
        for i in range(m):
            all_sigma_t_arr += (sigma_t >> i)
            all_sigma_t_square_arr += (sigma_t_square >> i)
        print("calc all_sigma arrays ended at: ", time.perf_counter())

        # in order to avoid the need to build the expanded diagonal matrices
        # we create a vector with the x_0....x_m values and multiply the Y vector by n copies of this vector
        rates_assist_arr += all_sigma_t_arr
        rates_assist_arr = self.safe_mul(rates_assist_arr, sum_ri_square_arr)
        print("rates_assist_arr: ", self.csp.decrypt_arr(rates_assist_arr))

        s0_assist_arr *= all_sigma_t_arr
        s0_assist_arr = ~s0_assist_arr
        s0_assist_arr -= all_sigma_t_square_arr

        print("calc rates and s0 values starting at: ", time.perf_counter())
        # calculate the rate and s0 values
        for i in range(0, n):
            for meth_vals in meth_vals_list:
                shifted_vals = meth_vals << (i*m)
            print("shifted: ", self.csp.decrypt_arr(shifted_vals))
            mult_assist = self.safe_mul(rates_assist_arr, shifted_vals)
            #~mult_assist # re-linearize
            print("mult_assist: ", self.csp.decrypt_arr(mult_assist))
            # r_val = self.calc_encrypted_array_sum(mult_assist, m)
            r_val = self.csp.sum_array(mult_assist)
            print("r_val: ", self.csp.decrypt_arr(r_val))
            mult_assist = self.safe_mul(s0_assist_arr, shifted_vals)
            #~mult_assist  # re-linearize
            # s0_val = self.calc_encrypted_array_sum(mult_assist, m)
            s0_val = self.csp.sum_array(mult_assist)
            #now need to add each value to the rates
            rates = rates + (r_val >> i)
            s0_vals += (s0_val >> i)
        print("calc rates and s0 values ending at: ", time.perf_counter())

        '''
        dec_rates = self.csp.decrypt_arr(rates)
        dec_s0 = self.csp.decrypt_arr(s0_vals)
        return dec_rates, dec_s0
        '''
        return rates, s0_vals, gamma_denom

    def adapted_time_step(self, rates, s0_vals, meth_vals, n, m, gamma):

        # calc the matrix  S = (S_ij - s0_i)*r_i
        # the main issue here is that S_ij is represented as long vectors which cannot be transposed as
        # the encrypted vectors do not support transposing
        # this means that each n elements in S_ij need to be multiplied by the same r_i

        enc_array_size = self.csp.get_enc_n() // 2 # the array is represented
        # the addition of 0 causes the creation of a new encrypted array.
        # without it, the variable on the left side of the = will just be a pointer to the one on the right
        calc_assist_s0 = s0_vals + 0
        calc_assist_rates = rates + 0
        iterations = min(m, (enc_array_size // n))-1

        for i in range(1, iterations):
            calc_assist_s0 += (s0_vals >> (i*m - 1))
            calc_assist_rates += (rates >> (i*m - 1))


        mask = np.zeros(enc_array_size, dtype=np.int64)
        for i in range(iterations):
            mask[i*m] = 1
        encoded_mask = self.csp.encode_array(mask)
        calc_assist_s0 *= encoded_mask
        calc_assist_rates *= encoded_mask

        separated_s0 = calc_assist_s0 + 0
        separated_rates = calc_assist_rates + 0

        for i in range(1, m):
            calc_assist_s0 += (separated_s0 >> i)
            calc_assist_rates += (separated_rates >> i)

        # create an array full of gamma values in order to easily multiply each Sij by gamma
        gamma_array = self.enc_array_same_num(gamma, enc_array_size)
        meth_vals_gamma = meth_vals * gamma_array

        # now we need to calc the nominator of the time step.
        # which is: sum(r_i(meth_vals_gamma - s^0_i))
        # first lets calculate (meth_vals_gamma - s^0_i)
        p = meth_vals_gamma - calc_assist_s0
        # now lets multiply by r_i
        r_p = p * calc_assist_rates
        # now the tricky part - calculate the sums for each tj
        # we will use the mask to sum each r_p_j*m
        # this should give us the required sum for each tj
        encrypted_mask = self.csp.encrypt_array(mask)
        new_ages = self.csp.sum_array(r_p * encrypted_mask)
        for i in range(1, m):
            new_ages += ((self.csp.sum_array(r_p * (encrypted_mask >> i))) >> i)

        # now we just need to calculate the denominator for the site step
        # which is sum(r_i^2)
        ri_squared = rates**2
        ri_squared = ~ri_squared
        sum_ri_squared = self.csp.sum_array(ri_squared)

        return new_ages, sum_ri_squared


    def calc_model(self):
        """
        model calculation
        runs the site step and time step once
        @return: ages, rates and s0 values calculated by the 2 steps
        """
        pass

        '''
        R, r = self.gen_site_step_masks()
        C, d = self.mask_matrices(R, r)
        rates, s0_vals = self.site_step(C, d, R, r)
        ages = self.time_step(rates, s0_vals)

        MLE_to_DO_file = open("network/mle_to_do_{}.bin".format(self.FILE_COUNTER), "wb")
        ages.tofile(MLE_to_DO_file, sep=',')
        rates.tofile(MLE_to_DO_file, sep=',')
        s0_vals.tofile(MLE_to_DO_file, sep=',')
        MLE_to_DO_file.close()
        self.FILE_COUNTER += 1

        return ages, rates, s0_vals
        '''
