import numpy as np
import time
from Pyfhel import Pyfhel

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
    rank_A = None

    FILE_COUNTER = 0

    def __init__(self, csp):
        self.csp = csp

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

    def gen_site_step_masks(self):
        """
        Generate the random data for the masking trick
        @return: random matrix R and random vector r
        """
        random_matrix_gen_threshold = RANDOM_MATRIX_GEN_THRESHOLD

        while True:
            rand_R = np.random.randint(1, high=RANDOM_MATRIX_MAX_VAL, size=(self.rank_A, self.rank_A))
            # the random matrix needs to be invertible. This is checked using the determinant.
            # assert if a random matrix cannot be generated after RANDOM_MATRIX_GEN_THRESHOLD tries
            if np.linalg.det(rand_R) > 0:
                break
            random_matrix_gen_threshold -= 1
            assert random_matrix_gen_threshold != 0, \
                "Failed to generate random matrix R after {} attempts".format(RANDOM_MATRIX_GEN_THRESHOLD)

        rand_r = np.random.randint(1, high=RANDOM_MATRIX_MAX_VAL, size=self.A.shape[1])

        return rand_R, rand_r

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
            sum_arr = sum_arr * encoded_mask_arr
            ~sum_arr
            dec_sum = self.csp.decrypt_arr(sum_arr)
            dec_temp_arr = self.csp.decrypt_arr(temp_arr)
            sum_arr += temp_arr
            dec_sum = self.csp.decrypt_arr(sum_arr)

        # as it seems that the encryption "remembers" the shifted numbers
        # ie. if we shift back later, we will get them instead of zeros,
        # multiply by a new array which contains only a single 1 in the first cell
        mask_arr = np.array([1])
        encrypted_mask_arr = self.csp.encrypt_array(mask_arr)
        new_sum = encrypted_mask_arr * sum_arr
        ~new_sum
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

    def site_step(self, m: int, n: int, ages: np.ndarray, meth_vals: np.ndarray):
        """
        The EPM site step algorithm. This step calculates beta = (XtX)^-1 XtY using the conclustions from
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


        print("calc sigma_t starting at: ", time.perf_counter())
        sigma_t = self.calc_encrypted_array_sum(ages, m)
        print("calc sigma_t ended at: ", time.perf_counter())
        square_ages = ages ** 2
        ~square_ages # re-liniarize after power as it seems this does not happen automatically
        sigma_t_square = self.calc_encrypted_array_sum(square_ages, m)
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
        print("rates_assist_arr: ", self.csp.decrypt_arr(rates_assist_arr))

        s0_assist_arr *= all_sigma_t_arr
        s0_assist_arr -= all_sigma_t_square_arr
        ~s0_assist_arr  # re-liniarize after mult as it seems this does not happen automatically

        print("calc rates and s0 values starting at: ", time.perf_counter())
        # calculate the rate and s0 values
        for i in range(0, n):
            shifted_vals = meth_vals << (i*m)
            print("shifted: ", self.csp.decrypt_arr(shifted_vals))
            mult_assist = rates_assist_arr * shifted_vals
            ~mult_assist # re-linearize
            print("mult_assist: ", self.csp.decrypt_arr(mult_assist))
            r_val = self.calc_encrypted_array_sum(mult_assist, m)
            print("r_val: ", self.csp.decrypt_arr(r_val))
            mult_assist = s0_assist_arr * shifted_vals
            ~mult_assist  # re-linearize
            s0_val = self.calc_encrypted_array_sum(mult_assist, m)
            #now need to add each value to the rates
            rates = rates + (r_val >> i)
            s0_vals += (s0_val >> i)
        print("calc rates and s0 values ending at: ", time.perf_counter())

        dec_rates = self.csp.decrypt_arr(rates)
        dec_s0 = self.csp.decrypt_arr(s0_vals)
        return dec_rates, dec_s0
        #return rates, s0_vals

    def time_step(self, rates: np.ndarray, s0_vals: np.ndarray):
        """
        The time step receives the rate and s0 values and needs to perform the following calculation:
        t_j = (sum[r_i(s_ij-s0_i)])/sum(r_i^2)
        As the s_ij values are encrypted and need to offload part of the task to CSP
        @param rates: the rates calculated by the site step
        @param s0_vals: the s0 values calculated by the site step
        @return: the ages calculated by the time step
        """
        # np.subtract only works on rows. Here we need it to work on the columns
        # hence the double transpose
        S = np.transpose(np.subtract(np.transpose(self.meth_vals), s0_vals)*rates)

        r_squared_sum = np.sum(rates ** 2)
        ages = np.sum(S, axis=0) / r_squared_sum
        dec_ages = self.csp.calc_masked_time_step(ages)

        mle_to_csp_time_step_file = open("network/mle_to_csp_time_step_{}.bin".format(self.FILE_COUNTER), "wb")
        ages.tofile(mle_to_csp_time_step_file, sep=',')
        mle_to_csp_time_step_file.close()

        csp_to_mle_time_step_file = open("network/csp_to_mle_time_step_{}.bin".format(self.FILE_COUNTER), "wb")
        dec_ages.tofile(csp_to_mle_time_step_file, sep=',')
        csp_to_mle_time_step_file.close()
        self.FILE_COUNTER += 1

        return dec_ages

    def calc_model(self):
        """
        model calculation
        runs the site step and time step once
        @return: ages, rates and s0 values calculated by the 2 steps
        """
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
