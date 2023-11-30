import numpy as np
from DataHandler.DataSets import DataSets
from DataHandler.DataFormat import format_array_for_enc, format_array_for_dec, pearson_correlation
from MLE.Mle import MLE
from CSP.Csp import CSP
from CRT.CrtVector import CrtVector
from CRT.CrtSet import CrtSet
from sympy.ntheory.modular import crt
import time
import logging, sys
import random
import sympy
import math

# debug print level. Mainly used for function time measurement printing
logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)


'''
Data owner (DO) implementation
The DO provides the inputs to the MLE for the model calculation
'''
class DO:

    def __init__(self, num_of_primes=10, correlation=0.91, prime_bits=30):
        self.num_of_primes = num_of_primes
        self.correlation = correlation
        self.prime_bits = prime_bits
        self.train_data = None
        self.formatted_ages = None
        self.formatted_meth_values = None
        self.m = None
        self.n = None

    def read_example_train_data(self):
        """
        read the sample training data
        @return: training data
        """
        data_sets = DataSets()
        full_train_data = data_sets.get_example_train_data()
        return full_train_data

    def read_example_test_data(self):
        data_sets = DataSets()
        full_test_data = data_sets.get_example_test_data()
        return full_test_data

    def encrypt_meth_vals(self, meth_vals, prime_index, csp):

        enc_array_size = csp.get_enc_n() // 2
        cols = meth_vals.shape[1]

        # as the encrypted data is stored in a 1X1 vector format, need to convert the large methylation value
        # array to one or more vectors of this type
        # each vector will contain the methylation values for several individuals
        # if the original methylation array looked as such for 3 individuals and 2 sites:
        #        ind_0 ind_1 ind_2
        # site_1  S_00 S_01  S_02
        # site_2  S_10 S_11  S_12
        # the new vector will look like this:
        # S_00 S_01 S_02 S_10 S_11 S_12 ...

        # calculate the amount of individual sites that can fit into a single vector
        elements_in_vector = enc_array_size // cols

        # now reshape the meth_vals array to match the new format
        meth_vals_total_size = meth_vals.shape[0] * meth_vals.shape[1]
        meth_vals_new_cols = elements_in_vector * cols
        meth_vals_new_rows = meth_vals_total_size // meth_vals_new_cols
        meth_vals_new_rows += 1 if (meth_vals_total_size % meth_vals_new_cols) > 0 else 0
        meth_vals_new_shape = np.copy(meth_vals).flatten()
        meth_vals_new_shape.resize((meth_vals_new_rows, meth_vals_new_cols), refcheck=False)
        # this is a workaround as for some shapes Pyfhel throws an exception on array sizes which are lower than maximum
        zero_cols_to_add = enc_array_size - meth_vals_new_cols
        if zero_cols_to_add > 0:
            temp_arr = np.zeros([meth_vals_new_rows, zero_cols_to_add], dtype=np.int64)
            meth_vals_new_shape = np.append(meth_vals_new_shape, temp_arr, axis=1)

        # create the new encrypted vector dictionary
        print("DO: Encrypting methylation values. shape is: ", meth_vals_new_shape.shape)
        encrypted_vector_list = np.apply_along_axis(csp.encrypt_array, axis=1,
                                                    arr=meth_vals_new_shape, prime_index=prime_index)

        return encrypted_vector_list

    def encrypt_train_data(self, meth_vals, ages, prime_index, csp):
        """
        Prepare and encrypt the training data for sending to MLE

        @param meth_vals: methylation values read from the training data
        @param ages: ages read from the training data
        @param prime_index: index of the prime in the crt_vector
        @param csp: the crypto service provider
        @return: Encrypted methylation values and ages
        """

        encrypted_meth_vals = self.encrypt_meth_vals(meth_vals, prime_index, csp)
        encrypted_transposed_meth_vals = self.encrypt_meth_vals(meth_vals.T, prime_index, csp)
        # encrypt the ages
        encrypted_ages = csp.encrypt_array(ages, prime_index)
        return encrypted_meth_vals, encrypted_transposed_meth_vals, encrypted_ages

    def pearson_correlation_indices(self, meth_vals: np.ndarray, ages: np.ndarray, percentage):
        """
        The pearson correlation algorithm for reducing the amount of processed data
        @param meth_vals: methylation values from input data
        @param ages: ages from input data
        @return: correlated methylation values
        """
        abs_pcc_coefficients = abs(pearson_correlation(meth_vals, ages))
        # correlation of .80 will return ~700 site indices
        # correlation of .91 will return ~24 site indices
        # these figures are useful for debug, our goal is to run the 700 sites
        correlated_meth_val_indices = np.where(abs_pcc_coefficients > percentage)[0]
        # correlated_meth_val_indices = np.where(abs_pcc_coefficients > .80)[0]
        #correlated_meth_vals = meth_vals[correlated_meth_val_indices, :]
        #return correlated_meth_vals
        return correlated_meth_val_indices


    def generate_primes(self, total_primes, enc_n, num_of_bits=30):
        primes = []
        # prime upper bound and lower bound
        prime_lb = 2 ** num_of_bits
        prime_ub = (2 ** (num_of_bits + 1)) - 1
        num_of_primes = 0
        while num_of_primes < total_primes:
            p = random.randint(prime_lb, prime_ub)
            if ((p - 1) % (2*enc_n) == 0) and sympy.isprime(p) and p not in primes:
                primes.append(p)
                num_of_primes += 1

        return primes

    def encrypt_and_pass_data_to_mle(self, csp, mle):
        tic = time.perf_counter()
        crt_vector = CrtVector()
        num_of_slots = csp.get_num_of_slots()
        poly_modulus_degree = csp.get_poly_modulus_degree()
        print("DO: Generating primes")
        primes = self.generate_primes(self.num_of_primes, poly_modulus_degree, self.prime_bits)

        print("DO: requesting keys from CSP")
        prime_indices = csp.gen_keys(primes)
        print("DO: Reading input data")
        train_data = self.read_example_train_data()
        train_individuals, train_cpg_sites, train_ages, train_methylation_values = train_data
        correlated_meth_val_indices = self.pearson_correlation_indices(train_methylation_values,
                                                                       train_ages, self.correlation)
        correlated_meth_vals = train_methylation_values[correlated_meth_val_indices, :]

        # format for encryption ie. round to l floating digits and convert to integer
        # as required by the homomorphic encryption
        self.formatted_meth_values = format_array_for_enc(correlated_meth_vals)
        self.formatted_ages = format_array_for_enc(train_ages)
        self.m = self.formatted_meth_values.shape[1]
        self.n = self.formatted_meth_values.shape[0]

        # the code doesn't support numbers of individuals and/or sites that are
        # larger than the polynomial modulus degree
        assert (self.m < num_of_slots) and (self.n < num_of_slots), (
            "number of individuals {} and number of sites {} must be lower than polynomial modulus degree: {}".format(self.m, self.n, ENC_N))

        # encrypt and create the CRT vector
        # as we plan to support multiple DOs, the prime index may not start from zero
        # but from where the last prime was added by the previous DO
        for prime_index in prime_indices:
            enc_meth_vals, enc_transposed_meth_vals, enc_ages = self.encrypt_train_data(self.formatted_meth_values,
                                                                                        self.formatted_ages,
                                                                                        prime_index, csp)
            crt_vector.add(CrtSet(prime_index, enc_ages, enc_meth_vals, enc_transposed_meth_vals))

        print("DO: passing encrypted data to MLE")
        mle.get_data_from_DO(crt_vector, self.m, self.n)
        return 0


        '''
        file_timestamp = time.strftime("%Y%m%d-%H%M%S")
        log_fp = open('ages_log_'+file_timestamp+'.log', 'w')

        mle.get_data_from_DO(self.m, self.n)


        toc = time.perf_counter()
        log_fp.write("Num of processes: {}\n".format(num_of_processes))
        log_fp.write("Num of primes: {}\n".format(NUM_OF_PRIMES))
        log_fp.write("poly modulus for encryption: {}\n".format(ENC_N))
        log_fp.write("prime mult length is: {}\n".format(len(str(math.prod(primes)))))
        log_fp.write("calculation took: {} minutes\n".format((toc - tic) / 60))
        log_fp.close()
        print("calculation took: ", toc - tic, " seconds")

        return 0
        '''



