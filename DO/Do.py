import numpy as np
from DataHandler.DataSets import DataSets
from DataHandler.DataFormat import format_array_for_enc, pearson_correlation
from MLE.Mle import MLE
import time
import logging, sys
from Math_Utils.MathUtils import read_primes_from_file
from CSP.Csp import CSP
from sympy.ntheory.modular import crt

# debug print level. Mainly used for function time measurement printing
logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)


'''
Data owner (DO) implementation
The DO provides the inputs to the MLE for the model calculation
'''


class DO:

    def __init__(self):
        self.train_data = None
        self.test_data = None

    def read_train_data(self):
        """
        read the sample training data
        @return: training data
        """
        data_sets = DataSets()
        full_train_data = data_sets.get_example_train_data()
        return full_train_data

    def encrypt_train_data(self, meth_vals, ages, csp):
        """
        Prepare and encrypt the training data for sending to MLE

        @param meth_vals: methylation values read from the training data
        @param ages: ages read from the training data
        @return: Encrypted methylation values and ages
        """

        encrypted_meth_vals = {}
        enc_array_size = csp.get_enc_n() // 2
        m = meth_vals.shape[1]

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
        elements_in_vector = enc_array_size // m

        # now reshape the meth_vals array to match the new format
        meth_vals_total_size = meth_vals.shape[0] * meth_vals.shape[1]
        meth_vals_new_cols = elements_in_vector*m
        meth_vals_new_rows = meth_vals_total_size // meth_vals_new_cols
        meth_vals_new_rows += 1 if (meth_vals_total_size % meth_vals_new_cols) > 0 else 0
        meth_vals_new_shape = np.copy(meth_vals)
        meth_vals_new_shape.resize((meth_vals_new_rows, meth_vals_new_cols), refcheck=False)

        # create the new encrypted vector dictionary
        print("Encrypting methylation values. shape is: ", meth_vals_new_shape.shape)
        encrypted_vector_list = np.apply_along_axis(csp.encrypt_array, axis=1, arr=meth_vals_new_shape)

        # encrypt the ages
        print("Encrypting ages")
        encrypted_ages = csp.encrypt_array(ages)

        return encrypted_vector_list, encrypted_ages



    def run_pearson_correlation(self, meth_vals: np.ndarray, ages: np.ndarray):
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
        correlated_meth_val_indices = np.where(abs_pcc_coefficients > .91)[0]
        # correlated_meth_val_indices = np.where(abs_pcc_coefficients > .80)[0]
        correlated_meth_vals = meth_vals[correlated_meth_val_indices, :]
        return correlated_meth_vals

    def calc_rss(self, rates: np.ndarray, s0_vals: np.ndarray, ages: np.ndarray, meth_vals: np.ndarray):
        """
        RSS calculation as defined for the EPM algorithm
        @param rates: rate values from the model
        @param s0_vals: s0 values from the model
        @param ages: age values from the model
        @param meth_vals: methylation values from the model
        @return: RSS value
        """
        total_error = 0.0
        col_num = 0
        for meth_col in meth_vals.T:
            t_j = ages[col_num]
            r_t_j = rates * t_j
            s0_plus_r_t_j = s0_vals + r_t_j
            m_rss = ((meth_col - s0_plus_r_t_j) ** 2)
            total_error += np.sum(m_rss)
            col_num += 1

        return total_error

    def run_crt(self, primes, res_list_of_lists):
        values = []
        transposed = np.array(res_list_of_lists).T.tolist()

        for res_list in transposed:
            values.append(crt(primes, res_list)[0])

        return values

    def calc_final_ages_crt(self, primes, numerator_list, denom_list):
        ages = self.run_crt(primes, numerator_list)
        sum_ri_squared = self.run_crt(primes, denom_list)
        final_ages = np.array(ages)/sum_ri_squared[0]
        print("age numerator: ",np.array(ages))
        print("sum_ri_squared: ", sum_ri_squared[0])
        return final_ages


    def calc_model(self):
        train_data = self.read_train_data()
        # comment out for no reduction
        data_sets = DataSets()
        #train_data = data_sets.reduce_data_size(train_data, 7503, 25)
        individuals, train_cpg_sites, train_ages, train_methylation_values = train_data
        correlated_meth_vals = self.run_pearson_correlation(train_methylation_values, train_ages)
        #correlated_meth_vals = train_methylation_values
        # format for encryption ie. round to 2 floating digits and convert to integer
        # as required by the homomorphic encryption
        logging.debug('Formatting methylation values and ages')
        formatted_meth_values = format_array_for_enc(correlated_meth_vals)
        formatted_ages = format_array_for_enc(train_ages)
        m = formatted_meth_values.shape[1]
        n = formatted_meth_values.shape[0]

        #primes = read_primes_from_file("/home/meirgold/git/EPM-code/primes.txt")
        primes = read_primes_from_file("/home/meirgold/git/EPM-code/very_large_primes.txt")
        moduli = []
        final_ages_list = []
        final_r_square_list = []
        primes_mul = 1
        #for i in range(7):
        for i in range(3):
            plaintext_prime = primes[i]
            print("Running secure algorithm for prime: ", plaintext_prime)
            primes_mul *= plaintext_prime
            csp = CSP(plaintext_prime)
            moduli.append(plaintext_prime)

            #encoded_n = csp.encode_array(np.array([n]))
            #encoded_m = csp.encode_array(np.array([m]))

            logging.debug('Encrypting ages and methylation  values')
            tic = time.perf_counter()

            enc_meth_vals, enc_ages = self.encrypt_train_data(formatted_meth_values, formatted_ages, csp)
            toc = time.perf_counter()
            logging.debug('This operation took: {:0.4f} seconds'.format(toc - tic))

            mle = MLE(csp)
            mle.get_data_from_DO(enc_meth_vals, enc_ages, m, n)
            new_ages, sum_ri_squared = mle.calc_model()
            decrypt_ages = csp.decrypt_arr(new_ages)[0:m]
            decrypt_sum_ri_squared = csp.decrypt_arr(sum_ri_squared)

            final_ages_list.append(decrypt_ages)
            final_r_square_list.append([decrypt_sum_ri_squared[0]])

        print("primes mul: ", primes_mul, " no. of digits: ", len(str(primes_mul)))

        final_ages = self.calc_final_ages_crt(moduli, final_ages_list, final_r_square_list)
        return final_ages


