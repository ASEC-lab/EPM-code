import numpy as np
from DataHandler.DataSets import DataSets
from DataHandler.DataFormat import format_array_for_enc, format_array_for_dec, pearson_correlation
from MLE.Mle import MLE
import time
import logging, sys
from Math_Utils.MathUtils import read_primes_from_file
from CSP.Csp import CSP
from sympy.ntheory.modular import crt
from multiprocessing import cpu_count, Queue, Process, Lock, current_process
import queue
import random
import sympy
from Pyfhel import Pyfhel
import math

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
        self.formatted_ages = None
        self.formatted_meth_values = None
        self.m = None
        self.n = None

    def read_train_data(self):
        """
        read the sample training data
        @return: training data
        """
        data_sets = DataSets()
        full_train_data = data_sets.get_example_train_data()
        return full_train_data

    def encrypt_meth_vals(self, meth_vals, csp):

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
        print("Encrypting methylation values. shape is: ", meth_vals_new_shape.shape)
        encrypted_vector_list = np.apply_along_axis(csp.encrypt_array, axis=1, arr=meth_vals_new_shape)
        return encrypted_vector_list

    def encrypt_train_data(self, meth_vals, ages, csp):
        """
        Prepare and encrypt the training data for sending to MLE

        @param meth_vals: methylation values read from the training data
        @param ages: ages read from the training data
        @return: Encrypted methylation values and ages
        """

        encrypted_meth_vals = self.encrypt_meth_vals(meth_vals, csp)
        encrypted_transposed_meth_vals = self.encrypt_meth_vals(meth_vals.T, csp)
        # encrypt the ages
        print("Encrypting ages")
        encrypted_ages = csp.encrypt_array(ages)

        return encrypted_meth_vals, encrypted_transposed_meth_vals, encrypted_ages



    def run_pearson_correlation(self, meth_vals: np.ndarray, ages: np.ndarray, percentage):
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
        final_ages = []
        prime_mult = math.prod(primes)
        half_prime_mult = prime_mult//2
        ages = self.run_crt(primes, numerator_list)
        sum_ri_squared = self.run_crt(primes, denom_list)

        print("age numerator:", ages)
        print("sum_ri_squared: ", sum_ri_squared[0])
        # avoid using numpy as it may have limitation on the object size
        #final_ages = np.array(ages)/sum_ri_squared[0]
        for age_num in ages:
            if age_num > half_prime_mult:
                age_num = -(prime_mult - age_num)
            age = age_num/sum_ri_squared[0]
            final_ages.append(age)

        return final_ages

    def generate_primes(self, total_primes, primes, enc_n):
        # prime upper bound and lower bound
        prime_lb = 2 ** 58
        prime_ub = (2 ** 59) - 1
        num_of_primes = 0
        while num_of_primes < total_primes:
            p = random.randint(prime_lb, prime_ub)
            if (p - 1) % enc_n == 0:
                if sympy.isprime(p) and p not in primes:
                    # test encryption as for some reason it does not work for every prime
                    ctxt = Pyfhel()
                    try:
                        print("trying: ",p)
                        result = ctxt.contextGen("bfv", n=enc_n, t=p, sec=128)
                    except ValueError:
                        print("Value Error")
                        result = False
                    else:
                        if result:
                            primes.append(p)
                            num_of_primes += 1

    def result_proc(self, num_of_primes, results_queue, file_timestamp):

        i = 0
        moduli = []
        final_ages_list = []
        final_r_square_list = []

        while i < num_of_primes:
            if not results_queue.empty():
                res = results_queue.get()
                moduli.append(res['moduli'])
                final_ages_list.append(res['ages'])
                final_r_square_list.append(res['sum_ri_squared'])
                i += 1

        final_ages = self.calc_final_ages_crt(moduli, final_ages_list, final_r_square_list)
        final_ages = format_array_for_dec(final_ages)

        with open('ages_'+file_timestamp+'.log', 'w') as fp:
            fp.write("ages:\n")
            for age in final_ages:
                fp.write(f"{age}\n")
        print(final_ages)

    def calc_process(self, calc_per_prime_queue, results_queue, enc_n):
        while True:
            try:
                result = {}
                prime = calc_per_prime_queue.get_nowait()
                print(current_process().name, " is calculating for prime: ", prime)
                csp = CSP(prime, enc_n)
                enc_meth_vals, enc_transposed_meth_vals, enc_ages = self.encrypt_train_data(self.formatted_meth_values,
                                                                                            self.formatted_ages, csp)
                mle = MLE(csp)
                mle.get_data_from_DO(enc_meth_vals, enc_transposed_meth_vals, enc_ages, self.m, self.n)

                new_ages, sum_ri_squared = mle.calc_model()
                decrypt_ages = csp.decrypt_arr(new_ages)[0:self.m]
                decrypt_sum_ri_squared = csp.decrypt_arr(sum_ri_squared)

                result['moduli'] = prime
                result['ages'] = decrypt_ages
                result['sum_ri_squared'] = decrypt_sum_ri_squared
                results_queue.put(result)
            except queue.Empty:
                break

        return True

    def calc_model_multi_process(self, num_of_primes=10, enc_n=2**13, correlation=0.91):
        tic = time.perf_counter()
        NUM_OF_PRIMES = num_of_primes
        ENC_N = enc_n
        num_of_cores = cpu_count()
        calc_per_prime_queue = Queue()
        results_queue = Queue()
        processes = []
        primes = []
        #self.generate_primes(NUM_OF_PRIMES, primes, ENC_N)
        primes = read_primes_from_file("very_large_primes.txt")
        primes = primes[0:NUM_OF_PRIMES]
        train_data = self.read_train_data()
        individuals, train_cpg_sites, train_ages, train_methylation_values = train_data
        correlated_meth_vals = self.run_pearson_correlation(train_methylation_values, train_ages, correlation)
        #correlated_meth_vals = train_methylation_values
        # format for encryption ie. round to 2 floating digits and convert to integer
        # as required by the homomorphic encryption
        self.formatted_meth_values = format_array_for_enc(correlated_meth_vals)
        self.formatted_ages = format_array_for_enc(train_ages)
        self.m = self.formatted_meth_values.shape[1]
        self.n = self.formatted_meth_values.shape[0]

        # the code doesn't support numbers of individuals and/or sites that are
        # larger than the polynomial modulus degree
        assert (self.m < ENC_N) and (self.n < ENC_N), (
            "number of individuals {} and number of sites {} must be lower than polynomial modulus degree: {}".format(self.m, self.n, ENC_N))

        for prime in primes:
            calc_per_prime_queue.put(prime)

        file_timestamp = time.strftime("%Y%m%d-%H%M%S")
        log_fp = open('ages_log_'+file_timestamp+'.log', 'w')

        num_of_processes = min(num_of_cores*2, NUM_OF_PRIMES)

        for process in range(num_of_processes):
            p = Process(target=self.calc_process, args=[calc_per_prime_queue, results_queue, ENC_N])
            processes.append(p)
            p.start()

        p = Process(target=self.result_proc, args=[NUM_OF_PRIMES, results_queue, file_timestamp])
        processes.append(p)
        p.start()

        # now wait for all processes to complete
        for p in processes:
            p.join()

        toc = time.perf_counter()
        log_fp.write("Num of processes: {}\n".format(num_of_processes))
        log_fp.write("Num of primes: {}\n".format(NUM_OF_PRIMES))
        log_fp.write("poly modulus for encryption: {}\n".format(ENC_N))
        log_fp.write("prime mult length is: {}\n".format(len(str(math.prod(primes)))))
        log_fp.write("calculation took: {} minutes\n".format((toc - tic) / 60))
        log_fp.close()
        print("calculation took: ", toc - tic, " seconds")

        return 0

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
        for i in range(10):
            plaintext_prime = primes[i]
            print("Running secure algorithm for prime: ", plaintext_prime)
            primes_mul *= plaintext_prime
            csp = CSP(plaintext_prime, 2**13)
            moduli.append(plaintext_prime)

            logging.debug('Encrypting ages and methylation  values')
            tic = time.perf_counter()

            enc_meth_vals, enc_transposed_meth_vals, enc_ages = self.encrypt_train_data(formatted_meth_values,
                                                                                        formatted_ages, csp)
            toc = time.perf_counter()
            logging.debug('This operation took: {:0.4f} seconds'.format(toc - tic))

            mle = MLE(csp)
            mle.get_data_from_DO(enc_meth_vals, enc_transposed_meth_vals, enc_ages, m, n)
            new_ages, sum_ri_squared = mle.calc_model()
            decrypt_ages = csp.decrypt_arr(new_ages)[0:m]
            decrypt_sum_ri_squared = csp.decrypt_arr(sum_ri_squared)

            final_ages_list.append(decrypt_ages)
            final_r_square_list.append([decrypt_sum_ri_squared[0]])

        print("primes mul: ", primes_mul, " no. of digits: ", len(str(primes_mul)))

        final_ages = self.calc_final_ages_crt(moduli, final_ages_list, final_r_square_list)
        final_ages = format_array_for_dec(final_ages)
        return final_ages


