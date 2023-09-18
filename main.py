import numpy as np
from CSP.Csp import CSP
from MLE.Mle import MLE
import datetime
from EPM_no_sec.Epm import EPM
from DataHandler.DataSets import DataSets
from DataHandler.DataFormat import pearson_correlation
from Math_Utils.MathUtils import find_primes, read_primes_from_file
from DO.Do import DO
from Pyfhel import PyCtxt, Pyfhel, PyPtxt
import sys
from sympy.ntheory.modular import crt
import time
import argparse

def epm_orig():
    """
    EPM implementation without encryption. Used for benchmarking.
    @return: calculated model params
    """
    max_iterations = 3 #100
    rss = 0.0000001
    data_sets = DataSets()
    # read training data
    full_train_data = data_sets.get_example_train_data()
    train_samples, train_cpg_sites, train_ages, train_methylation_values = full_train_data
    # run pearson correlation in order to reduce the amount of processed data
    abs_pcc_coefficients = abs(pearson_correlation(train_methylation_values, train_ages))
    # correlation of .80 will return ~700 site indices
    # correlation of .91 will return ~24 site indices
    # these figures are useful for debug, our goal is to run the 700 sites
    correlated_meth_val_indices = np.where(abs_pcc_coefficients > .91)[0]
    correlated_meth_val = train_methylation_values[correlated_meth_val_indices, :]
    #correlated_meth_val = train_methylation_values
    # run the algorithm
    epm = EPM(correlated_meth_val, train_ages)
    model = epm.calc_model(max_iterations, rss)
    return model


def format_array_for_enc(arr: np.ndarray) -> np.ndarray:
    """
    prepare the data for usage with the BFV homomorphic encryption
    1. Round the numbers to the defined number of digits
    2. Format the numbers into integers
    @param arr: the array to format
    @return: the formatted array
    """

    rounded_arr = arr.round(decimals=2)
    rounded_arr = rounded_arr * (10 ** 2)
    int_arr = rounded_arr.astype(int)
    return int_arr

def epm_orig_new_method():
    max_iterations = 100
    rss = 0.0000001
    data_sets = DataSets()
    # read training data
    full_train_data = data_sets.get_example_train_data()
    #full_train_data = data_sets.reduce_data_size(full_train_data, 7503, 25)
    train_samples, train_cpg_sites, train_ages, train_methylation_values = full_train_data
    # run pearson correlation in order to reduce the amount of processed data
    abs_pcc_coefficients = abs(pearson_correlation(train_methylation_values, train_ages))
    # correlation of .80 will return ~700 site indices
    # correlation of .91 will return ~24 site indices
    # these figures are useful for debug, our goal is to run the 700 sites
    correlated_meth_val_indices = np.where(abs_pcc_coefficients > .91)[0]
    correlated_meth_val = train_methylation_values[correlated_meth_val_indices, :]
    #correlated_meth_val = train_methylation_values

    # uncommet these lines to use the rounded integer values for this algorithm
    # unfortunately, the numbers are so large that we receive an overflow
    # but this is a good method to print out the expected number sizes we may reach

    formatted_ages = format_array_for_enc(train_ages)
    formatted_correlated_meth_val = format_array_for_enc(correlated_meth_val)

    #formatted_ages = train_ages
    #formatted_correlated_meth_val = correlated_meth_val
    # run the algorithm
    epm = EPM(formatted_correlated_meth_val, formatted_ages)
    ages = epm.calc_model_new_method()
    return ages

def test_do():
    do = DO()
    ages = do.calc_model()
    return ages

def test_do_multi_process(n, p, c):
    do = DO()
    #ages = do.calc_model_multi_process(num_of_primes=30, enc_n=2**13, correlation=0.80)
    ages = do.calc_model_multi_process(num_of_primes=p, enc_n=2**n, correlation=c)
    return ages


def test_arr_sum():
    plaintext_prime = 1462436364289
    csp = CSP(plaintext_prime, 2**13)
    mle = MLE(csp)

    arr = np.arange(1, 20)
    enc_arr = csp.encrypt_array(arr)

    tic = time.perf_counter()
    new_sum = mle.calc_encrypted_array_sum(enc_arr, 3)


    dec_arr = csp.decrypt_arr(new_sum)
    print(dec_arr[0])

    tic = time.perf_counter()
    mle.calc_encrypted_array_sum(enc_arr, len(arr))
    print("new sum method took: ", time.perf_counter()-tic, "seconds")

def test_max_float():
    max_float = sys.float_info.max
    new_float = 2**3000
    try:
        result = new_float/4
    except OverflowError:
        print("could not divide")
        result = new_float//4
    print(result)

def main(n, p, c):
    #test_max_float()
    #exit()
    #test_read_primes()
    #test_primes()
    #bgv_test()
    #bgv_mult()
    #bfv_test()
    #test_arr_sum()

    #bgv_test2()

    # this runs the encrypted version
    #ages = test_do()
    ages = test_do_multi_process(n, p, c)
    # epm cleartext testing using the new algorithm without division
    #ages = epm_orig_new_method()
    # original algorithm
    #ages = epm_orig()
    #print(ages)
    #with open('ages_472.txt', 'w') as f:
    #    for age in ages:
    #        f.write(f"{age}\n")
    #print(len(ages))
    #mult_primes()
    # epm cleartext testing using the original cleartext algorithm with division


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--polynomial", help="Polynomial modulus")
    parser.add_argument("-p", "--primes", help="Number of primes")
    parser.add_argument("-c", "--correlation", help="Correlation percentage")

    args = parser.parse_args()
    n = 13
    p = 10
    c = 0.91
    if args.polynomial:
        n = int(args.polynomial)
    if args.primes:
        p = int(args.primes)
    if args.correlation:
        c = float(args.correlation)

    main(n, p, c)

