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
    correlated_meth_val_indices = np.where(abs_pcc_coefficients > .80)[0]
    correlated_meth_val = train_methylation_values[correlated_meth_val_indices, :]

    full_test_data = data_sets.get_example_test_data()
    test_samples, test_cpg_sites, test_ages, test_methylation_values = full_test_data
    correlated_test_meth_val = test_methylation_values[correlated_meth_val_indices, :]

    # run the algorithm
    epm = EPM(correlated_meth_val, train_ages, correlated_test_meth_val)
    model_output = epm.calc_model(max_iterations, rss)

    file_timestamp = time.strftime("%Y%m%d-%H%M%S")
    with open('epm_orig_' + file_timestamp + '.log', 'w') as fp:
        fp.write("ages\n")
        fp.write(f"{model_output['ages']}\n")
        fp.write("predicted_ages\n")
        fp.write(f"{model_output['predicted_ages']}\n")
        fp.write("rate values:\n")
        fp.write(f"{model_output['rates']}\n")
        fp.write("s0 values:\n")
        fp.write(f"{model_output['s0']}\n")
        fp.write("num of iterations:\n")
        fp.write(f"{model_output['num_of_iterations']}\n")
        fp.write("rss error:\n")
        fp.write(f"{model_output['rss_err']}\n")


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
    #correlated_meth_val_indices = np.where(abs_pcc_coefficients > .80)[0]
    correlated_meth_val = train_methylation_values[correlated_meth_val_indices, :]
    #correlated_meth_val = train_methylation_values

    # uncomment these lines to use the rounded integer values for this algorithm
    # unfortunately, the numbers are so large that we receive an overflow
    # but this is a good method to print out the expected number sizes we may reach

    formatted_ages = format_array_for_enc(train_ages)
    formatted_correlated_meth_val = format_array_for_enc(correlated_meth_val)

    #formatted_ages = train_ages
    #formatted_correlated_meth_val = correlated_meth_val
    # run the algorithm
    epm = EPM(formatted_correlated_meth_val, formatted_ages, formatted_correlated_meth_val)
    ages = epm.calc_model_new_method()
    return ages


def test_do_multi_process(n, p, c, r, b):
    do = DO()
    ages = do.calc_model_multi_process(num_of_primes=p, enc_n=2**n, correlation=c, rounds=r, prime_bits=b)
    return ages


def main(n, p, c, r, b):
    # this runs the encrypted version
    ages = test_do_multi_process(n, p, c, r, b)
    # epm cleartext testing using the new algorithm without division
    #ages = epm_orig_new_method()
    # original algorithm
    # ages = epm_orig()
    #with open('epm_orig_results.txt', 'w') as fp:
    #    fp.write(f"{ages}\n")

    print(ages)


if __name__ == '__main__':
    #test_num_of_mult()
    #epm_orig_new_method()
    #exit()

    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--polynomial", help="Polynomial modulus")
    parser.add_argument("-p", "--primes", help="Number of primes")
    parser.add_argument("-c", "--correlation", help="Correlation percentage")
    parser.add_argument("-r", "--rounds", help="number of CEM rounds")
    parser.add_argument("-b", "--bits", help="number bits per prime")
    parser.add_argument("-o", "--orig", action='store_true', help="run the original cleartext algorithm")

    args = parser.parse_args()

    n = 13
    p = 10
    c = 0.91
    r = 2
    b = 30

    # for large dataset run:
    # n=14 p=52, c=0.8, r=3, b=30

    if args.polynomial:
        n = int(args.polynomial)
    if args.primes:
        p = int(args.primes)
    if args.correlation:
        c = float(args.correlation)
    if args.rounds:
        r = int(args.rounds)
    if args.bits:
        b = int(args.bits)

    if args.orig:
        model = epm_orig()
    else:
        main(n, p, c, r, b)

