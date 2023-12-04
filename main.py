import numpy as np
from CSP.Csp import CSP
from MLE.Mle import MLE
from DataHandler.DataSets import DataSets
from EPM_no_sec.Epm import EPM
from DO.Do import DO
import time
import argparse

'''
main file with examples for running secure and cleartext implementations

Coded by Meir Goldenberg  
meirgold@hotmail.com
'''

def epm_cleartext(correlation_percentage=0.8, max_iterations=100):
    """
    EPM implementation without encryption. Used for benchmarking.
    @return: calculated model params
    """
    rss = 0.0000001
    data_sets = DataSets()
    ages, correlated_meth_val = data_sets.load_and_prepare_example_data(correlation_percentage)

    # run the algorithm
    epm = EPM(correlated_meth_val, ages, None)
    model_output = epm.calc_ages(max_iterations, rss)

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


def epm_cleartext_no_division(correlation_percentage=0.8, iterations=3):
    data_sets = DataSets()
    ages, correlated_meth_val = data_sets.load_and_prepare_example_data(correlation_percentage)
    # uncomment these lines to use the rounded integer values for this algorithm
    # unfortunately, the numbers are too large for numpy to handle that we receive an overflow
    # but this is a good method to print out the expected number sizes we may reach

    #formatted_ages = format_array_for_enc(train_ages)
    #formatted_correlated_meth_val = format_array_for_enc(correlated_meth_val)

    formatted_ages = ages
    formatted_correlated_meth_val = correlated_meth_val
    # run the algorithm
    epm = EPM(formatted_correlated_meth_val, formatted_ages, None)
    ages = epm.calc_ages_no_division(iterations)
    file_timestamp = time.strftime("%Y%m%d-%H%M%S")
    with open('epm_orig_no_division_' + file_timestamp + '.log', 'w') as fp:
        fp.write("ages\n")
        fp.write(f"{ages}\n")


def calc_ages_secure(polynomial_modulus_degree, num_of_primes, correlation, rounds, prime_bits, auto_recrypt):
    tic = time.perf_counter()
    csp = CSP(2**polynomial_modulus_degree)
    mle = MLE(csp, rounds, auto_recrypt)
    do = DO(num_of_primes=num_of_primes, correlation=correlation, prime_bits=prime_bits)
    do.encrypt_and_pass_data_to_mle(csp, mle)
    mle.calc_model_multi_process()
    csp.decrypt_and_publish_results(mle.crt_vector, mle.m, mle.n)
    toc = time.perf_counter()
    file_timestamp = time.strftime("%Y%m%d-%H%M%S")
    with open('epm_secure_' + file_timestamp + '.log', 'w') as fp:
        fp.write("Parameters:\n")
        fp.write("  polynomial modulus degree: {}\n".format(polynomial_modulus_degree))
        fp.write("  num of primes: {}\n".format(num_of_primes))
        fp.write("  bits per prime: {}\n".format(prime_bits))
        fp.write("  correlation: {}\n".format(correlation))
        fp.write("  CEM rounds: {}\n".format(rounds))
        fp.write("  allow auto recrypt: {}\n".format(auto_recrypt))
        fp.write("Number of processes: {}\n".format(mle.num_of_processes))
        fp.write("Total execution time: {} minutes\n".format((toc - tic)/60))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--polynomial", help="Polynomial modulus")
    parser.add_argument("-p", "--primes", help="Number of primes")
    parser.add_argument("-c", "--correlation", help="Correlation percentage")
    parser.add_argument("-r", "--rounds", help="number of CEM rounds")
    parser.add_argument("-b", "--bits", help="number bits per prime")
    parser.add_argument("-a", "--auto_recrypt", action='store_true',
                        help="allow auto recrypt upon low noise level")

    parser.add_argument("-o", "--orig_cleartext", action='store_true',
                        help="run the original cleartext algorithm")
    parser.add_argument("-d", "--cleartext_no_division", action='store_true',
                        help="run the original cleartext algorithm with no division")

    args = parser.parse_args()

    # default values
    n = 13
    p = 10
    c = 0.91
    r = 2
    b = 30
    a = False

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
    if args.auto_recrypt:
        a = True

    if args.orig_cleartext:
        epm_cleartext(c, r)
    elif args.cleartext_no_division:
        epm_cleartext_no_division(c, r)
    else:
        calc_ages_secure(n, p, c, r, b, a)


if __name__ == '__main__':
    main()


