import numpy as np
from CSP.Csp import CSP
import datetime
from EPM_no_sec.Epm import EPM
from DataHandler.DataSets import DataSets
from DataHandler.DataFormat import pearson_correlation
from DO.Do import DO

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
    train_samples, train_cpg_sites, train_ages, train_methylation_values = full_train_data
    # run pearson correlation in order to reduce the amount of processed data
    abs_pcc_coefficients = abs(pearson_correlation(train_methylation_values, train_ages))
    # correlation of .80 will return ~700 site indices
    # correlation of .91 will return ~24 site indices
    # these figures are useful for debug, our goal is to run the 700 sites
    correlated_meth_val_indices = np.where(abs_pcc_coefficients > .91)[0]
    correlated_meth_val = train_methylation_values[correlated_meth_val_indices, :]

    # uncommet these lines to use the rounded integer values for this algorithm
    # unfortunately, the numbers are so large that we receive an overflow
    # but this is a good method to print out the expected number sizes we may reach

    # formatted_ages = format_array_for_enc(train_ages)
    # formatter_correlated_meth_val = format_array_for_enc(correlated_meth_val)

    formatted_ages = train_ages
    formatter_correlated_meth_val = correlated_meth_val
    # run the algorithm
    epm = EPM(formatter_correlated_meth_val, formatted_ages)
    ages = epm.calc_model_new_method()
    return ages

def test_do():
    csp = CSP()
    do = DO(csp)
    ages = do.calc_model()
    return ages

def main():

    # this runs the encrypted version
    ages = test_do()

    # epm cleartext testing using the new algorithm without division
    # ages = epm_orig_new_method()

    # epm cleartext testing using the original cleartext algorithm with division
    # ages = epm_orig()


if __name__ == '__main__':
    main()

