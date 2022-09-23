import numpy as np
from Pyfhel import PyCtxt, Pyfhel, PyPtxt
import multiprocessing

# a set of functions for formatting the input data

# Global definitions
# the number of floating point digits to round to
FLOATING_DIGIT_ROUND = 2


def format_array_for_enc(arr: np.ndarray) -> np.ndarray:
    """
    prepare the data for usage with the homomorphic encryption
    1. Round the numbers to the defined number of digits
    2. Format the numbers into integers
    @param arr: the array to format
    @return: the formatted array
    """

    rounded_arr = arr.round(decimals=FLOATING_DIGIT_ROUND)
    rounded_arr = rounded_arr * (10 ** FLOATING_DIGIT_ROUND)
    int_arr = rounded_arr.astype(int)
    return int_arr


def restore_array(arr: np.ndarray) -> np.ndarray:
    """
    revert the formatting from the previous function
    @param arr: the array to restore
    @return: the restored array
    """
    return arr/(10 ** FLOATING_DIGIT_ROUND)


def enc_array(arr: np.ndarray):
    assert arr.ndim == 1, "Only 1 dimensional arrays are supported"
    pyfhel_server = Pyfhel()
    pyfhel_server.load_context("../context.con")
    pyfhel_server.load_public_key("../pubkey.pk")
    pyfhel_server.load_rotate_key("../rotkey.pk")
    arr_encoded = pyfhel_server.encodeInt(arr)
    arr_encrypted = pyfhel_server.encryptPtxt(arr_encoded)
    return arr_encrypted

'''
def enc_array_elements(arr: np.ndarray):
    pyfhel_server = Pyfhel()
    pyfhel_server.load_context("../context.con")
    pyfhel_server.load_public_key("../pubkey.pk")

    rows = []
    if arr.ndim > 1:
        for row in arr:
            row_encrypted = []
            for num in row:
                num_encoded = pyfhel_server.encode(num)
                num_encrypted = pyfhel_server.encrypt(num_encoded)
                row_encrypted.append(num_encrypted)
            #row_encoded = pyfhel_server.encode(row)
            #row_encrypted = pyfhel_server.encrypt(row_encoded)
            rows.append(row_encrypted)
        arr_encrypted = np.stack(rows, axis=0)
    else:
        arr_encoded = pyfhel_server.encode(arr)
        arr_encrypted = pyfhel_server.encrypt(arr_encoded)
    return arr_encrypted


def mul_matrix(arr1: np.ndarray, arr2: np.ndarray):
    arr1_rows, arr1_cols = arr1.shape
    arr2_rows, arr2_cols = arr2.shape
    enc_mult = []
    zero_encrypted = 0

    for row2 in arr2.transpose():
        enc_row = []
        for row1 in arr1:
            num_sum = zero_encrypted
            for i in range(arr1_cols):
                mult = row1[i] * row2[i]
                num_sum = num_sum + mult
            enc_row.append(num_sum)
        enc_mult.append(enc_row)

    return np.asarray(enc_mult)


def mult_enc_matrix(arr1: np.ndarray, arr2: np.ndarray):
    enc_mult = []
    pyfhel_server = Pyfhel()
    pyfhel_server.load_context("../context.con")
    pyfhel_server.load_public_key("../pubkey.pk")
    arr1_rows, arr1_cols = arr1.shape
    arr2_rows, arr2_cols = arr2.shape
    assert arr1_cols == arr2_rows, \
        "mult_enc_matrix: num of cols in array1 {} does not match num of rows in array2 {}".format(arr1_cols, arr2_rows)

    zero_encoded = pyfhel_server.encode(0)
    zero_encrypted = pyfhel_server.encrypt(zero_encoded)
    for row2 in arr2.transpose():
        enc_row = []
        for row1 in arr1:
            num_sum = zero_encrypted
            for i in range(arr1_cols):
                mult = pyfhel_server.multiply(row1[i], row2[i])
                num_sum = pyfhel_server.add(num_sum, mult)
            enc_row.append(num_sum)
        enc_mult.append(enc_row)

    return np.asarray(enc_mult)

'''

def pearson_correlation(meth_vals: np.array, ages: np.array) -> np.array:
    """
    alculate pearson correlation coefficient between rows of input methylation values and ages
    @param meth_vals: the methylation value array
    @param ages:  the age vector
    @return: the correlation value
    """
    # calculate mean for each row and ages mean
    matrix_means = np.mean(meth_vals, axis=1)
    ages_mean = np.mean(ages)

    # subtract means from observed values
    transformed_matrix = meth_vals - matrix_means.reshape([-1,1])
    transformed_ages = ages - ages_mean

    # calculate covariance
    covariance = np.sum(transformed_matrix * transformed_ages, axis=1)
    variance_meth = np.sqrt(np.sum(transformed_matrix ** 2, axis=1))
    variance_ages= np.sqrt(np.sum(transformed_ages ** 2))

    return covariance / (variance_meth * variance_ages)
