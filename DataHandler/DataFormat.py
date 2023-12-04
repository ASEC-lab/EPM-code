import numpy as np

'''
a set of functions for formatting the input data

Coded by Meir Goldenberg  
meirgold@hotmail.com

'''

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


def format_array_for_dec(arr):
    '''
    the opposite of format_array_for_enc
    formats the array after decryption
    @param arr: the array to format
    @return: the formatted array
    '''
    result = np.array(arr)/(10 ** FLOATING_DIGIT_ROUND)
    return result

'''
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

def pearson_correlation(meth_vals: np.array, ages: np.array) -> np.array:
    """
    calculate pearson correlation coefficient between rows of input methylation values and ages
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
