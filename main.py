import numpy as np
from CSP.Csp import CSP
import datetime
from EPM_no_sec.Epm import EPM
from DataHandler.DataSets import DataSets
from DataHandler.DataFormat import pearson_correlation
from Math_Utils.MathUtils import find_primes, read_primes_from_file
from DO.Do import DO
from Pyfhel import PyCtxt, Pyfhel, PyPtxt
import sys
from sympy.ntheory.modular import crt
import math

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
    full_train_data = data_sets.reduce_data_size(full_train_data, 7503, 25)
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

    formatted_ages = format_array_for_enc(train_ages)
    formatter_correlated_meth_val = format_array_for_enc(correlated_meth_val)

    #formatted_ages = train_ages
    #formatter_correlated_meth_val = correlated_meth_val
    # run the algorithm
    epm = EPM(formatter_correlated_meth_val, formatted_ages)
    ages = epm.calc_model_new_method()
    return ages

def test_do():
    do = DO()
    ages = do.calc_model()
    return ages

def bgv_mult():
    HE = Pyfhel()
    bgv_params = {
        'scheme': 'BGV',
        'n': 2 ** 13,
        't': 65537,
        'sec': 128,
    }
    HE.contextGen(**bgv_params)  # Generate context for bgv scheme
    HE.keyGen()  # Key Generation: generates a pair of public/secret keys
    HE.rotateKeyGen()  # Rotate key generation --> Allows rotation/shifting
    HE.relinKeyGen()  # Relinearization key generation
    integer1 = np.array([127], dtype=np.int64)
    ctxt1 = HE.encryptBGV(integer1)
    ctxtMul1 = ctxt1 * 2


def bfv_test_prime(prime):
    HE = Pyfhel()  # Creating empty Pyfhel object
    bfv_params = {
        'scheme': 'BFV',  # can also be 'bfv'
        'n': 2 ** 13,  # Polynomial modulus degree, the num. of slots per plaintext,
        #  of elements to be encoded in a single ciphertext in a
        #  2 by n/2 rectangular matrix (mind this shape for rotations!)
        #  Typ. 2^D for D in [10, 16]
        't': prime, #65537, # 1462436364289,  # Plaintext modulus. Encrypted operations happen modulo t
        #  Must be prime such that t-1 be divisible by 2^N.
        #'t_bits': 20,  # Number of bits in t. Used to generate a suitable value
        #  for t. Overrides t if specified.
        'sec': 128,  # Security parameter. The equivalent length of AES key in bits.
        #  Sets the ciphertext modulus q, can be one of {128, 192, 256}
        #  More means more security but also slower computation.
    }
    HE.contextGen(**bfv_params)  # Generate context for bfv scheme
    HE.keyGen()  # Key Generation: generates a pair of public/secret keys
    HE.rotateKeyGen()  # Rotate key generation --> Allows rotation/shifting
    HE.relinKeyGen()  # Relinearization key generation

    arr1 = np.array([8000, 9000, 1010], dtype=np.int64)
    arr2 = np.array([-8, 0, 0], dtype=np.int64)
    arr3 = np.array([20000, 0, 0], dtype=np.int64)

    ptxt1 = HE.encodeInt(arr1)  # Creates a PyPtxt plaintext with the encoded arr1
    ptxt2 = HE.encodeInt(arr2)  # plaintexts created from arrays shorter than 'n' are filled with zeros.
    ptxt3 = HE.encodeInt(arr3)
    ctxt1 = HE.encryptPtxt(ptxt1)  # Encrypts the plaintext ptxt1 and returns a PyCtxt
    ctxt2 = HE.encryptPtxt(ptxt2)  # Alternatively you can use HE.encryptInt(arr2)
    ctxt3 = HE.encryptPtxt(ptxt3)  # Alternatively you can use HE.encryptInt(arr2)

    ccMul = ctxt1 * ctxt2  # Calls HE.multiply(ctxt1, ctxt2, in_new_ctxt=True)
    ccMul = ~ccMul
    ccMul = ccMul + ctxt3
    ccMul = ccMul * ctxt2
    ccMul = ccMul * ctxt2
    rccMul = HE.decryptInt(ccMul)

    print("real result: ", (8000 * (-8) + 20000) * (-8) * (-8))
    return rccMul[0]


def bfv_test():
    primes = [65537, 114689]
    #primes = [114689]
    results = []
    for prime in primes:
        res = bfv_test_prime(prime)
        results.append(res)
    print(results)
    crt_result = crt(primes, results)[0]
    primes_mult = np.prod(primes)
    if crt_result > primes_mult:
        crt_result = primes_mult - crt_result

    print(crt_result)


def bgv_test():
    HE = Pyfhel()  # Creating empty Pyfhel object

    # HE.contextGen(scheme='bgv', n=2**14, t_bits=20)  # Generate context for 'bfv'/'bgv'/'ckks' scheme

    bgv_params = {
        'scheme': 'BGV',  # can also be 'bgv'
        'n': 2 ** 13,  # Polynomial modulus degree, the num. of slots per plaintext,
        #  of elements to be encoded in a single ciphertext in a
        #  2 by n/2 rectangular matrix (mind this shape for rotations!)
        #  Typ. 2^D for D in [10, 16]
        't': 1462436364289, #114689,  # Plaintext modulus. Encrypted operations happen modulo t
        #  Must be prime such that t-1 be divisible by 2^N.
        #'t_bits': 20,  # Number of bits in t. Used to generate a suitable value
        #  for t. Overrides t if specified.
        'sec': 128,  # Security parameter. The equivalent length of AES key in bits.
        #  Sets the ciphertext modulus q, can be one of {128, 192, 256}
        #  More means more security but also slower computation.
    }
    HE.contextGen(**bgv_params)  # Generate context for bgv scheme
    HE.keyGen()  # Key Generation: generates a pair of public/secret keys
    HE.rotateKeyGen()  # Rotate key generation --> Allows rotation/shifting
    HE.relinKeyGen()  # Relinearization key generation

    print("\n2. Pyfhel FHE context generation")
    print(f"\t{HE}")

    integer1 = np.array([127], dtype=np.int64)
    integer2 = np.array([-2], dtype=np.int64)
    ctxt1 = HE.encryptBGV(integer1)  # Encryption makes use of the public key
    ctxt2 = HE.encryptBGV(integer2)  # For BGV, encryptBGV function is used.
    print("\n3. BGV Encryption, ")
    print("    int ", integer1, '-> ctxt1 ', type(ctxt1))
    print("    int ", integer2, '-> ctxt2 ', type(ctxt2))

    ctxtSum = ctxt1 + ctxt2  # `ctxt1 += ctxt2` for inplace operation
    ctxtSub = ctxt1 - ctxt2  # `ctxt1 -= ctxt2` for inplace operation
    ctxtMul = ctxt1 * ctxt2  # `ctxt1 *= ctxt2` for inplace operation
    print("\n4. Operating with encrypted integers")
    print(f"Sum: {ctxtSum}")
    print(f"Sub: {ctxtSub}")
    print(f"Mult:{ctxtMul}")

    resSum = HE.decryptBGV(ctxtSum)  # Decryption must use the corresponding function
    #  decryptBGV.
    resSub = HE.decrypt(ctxtSub)  # `decrypt` function detects the scheme and
    #  calls the corresponding decryption function.
    resMul = HE.decryptBGV(ctxtMul)
    print("\n5. Decrypting result:")
    print("     addition:       decrypt(ctxt1 + ctxt2) =  ", resSum)
    print("     substraction:   decrypt(ctxt1 - ctxt2) =  ", resSub)
    print("     multiplication: decrypt(ctxt1 + ctxt2) =  ", resMul)


def test_read_primes():
    primes = read_primes_from_file("primes.txt")
    print(primes)

def test_primes():
    sys.set_int_max_str_digits(0)
    f = open("primes.txt", "w")
    j = 1
    prime_list = find_primes(16384**2, 8192,400)
    #prime_list = find_primes(8192, 8192, 400)
    print(prime_list)
    for prime in prime_list:
        j *= prime
        f.write("%s\n" % prime)
    print(j)
    f.close()

def mult_primes():
    mul_val = 1
    primes = []
    with open("primes.txt", 'r') as fp:
        for line in fp:
            x = line[:-1]
            primes.append(int(x))

    for i in range(10):
        mul_val = mul_val * primes[i]
        print(i, ": ", mul_val, "len: ", len(str(mul_val)))

def bgv_test2():
    #primes = [26836993, 26787841, 26714113, 26591233]
    primes = [24159191041, 29796335617, 34896609281, 38923141121]

    modulu_results = []
    pyCtxts = []
    i = 0
    a = 25
    b = 37
    for prime in primes:
        pyCtxts.append(Pyfhel())
        pyCtxts[i].contextGen("bfv", n=2 ** 13, t=prime, sec=128)

        pyCtxts[i].keyGen()
        pyCtxts[i].rotateKeyGen()
        pyCtxts[i].relinKeyGen()

        a_encoded = pyCtxts[i].encode(np.array([a], dtype=int))
        b_encoded = pyCtxts[i].encode(np.array([b], dtype=int))
        Pa = pyCtxts[i].encryptPtxt(a_encoded)
        Pb = pyCtxts[i].encryptPtxt(b_encoded)

        Pa_2 = Pa * Pa
        Pa_2 = ~Pa_2
        Pa_3 = Pa_2 * Pa
        Pa_3 = ~Pa_3
        Pa_4 = Pa_3 * Pa

        Pb_2 = Pb * Pb
        Pb_2 = ~Pb_2

        dec_array = pyCtxts[i].decrypt(mul)
        modulu_results.append(dec_array[0])
        i += 1

    if len(primes) > 1:
        result = crt(primes, modulu_results)
    else:
        result = modulu_results
    print(result[0])


def main():
    #test_read_primes()
    #test_primes()
    #bgv_test()
    #bgv_mult()
    #bfv_test()

    #bgv_test2()

    # this runs the encrypted version
    ages = test_do()
    # epm cleartext testing using the new algorithm without division
    #ages = epm_orig_new_method()
    print(ages)
    #mult_primes()
    # epm cleartext testing using the original cleartext algorithm with division
    # ages = epm_orig()


if __name__ == '__main__':
    main()

