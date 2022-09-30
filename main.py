import numpy as np
from CSP.Csp import CSP
from MLE.Mle import MLE
import datetime
from Pyfhel import PyCtxt, Pyfhel, PyPtxt
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
    correlated_meth_val_indices = np.where(abs_pcc_coefficients > .80)[0]
    correlated_meth_val = train_methylation_values[correlated_meth_val_indices, :]
    # run the algorithm
    epm = EPM(correlated_meth_val, train_ages)
    model = epm.calc_model(max_iterations, rss)
    return model

def epm_orig_new_method():
    max_iterations = 100
    rss = 0.0000001
    data_sets = DataSets()
    # read training data
    full_train_data = data_sets.get_example_train_data()
    train_samples, train_cpg_sites, train_ages, train_methylation_values = full_train_data
    # run pearson correlation in order to reduce the amount of processed data
    abs_pcc_coefficients = abs(pearson_correlation(train_methylation_values, train_ages))
    correlated_meth_val_indices = np.where(abs_pcc_coefficients > .80)[0]
    correlated_meth_val = train_methylation_values[correlated_meth_val_indices, :]
    # run the algorithm
    epm = EPM(correlated_meth_val, train_ages)
    ages = epm.calc_model_new_method()
    return ages

'''
def main_old():
    print("Calculating encrypted model")
    start_time = datetime.datetime.now()
    enc_epm_model = encrypted_epm()
    enc_end_time = datetime.datetime.now()
    print("Calculating non-encrypted model")
    orig_model = epm_orig()
    orig_model_end_time = datetime.datetime.now()

    with open("output.txt", 'w') as f:
        f.write("Encrypted model:\n")
        for key, value in enc_epm_model.items():
            f.write('%s:%s\n' % (key, value))

        f.write("Original model:\n")
        for key, value in orig_model.items():
            f.write('%s:%s\n' % (key, value))

        f.write("enc model calc start: {}\n".format(start_time))
        f.write("enc model calc end: {}\n".format(enc_end_time))
        f.write("non-enc model calc end: {}\n".format(orig_model_end_time))
    print(enc_epm_model)
    print(orig_model)
'''


def fhe_test2():
    HE = Pyfhel()  # Creating empty Pyfhel object
    bfv_params = {
        'scheme': 'BFV',  # can also be 'bfv'
        'n': 2 ** 13,  # Polynomial modulus degree, the num. of slots per plaintext,
        #  of elements to be encoded in a single ciphertext in a
        #  2 by n/2 rectangular matrix (mind this shape for rotations!)
        #  Typ. 2^D for D in [10, 16]
        't': 65537,  # Plaintext modulus. Encrypted operations happen modulo t
        #  Must be prime such that t-1 be divisible by 2^N.
        't_bits': 20,  # Number of bits in t. Used to generate a suitable value
        #  for t. Overrides t if specified.
        'sec': 128,  # Security parameter. The equivalent length of AES key in bits.
        #  Sets the ciphertext modulus q, can be one of {128, 192, 256}
        #  More means more security but also slower computation.
    }
    HE.contextGen(**bfv_params)  # Generate context for bfv scheme
    HE.keyGen()  # Key Generation: generates a pair of public/secret keys
    HE.rotateKeyGen()  # Rotate key generation --> Allows rotation/shifting
    HE.relinKeyGen()  # Relinearization key generation

    print("\n1. Pyfhel FHE context generation")
    print(f"\t{HE}")

    arr1 = np.arange(bfv_params['n'], dtype=np.int64)  # Max possible value is t/2-1. Always use type int64!
    arr2 = np.array([-bfv_params['t'] // 2, -1, 1], dtype=np.int64)  # Min possible value is -t/2.
    arr3 = np.array([50])


    ptxt1 = HE.encodeInt(arr1)  # Creates a PyPtxt plaintext with the encoded arr1
    ptxt2 = HE.encodeInt(arr2)  # plaintexts created from arrays shorter than 'n' are filled with zeros.
    ptxt3 = HE.encodeInt(arr3)

    ctxt1 = HE.encryptPtxt(ptxt1)  # Encrypts the plaintext ptxt1 and returns a PyCtxt
    ctxt2 = HE.encryptPtxt(ptxt2)  # Alternatively you can use HE.encryptInt(arr2)
    ctxt3 = HE.encryptPtxt(ptxt3)

    # Otherwise, a single call to `HE.encrypt` would detect the data type,
    #  encode it and encrypt it
    # > ctxt1 = HE.encrypt(arr1)

    print("\n2. Integer Encoding & Encryption, ")
    print("->\tarr1 ", arr1, '\n\t==> ptxt1 ', ptxt1, '\n\t==> ctxt1 ', ctxt1)
    print("->\tarr2 ", arr2, '\n\t==> ptxt2 ', ptxt2, '\n\t==> ctxt2 ', ctxt2)
    print("->\tarr3 ", arr3, '\n\t==> ptxt3 ', ptxt3, '\n\t==> ctxt3 ', ctxt3)
    HE.add()
    # Ciphertext-ciphertext ops:
    ccSum = ctxt1 + ctxt2  # Calls HE.add(ctxt1, ctxt2, in_new_ctxt=True)
    #  `ctxt1 += ctxt2` for inplace operation
    ccSub = ctxt1 - ctxt2  # Calls HE.sub(ctxt1, ctxt2, in_new_ctxt=True)
    #  `ctxt1 -= ctxt2` for inplace operation
    ccMul = ctxt1 * ctxt2  # Calls HE.multiply(ctxt1, ctxt2, in_new_ctxt=True)
    #  `ctxt1 *= ctxt2` for inplace operation
    cSq = ctxt1 ** 2  # Calls HE.square(ctxt1, in_new_ctxt=True)
    #  `ctxt1 **= 2` for inplace operation
    cNeg = -ctxt1  # Calls HE.negate(ctxt1, in_new_ctxt=True)
    #
    cPow = ctxt1 ** 3  # Calls HE.power(ctxt1, 3, in_new_ctxt=True)
    #  `ctxt1 **= 3` for inplace operation
    cRotR = ctxt1 >> 2  # Calls HE.rotate(ctxt1, k=2, in_new_ctxt=True)
    #  `ctxt1 >>= 2` for inplace operation
    # WARNING! the encoded data is placed in a n//2 by 2
    #  matrix. Hence, these rotations apply independently
    #  to each of the rows!
    cRotL = ctxt1 << 2  # Calls HE.rotate(ctxt1, k=-2, in_new_ctxt=True)
    #  `ctxt1 <<= 2` for inplace operation

    # Ciphetext-plaintext ops
    cpSum = ctxt1 + ptxt2  # Calls HE.add_plain(ctxt1, ptxt2, in_new_ctxt=True)
    # `ctxt1 += ctxt2` for inplace operation
    cpSub = ctxt1 - ptxt2  # Calls HE.sub_plain(ctxt1, ptxt2, in_new_ctxt=True)
    # `ctxt1 -= ctxt2` for inplace operation
    cpMul = ctxt1 * ptxt2  # Calls HE.multiply_plain(ctxt1, ptxt2, in_new_ctxt=True)
    # `ctxt1 *= ctxt2` for inplace operation

    print("3. Secure operations")
    print(" Ciphertext-ciphertext: ")
    print("->\tctxt1 + ctxt2 = ccSum: ", ccSum)
    print("->\tctxt1 - ctxt2 = ccSub: ", ccSub)
    print("->\tctxt1 * ctxt2 = ccMul: ", ccMul)
    print(" Single ciphertext: ")
    print("->\tctxt1**2      = cSq  : ", cSq)
    print("->\t- ctxt1       = cNeg : ", cNeg)
    print("->\tctxt1**3      = cPow : ", cPow)
    print("->\tctxt1 >> 2    = cRotR: ", cRotR)
    print("->\tctxt1 << 2    = cRotL: ", cRotL)
    print(" Ciphertext-plaintext: ")
    print("->\tctxt1 + ptxt2 = cpSum: ", cpSum)
    print("->\tctxt1 - ptxt2 = cpSub: ", cpSub)
    print("->\tctxt1 * ptxt2 = cpMul: ", cpMul)

    print("\n4. Relinearization-> Right after each multiplication.")
    print(f"ccMul before relinearization (size {ccMul.size()}): {ccMul}")
    ~ccMul  # Equivalent to HE.relinearize(ccMul). Relin always happens in-place.
    print(f"ccMul after relinearization (size {ccMul.size()}): {ccMul}")
    print(f"cPow after 2 mult&relin rounds:  (size {cPow.size()}): {cPow}")

    r1 = HE.decryptInt(ctxt1)
    r2 = HE.decryptInt(ctxt2)
    rccSum = HE.decryptInt(ccSum)
    rccSub = HE.decryptInt(ccSub)
    rccMul = HE.decryptInt(ccMul)
    rcSq = HE.decryptInt(cSq)
    rcNeg = HE.decryptInt(cNeg)
    rcPow = HE.decryptInt(cPow)
    rcRotR = HE.decryptInt(cRotR)
    rcRotL = HE.decryptInt(cRotL)
    rcpSum = HE.decryptInt(cpSum)
    rcpSub = HE.decryptInt(cpSub)
    rcpMul = HE.decryptInt(cpMul)

    r3 = HE.decryptInt(ctxt3)
    print("5. Decrypting results")
    print(" Original ciphertexts: ")
    print("   ->\tctxt1 --(decr)--> ", r1)
    print("   ->\tctxt2 --(decr)--> ", r2)
    print(" Ciphertext-ciphertext Ops: ")
    print("   ->\tctxt1 + ctxt2 = ccSum --(decr)--> ", rccSum)
    print("   ->\tctxt1 - ctxt2 = ccSub --(decr)--> ", rccSub)
    print("   ->\tctxt1 * ctxt2 = ccMul --(decr)--> ", rccMul)
    print(" Single ciphertext: ")
    print("   ->\tctxt1**2      = cSq   --(decr)--> ", rcSq)
    print("   ->\t- ctxt1       = cNeg  --(decr)--> ", rcNeg)
    print("   ->\tctxt1**3      = cPow  --(decr)--> ", rcPow)
    print("   ->\tctxt1 >> 2    = cRotR --(decr)--> ", rcRotR)
    print("   ->\tctxt1 << 2    = cRotL --(decr)--> ", rcRotL)
    print(" Ciphertext-plaintext ops: ")
    print("   ->\tctxt1 + ptxt2 = cpSum --(decr)--> ", rcpSum)
    print("   ->\tctxt1 - ptxt2 = cpSub --(decr)--> ", rcpSub)
    print("   ->\tctxt1 * ptxt2 = cpMul --(decr)--> ", rcpMul)
    print("r3: ", r3)


def fhe_test():
    HE = Pyfhel()  # Creating empty Pyfhel object
    bfv_params = {
        'scheme': 'BFV',  # can also be 'bfv'
        'n': 2 ** 13,  # Polynomial modulus degree, the num. of slots per plaintext,
        #  of elements to be encoded in a single ciphertext in a
        #  2 by n/2 rectangular matrix (mind this shape for rotations!)
        #  Typ. 2^D for D in [10, 16]
        't': 65537,  # Plaintext modulus. Encrypted operations happen modulo t
        #  Must be prime such that t-1 be divisible by 2^N.
        't_bits': 20,  # Number of bits in t. Used to generate a suitable value
        #  for t. Overrides t if specified.
        'sec': 128,  # Security parameter. The equivalent length of AES key in bits.
        #  Sets the ciphertext modulus q, can be one of {128, 192, 256}
        #  More means more security but also slower computation.
    }
    HE.contextGen(**bfv_params)  # Generate context for bfv scheme
    HE.keyGen()  # Key Generation: generates a pair of public/secret keys
    HE.rotateKeyGen()  # Rotate key generation --> Allows rotation/shifting
    HE.relinKeyGen()  # Relinearization key generation

    print("\n1. Pyfhel FHE context generation")
    print(f"\t{HE}")

    # arr_big = np.array(range(18000))
    arr3 = np.array([1, 2, 3])
    arr_neg = np.array([-1, -7, 3])
    # ptxt_big = HE.encodeInt(arr_big)
    # ctxt_big = HE.encryptPtxt(ptxt_big)
    ptxt3 = HE.encodeInt(arr3)
    ctxt3 = HE.encryptPtxt(ptxt3)
    ptxt_arr_neg = HE.encodeInt(arr_neg)
    ctxt_arr_neg = HE.encryptPtxt(ptxt_arr_neg)


    ccsum = ctxt3 << 2
    mul_by_neg = ccsum * ctxt_arr_neg
    ccsquare = ctxt3 ** 2
    ~ccsquare
    ccsquare_shift = ccsquare << 2
    ccmult_regular_num = ctxt3 * 100
    print(ctxt3)
    ccsum_dec = HE.decryptInt(ccsum)
    print("r3: ", ccsum_dec[8191])
    ccsquare_dec = HE.decryptInt(ccsquare)
    print("ccsquare: ", ccsquare_dec)
    ccmult_regular_num_dec = HE.decryptInt(ccmult_regular_num)
    print("ccmult_regular_num: ", ccmult_regular_num_dec)
    mul_by_neg_dec = HE.decryptInt(mul_by_neg)
    print("mul by neg: ", mul_by_neg_dec)


def test_recrypt():
    HE = Pyfhel()  # Creating empty Pyfhel object
    HE.contextGen("bfv", n=2 ** 13, t_bits=20, sec=128)
    HE.keyGen()  # Key Generation: generates a pair of public/secret keys
    HE.rotateKeyGen()  # Rotate key generation --> Allows rotation/shifting
    HE.relinKeyGen()  # Relinearization key generation

    print("\n1. Pyfhel FHE context generation")
    print(f"\t{HE}")

    arr3 = np.array([1, 2, 3])
    arr_neg = np.array([-1, -7, 3])
    ptxt3 = HE.encodeInt(arr3)
    ctxt3 = HE.encryptPtxt(ptxt3)
    ptxt_arr_neg = HE.encodeInt(arr_neg)
    ctxt_arr_neg = HE.encryptPtxt(ptxt_arr_neg)
    step = 0
    lvl = HE.noise_level(ctxt3)
    while lvl > 0:
        print(f"\tStep {step}: noise_lvl {lvl}, res {HE.decryptInt(ctxt3)[:4]}")
        step += 1
        result = ctxt3 * ctxt_arr_neg  # Multiply in-place
        ctxt3 = ~(ctxt3)  # Always relinearize after each multiplication!
        lvl = HE.noise_level(ctxt3)

    print(f"\tFinal Step {step}: noise_lvl {lvl}, res {HE.decryptInt(ctxt3)[:4]}")
    print("---------------------------------------")


def test_mle():
    csp = CSP()
    mle = MLE(csp)
    m = 3
    n = 2
    Y = np.array([2, 5, 9, 7, 20, 15], dtype=np.int64)
    ages = np.array([10, 11, 12], dtype=np.int64)
    encrypted_Y = csp.encrypt_array(Y)
    encrypted_ages = csp.encrypt_array(ages)
    rates, s0 = mle.calc_beta_corollary1(m, n, ages, Y)
    print(rates)
    print(s0)
    ages = mle.site_step()
    rates, s0 = mle.adapted_site_step(m, n, encrypted_ages, encrypted_Y, 1)
    print(csp.decrypt_arr(rates))
    print(csp.decrypt_arr(s0))

    dummy_rates = np.array([2, 3])
    dummy_s0 = np.array([50, 60])
    dummy_meth_vals = np.array([1, 2, 3, 4, 5, 6])
    encrypted_dummy_meth_vals = csp.encrypt_array(dummy_meth_vals)
    encrypted_dummy_rates = csp.encrypt_array(dummy_rates)
    encrypted_dummy_s0 = csp.encrypt_array(dummy_s0)
    enc_gamma = csp.encrypt_array(np.array([5]))
    mle.adapted_time_step(encrypted_dummy_rates, encrypted_dummy_s0, encrypted_dummy_meth_vals, 2, 3, enc_gamma)

def test_do():
    csp = CSP()
    do = DO(csp)
    meth_vals = np.arange(0, 12000, 1,  dtype=np.int64)
    meth_vals = meth_vals.reshape(30, 400)
    ages = np.array([10, 11, 12], dtype=np.int64)
    #do.encrypt_train_data(meth_vals, ages)
    do.calc_model()

def main():
    # fhe_test()
    #test_mle()
    test_do()

    #test_recrypt()

    '''
    # epm cleartext testing
    model = epm_orig()
    np.savetxt('orig.out', model['ages'], delimiter=',')
    ages = epm_orig_new_method()
    np.savetxt('new.out', ages, delimiter=',')
    '''

    '''
    csp = CSP()
    do = DO(csp)
    do.calc_model()
    '''
if __name__ == '__main__':
    main()

