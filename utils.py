def estimate_max_val(ages, meth_vals):
    flat_meth_vals = meth_vals.flatten().tolist()
    ages_list = ages.tolist()
    m = ages.size
    n = meth_vals.flatten().size
    age = 9999
    iterations = 3
    factor = 1
    for i in range(iterations):
        max_age = age
        max_meth_val = 99
        lambda_1 = (m * max_age)**2
        r_i = factor * (m * (2 * m * max_age) * max_meth_val)
        s_i = m * (max_age * m * max_age + m * (max_age**2)) * max_meth_val

        age = n * r_i * (lambda_1 * max_meth_val + s_i)
        temp = n * factor * 6 * (m**4) * (max_age**3) * (max_meth_val**2)

        factor = m * (r_i ** 2)
        print("iter :", i, "estimated age: ", age, "age len: ", len(str(age)))

        print("alternate age: ", temp, "aget len: ", len(str(temp)))


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


def test_num_of_mult():
    pyfhelCtxt = Pyfhel()
    test_arr = np.array([2, 0, 0, 0])
    mult_arr = np.array([4, 0, 0, 0])
    square_arr = np.array([4, 0, 0, 0])
    for poly in range(14, 15):
        print("Generating context for poly:", poly, "\n")
        pyfhelCtxt.contextGen("bfv", n=2**poly, t_bits=20, sec=128)
        pyfhelCtxt.keyGen()
        pyfhelCtxt.rotateKeyGen()
        pyfhelCtxt.relinKeyGen()
        test_arr_encoded = pyfhelCtxt.encodeInt(test_arr)
        test_arr_encrypted = pyfhelCtxt.encryptPtxt(test_arr_encoded)
        mult_arr_encoded = pyfhelCtxt.encodeInt(mult_arr)
        mult_arr_encrypted = pyfhelCtxt.encryptPtxt(mult_arr_encoded)
        square_arr_encoded = pyfhelCtxt.encodeInt(square_arr)
        square_arr_encrypted = pyfhelCtxt.encryptPtxt(square_arr_encoded)
        i = 0
        square_arr_encrypted = square_arr_encrypted**2
        square_arr_encrypted = ~square_arr_encrypted
        noise_level = pyfhelCtxt.noise_level(test_arr_encrypted)
        while (noise_level > 0) and (i < 12):
            test_arr_encrypted = ~test_arr_encrypted
            test_arr_encrypted = test_arr_encrypted * mult_arr_encrypted
            noise_level = pyfhelCtxt.noise_level(test_arr_encrypted)
            print("iteration", i, " noise level: ", noise_level, "\n")
            i += 1
        print("Noise level reached 0 after ", i, "iterations\n")

        test_arr_encoded = pyfhelCtxt.encodeInt(test_arr)
        test_arr_encrypted = pyfhelCtxt.encryptPtxt(test_arr_encoded)
        i = 0
        noise_level = pyfhelCtxt.noise_level(test_arr_encrypted)
        while (noise_level > 0) and (i < 12):
            test_arr_encrypted = ~test_arr_encrypted
            if i < 2:
                test_arr_encrypted = test_arr_encrypted * mult_arr_encrypted
            else:
                test_arr_encrypted = test_arr_encrypted * square_arr_encrypted
            noise_level = pyfhelCtxt.noise_level(test_arr_encrypted)

            print("iteration ", i, " noise level: ", noise_level, "\n")
            i += 1
        print("Noise level reached 0 after ", i, "iterations\n")