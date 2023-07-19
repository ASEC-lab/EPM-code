import numpy as np
from CRT.CRT import Crt, P400, P500, P10, CRT_to_num, num_to_CRT, P300, P200, P100, CRT_array_to_matrix, CRT_vec_to_num, \
    P5
from CSP.Csp import Csp
import datetime
from EPM_no_sec.Epm import EPM
from DataHandler.DataSets import DataSets
from DataHandler.DataFormat import pearson_correlation
from DO.Do import DO
from time import time
from CSP.Cipher import Cipher, Decipher


def norm_2d(a: np.ndarray, b: np.ndarray):
    return np.divide(np.sqrt(np.sum((a - b) ** 2)), len(a))


def epm_orig(max_iterations: int = 3):
    """
    EPM implementation without encryption. Used for benchmarking.
    @return: calculated model params
    """
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
    model = epm.calc_model(iter_limit=max_iterations, error_tolerance=rss)
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


def epm_orig_new_method(max_iterations: int = 100, use_CRT: bool = True, Testing: bool = False) -> np.ndarray:
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

    formatted_ages = format_array_for_enc(train_ages)
    formatter_correlated_meth_val = format_array_for_enc(correlated_meth_val)

    formatted_ages[formatted_ages < 0] = 0
    formatter_correlated_meth_val[formatter_correlated_meth_val < 0] = 0

    # formatted_ages = train_ages
    # formatter_correlated_meth_val = correlated_meth_val
    # run the algorithm

    primes = P200 if use_CRT else None
    epm = EPM(formatter_correlated_meth_val, formatted_ages, primes=primes, use_crt=use_CRT)
    # ages = epm.calc_model_new_method()
    if use_CRT:
        ages = epm.calc_model_new_method_crt(max_iterations=max_iterations, Testing=Testing)
    else:
        ages = epm.calc_model_new_method(max_iterations=max_iterations, Testing=Testing)

    x = 2
    return ages


def test_do():
    # print("1")
    csp = CSP()
    # print("2")
    do = DO(csp)
    # print("3")
    ages = do.calc_model()
    return ages


def main():
    # this runs the encrypted version
    # ages = test_do()

    # epm cleartext testing using the new algorithm without division
    ##TODO : Check the next function
    """Meir function"""
    # ages = epm_orig_new_method(use_CRT=True, Testing=True)
    # print(ages)
    # gamma with CRT: 1043014041222916551500019474596939311273371568850647273081819490395836
    # epm cleartext testing using the original cleartext algorithm with division
    ##TODO: these are the correct results
    # ages_orig = epm_orig(max_iterations=1)
    x = 2
    # for age in ages2['ages']:
    #     print(age)
    #
    # exit(2)

    # ages_test = epm_orig_new_method(max_iterations=1, use_CRT=True)


    """Decipher Testing"""
    num1 = 1300
    num2 = -500
    num3 = -1550
    num4 = 0
    num5 = 1000
    P = P5
    N = np.prod(P)
    D1 = Decipher(number=num1, N=N, primes=P)
    D2 = Decipher(number=num2, N=N, primes=P)
    D3 = Decipher(number=num3, N=N, primes=P)
    D4 = Decipher(number=num4, N=N, primes=P)
    D5 = Decipher(number=num5, N=N, primes=P)

    # D_pos_pos = D1 + D5
    # D_pos_neg1 = D1 + D3
    # D_pos_neg2 = D1 + D2
    # D_neg_neg = D2 + D3


    D_neg_neg1 = D3 - D2
    D_neg_neg2 = D2 - D3

    D_pos_pos = D1 * D5
    D_pos_neg = D1 * D2
    D_neg_neg = D2 * D3

    # print("D_pos_pos: ", D_pos_pos.to_num())
    # print("D_pos_neg1: ", D_pos_neg1.to_num())
    # print("D_pos_neg2: ", D_pos_neg2.to_num())
    # print("D_neg_neg: ", D_neg_neg.to_num())

    print("D_pos_pos: ", D_pos_pos.to_num())
    print("D_pos_neg: ", D_pos_neg.to_num())
    print("D_neg_neg: ", D_neg_neg.to_num())


    # print("D_neg_neg1: ", D_neg_neg1.to_num())
    # print("D_neg_neg2: ", D_neg_neg2.to_num())






    exit(1)



    """CRT testing"""
    num1 = 1300
    num2 = -500
    num3 = -1550
    num4 = 0
    num5 = 1000

    P = P5
    N = np.prod(P)

    C1 = Crt(num=num1, dividers=P)
    C2 = Crt(num=num2, dividers=P)
    C3 = Crt(num=num3, dividers=P)
    C4 = Crt(num=num4, dividers=P)
    C5 = Crt(num=num5, dividers=P)

    C_sub = C2 - C3
    C_add = C5 + C3
    C_sub_pos = C5 - C1
    C_add_neg = C1 + C3
    # print(C2.Crt_to_num())
    print("subtraction: ", C_sub.Crt_to_num())
    print("addition: ", C_add.Crt_to_num())
    print("subtraction_positive: ", C_sub_pos.Crt_to_num())
    print("addition negative: ", C_add_neg.Crt_to_num())
    x = 2
    #
    # C = C1 * C4 + C3 * C2 + C4
    #
    # l = np.asarray([C1, C2, C3]).astype(object)
    # g = [x.moduli for x in l]
    # m = np.asarray(g).astype(np.int64)
    # for x in g:
    #     print(Crt(values=x, dividers=P).Crt_to_num())

    x = 2
    #
    # C = (C1*C2*C3)+(C1**3)
    # c_num = C.Crt_to_num()
    # real = num1*num2*num3 + num1**3
    #
    # print(C==real)

    # P3 = np.asarray([13, 17, 19, 23])
    # x = 7000
    # CRT = Crt(num=x, dividers=P3)
    # if(CRT==x):
    #     print("good")
    # else:
    #     print("bad")
    #
    # if CRT==x+1:
    #     print("bad")

    # num1 = 68141292
    # num2 = 6681392
    # num3 = 17924
    # sum = num1 + num2 + num3
    # CRT1 = Crt(num=num1, dividers=P400)
    # CRT2 = Crt(num=num2, dividers=P400)
    # CRT3 = Crt(num=num3, dividers=P400)
    #
    # c1 = CRT3-CRT2
    # c2 = CRT2-CRT3
    # print(c1.Crt_to_num(), num3-num2)
    # print(c2.Crt_to_num(), num2-num3)
    #
    # CRT_SUM = CRT1 + CRT2 + CRT3
    # print(CRT_SUM.Crt_to_num(), sum)
    # print()
    # #
    # #
    #
    # l = np.asarray([CRT1, CRT2, CRT3]).astype(object)
    # b = np.asarray([CRT3, CRT3, CRT3]).astype(object)
    # g = np.sum(l)
    # print(g.Crt_to_num())

    #
    # print("truth: {}".format(num1 * num3), end=" ")
    # g[0].print()
    #
    # print("truth: {}".format(num2 * num3), end=" ")
    # g[1].print()
    #
    # print("truth: {}".format(num3 * num3), end=" ")
    # g[2].print()

    """Comparing us to benchmark"""
    # for i in range(1,20):
    #     start = time()
    #     ages_new = epm_orig_new_method(max_iterations=i)
    #
    #
    #     time_new = time()-start
    #
    #     start=time()
    #     ages_bench = epm_orig_new_method_loay(max_iterations=i)
    #     time_bench = time()-start
    #
    #     norm = norm_2d(a=ages_new, b=ages_bench)
    #
    #     print("iteration: {}, time bench: {:.2f}s, time new: {:.2f}s, norm={:.2f}".format(i,time_bench, time_new, norm))


if __name__ == '__main__':
    main()
