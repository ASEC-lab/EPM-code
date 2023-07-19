from sympy.ntheory.modular import crt
from copy import deepcopy
import numpy as np
import math

NEGATIVE = "negative"
REPRESENTATION = "representation"

TESTING = False
NEG_METHOD = REPRESENTATION
P100 = np.asarray([547, 557, 563, 569, 571, 577, 587, 593, 599, 601, 607, 613, 617, 619, 631, 641, 643, 647, 653, 659])

P200 = np.asarray(
    [1229, 1231, 1237, 1249, 1259, 1277, 1279, 1283, 1289, 1291, 1297, 1301, 1303, 1307, 1319, 1321, 1327, 1361, 1367,
     1373])

P300 = np.asarray(
    [1993, 1997, 1999, 2003, 2011, 2017, 2027, 2029, 2039, 2053, 2063, 2069, 2081, 2083, 2087, 2089, 2099, 2111, 2113,
     2129])

P400 = np.asarray([2749, 2753, 2767, 2777, 2789, 2791, 2797, 2801, 2803, 2819, 2833, 2837, 2843, 2851, 2857, 2861, 2879,
                   2887, 2897, 2903])

P500 = np.asarray(
    [3581, 3583, 3593, 3607, 3613, 3617, 3623, 3631, 3637, 3643, 3659, 3671, 3673, 3677, 3691, 3697, 3701, 3709, 3719,
     3727])

P10 = np.asarray([2, 3, 5, 7, 11, 13, 17, 19, 23, 29])
P5 = np.asarray([13, 17, 19, 23, 29])

P30 = np.asarray([127, 131, 137, 139, 149, 151, 157, 163, 167, 173])
P20 = np.asarray([73, 79, 83, 89, 97, 101, 103, 107, 109, 113])


class Crt:

    def __init__(self, dividers: np.ndarray, num: int = 0, check: bool = True, values: np.ndarray = None,
                 is_negative=None):
        self.num = num if TESTING else 0
        self.dividers = dividers
        self.N = np.prod(dividers)
        if is_negative is None and NEG_METHOD == NEGATIVE:
            self.is_negative = True if num < 0 else False
        else:
            self.is_negative = is_negative
        if values is None and NEG_METHOD == NEGATIVE:
            self.moduli = np.mod(num, dividers).astype(np.int64)
        elif values is None and NEG_METHOD == REPRESENTATION:
            if num >= 0:
                self.moduli = np.mod(num, dividers).astype(np.int64)
            else:
                new_num = self.N + num
                self.moduli = np.mod(new_num, dividers).astype(np.int64)
        else:
            self.moduli = values

        if check and TESTING:
            if not self.check():
                print("CRT object {} invalid!".format(id(self)))
                print("Num: {},  CRT_to_num: {}".format(self.num, self.Crt_to_num()))

    def __add__(self, other, check: bool = True):
        moduli = np.add(self.moduli, other.moduli)
        moduli = np.mod(moduli, self.dividers)

        if check and not np.array_equal(self.dividers, other.dividers):
            print("CRT {} addition invalid!".format(id(self)))

        if NEG_METHOD == NEGATIVE:
            is_negative = self.Crt_to_num() + other.Crt_to_num() < 0
        else:
            is_negative = None
        return Crt(num=self.num + other.num, dividers=self.dividers, values=moduli, is_negative=is_negative)

    def __sub__(self, other, check: bool = True):
        moduli = np.subtract(self.moduli, other.moduli)
        moduli = np.mod(moduli, self.dividers)

        if check and not np.array_equal(self.dividers, other.dividers):
            print("CRT {} addition invalid!".format(id(self)))

        if NEG_METHOD == NEGATIVE:
            is_negative = self.Crt_to_num() - other.Crt_to_num() < 0
        else:
            is_negative = None
        return Crt(num=self.num - other.num, dividers=self.dividers, values=moduli, is_negative=is_negative)

    def __mul__(self, other, check: bool = True):
        moduli = np.multiply(self.moduli, other.moduli)
        moduli = np.mod(moduli, self.dividers)

        if check and not np.array_equal(self.dividers, other.dividers):
            print("CRT {} addition invalid!".format(id(self)))

        if NEG_METHOD==NEGATIVE:
            is_negative = self.is_negative ^ other.is_negative
        else:
            is_negative = None
        return Crt(num=self.num * other.num, dividers=self.dividers, values=moduli, is_negative=is_negative)

    def __pow__(self, power: int, check=True):
        moduli = np.power(self.moduli, power)
        moduli = np.mod(moduli, self.dividers)
        if NEG_METHOD == NEGATIVE:
            is_negative = self.Crt_to_num() ** power < 0
        else:
            is_negative = None
        return Crt(num=self.num ** power, dividers=self.dividers, values=moduli, is_negative=is_negative)

    def __eq__(self, x: int) -> bool:
        return x == self.Crt_to_num()

    def check(self):
        if self.Crt_to_num() == self.num:
            return True
        return False

    def max_val(self):
        dividers = deepcopy(self.dividers)
        dividers = np.log(dividers)
        return np.sum(dividers)

    def Crt_to_num(self):
        if NEG_METHOD == NEGATIVE:
            if self.is_negative:
                minus_one = Crt(num=-1, dividers=self.dividers)
                this_crt = deepcopy(self) * minus_one
                return -this_crt.Crt_to_num()
            else:
                return CRT_to_num(values=self.moduli, div=self.dividers)
        else:
            # neg method is based on representation
            num = CRT_to_num(values=self.moduli, div=self.dividers)
            if num <= self.N / 2:
                return num
            else:
                return num - self.N

    def print(self):
        print("num: {}, CRT representation: {}".format(self.num, self.Crt_to_num()))


def CRT_array_to_matrix(array: np.ndarray[Crt]) -> np.ndarray[np.int64]:
    """
    @param array: ndarray of CRT objects
    @return: a matrix, of which line i is the moduli of the i-th crt element in array
    """
    return np.asarray([x.moduli for x in array]).astype(np.int64)


def matrix_to_CRT_array(matrix: np.ndarray, dividers: np.ndarray) -> np.ndarray[Crt]:
    A = np.asarray([Crt(values=vector, dividers=dividers) for vector in matrix]).astype(object)
    return A


def CRT_to_num(values, div):
    """
    @param values: moduli values
    @param div: dividers
    @return: the corresponding number
    """
    return crt(div, values)[0]


def CRT_vec_to_num(values: np.ndarray[Crt]):
    """
    @param values: an ndarray of Crt objects
    @return: a vector of numbers, where the i-th number is the natural representation of the i-th Crt element
    """

    return np.asarray([x.Crt_to_num() for x in values], dtype=np.float64)


def num_to_CRT(x, div):
    """
    @param x: number
    @param div: division vector (CRT)
    @return: a corresponding CRT vector
    """
    # return np.array([x % div[i] for i in range(len(div))])
    return np.mod(x, div)


def CRT_addition(values1, values2, div):
    return np.array([(values1[i] + values2[i]) % div[i] for i in range(len(div))])


def CRT_subtraction(values1, values2, div):
    return np.array([np.abs(values1[i] - values2[i]) % div[i] for i in range(len(div))])


def CRT_multiplication(values1, values2, div):
    return np.array([(values1[i] * values2[i]) % div[i] for i in range(len(div))])


def CRT_multiplication_mat(matt, arr, div):
    res = []
    for i in range(matt.shape[0]):
        res.append(CRT_multiplication(matt[i], arr, div))
    return np.array(res)


def CRT_addition_2darr(matt, div):
    return np.array([np.sum(matt[:, i]) % div[i] for i in range(len(div))])


def CRT_subtract_elementwise(mat, vec, div):
    res = np.zeros((mat.shape[0], mat.shape[1], mat.shape[2]))
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            res[i, j]: CRT_subtraction(mat[i, j], vec[j], div)
    return res


def CRT_multiplicate_dotwise(mat, vec, div):
    res = np.zeros((mat.shape[0], mat.shape[1], mat.shape[2]))
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            res[i, j] = CRT_multiplication(mat[i, j], vec[j], div)
    return res


def CRT_sum_square(vec, div):
    res = np.zeros((vec.shape[0], vec.shape[1]))
    for i in range(vec.shape[0]):
        res[i] = CRT_multiplication(vec[i], vec[i], div)
    return np.sum(res, axis=0)
