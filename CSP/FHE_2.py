from Pyfhel import PyCtxt, Pyfhel, PyPtxt
from typing import Dict, List, Tuple
import numpy as np
from time import time
from sympy.ntheory.modular import crt
import math
from Cipher import Cipher, Decipher

DEBUG = True
MIN = 13
OFFSET = 4


def get_primes() -> np.ndarray:
    P = np.load('primes.npy')
    return P


Default_Params = {
    'sec': 128,
}

BGV = 'BGV'
BFV = 'BFV'
# 17, 18, 20-42, 45-60,
BitSizes = set([17, 18] + list(range(20, 43)) + list(range(45, 61)))


def bit_size(num: int) -> int:
    if num > 0:
        return math.ceil(math.log(num, 2))
    else:
        return None


def pyfhel_builder(params: dict) -> Pyfhel:
    H = Pyfhel()  # Creating empty Pyfhel object
    # if params.get('t_bits') not in BitSizes:
    #     print("Cannot build Pyfhel with {} bits.".format(params['t_bits']))
    H.contextGen(**params)  # Generate context for bfv scheme
    H.keyGen()  # Key Generation: generates a pair of public/secret keys
    H.rotateKeyGen()  # Rotate key generation --> Allows rotation/shifting
    H.relinKeyGen()  # Relinearization key generation
    return H


def product(array):
    return math.prod([int(x) for x in array])
    # p = 1
    # for value in array:
    #     _value = int(value)
    #     p = p*_value
    # return p


def factorial_wobbly(array):
    return 2 ** sum([bit_size(x) for x in array])


def get_next_prime(index: int, Primes: np.ndarray[List[int]]) -> Tuple[int, int]:
    prime, n = Primes[index][0], Primes[index][1]
    if n > MIN:
        n = MIN
    return prime, n


class FHE:
    params = None
    Primes = None
    length = None
    label = None
    scheme = None
    N = None
    E = List[Pyfhel]

    def __init__(self, params: Dict[str, int] = None, label: int = 0, number_of_primes: int = 9,
                 scheme: str = BGV, _N: int = 0):
        """
        @param params:
        @param label:
        @param number_of_primes:
        @param scheme:
        @param starting_bits:
        """
        start = time()
        if params is None: params = Default_Params
        params['scheme'] = scheme

        self.scheme = scheme
        self.params = params
        self.length = number_of_primes
        self.E, self.Primes = [], []
        self.label = label

        _Primes = get_primes()

        for i in range(number_of_primes):
            _t, _n = get_next_prime(index=i+OFFSET, Primes=_Primes)
            params['t'] = _t
            params['n'] = 2 ** _n
            F = pyfhel_builder(params=params)
            self.E.append(F)
            self.Primes.append(F.t)

        self.Primes = np.asarray(self.Primes, dtype=np.int64)
        if _N > 0:
            self.N = _N
        else:
            self.N = product(self.Primes)

        if DEBUG:
            print("FHE object with scheme {} and {} primes constructed in {:.2f} seconds".format(scheme, self.length,
                                                                                                 time() - start))
            self.verify_E()

    def get_primes(self) -> np.ndarray[np.int64]:
        return self.Primes

    def verify_E(self):
        for i in range(len(self.E)):
            if self.E[i].t != self.Primes[i]:
                print("E[{}] invalid; t={}, supposed to be: {}".format(i, self.E[i].t, self.Primes[i]))
        if self.length != len(self.E):
            print("WARNING: length mismatch. E: {}, Primes: {}")
            self.length = len(self.E)

    def decrypt_2D(self, ciphers: List[Cipher]) -> np.ndarray[Decipher]:
        result = [self.decrypt_1D(ciphers[i]) for i in range(len(ciphers))]
        return np.asarray(result, dtype=object)

    def decrypt_1D(self, cipher: Cipher, name: str = None) -> Decipher:
        values = np.asarray([self.E[i].decrypt(cipher.values[i])[0] for i in range(self.length)], dtype=np.int64)
        return Decipher(values=values, N=self.N, primes=self.Primes, name=name)

    def encrypt_1D(self, v: Decipher) -> Cipher:
        ciphers = []
        if self.scheme == BGV:
            ciphers = [self.E[i].encryptBGV(np.asarray([v.moduli[i]], dtype=np.int64)) for i in range(self.length)]
        elif self.scheme == BFV:
            ciphers = [self.E[i].encryptInt(np.asarray([v.moduli[i]], dtype=np.int64)) for i in range(self.length)]
        else:
            print("encrypt: unknown scheme. expected {} or {}, got {}".format(BFV, BGV, self.scheme))
        return Cipher(ciphers=ciphers, name=v.name)

    def encrypt_2D(self, Vector: List[Decipher]) -> np.ndarray[Cipher]:
        result = [self.encrypt_1D(Vector[i]) for i in range(len(Vector))]
        return np.asarray(result, dtype=object)

    def validate_1D(self, cipher: Cipher) -> np.int64:
        decipher = self.decrypt_1D(cipher=cipher)
        return decipher.to_num()

    def validate_2D(self, ciphers: List[Cipher]) -> np.ndarray[np.int64]:
        nums = [crt_to_num(moduli=self.decrypt_2D(ciphers=ciphers)[i], dividers=self.Primes) for i in
                range(len(ciphers))]
        return np.asarray(nums, dtype=np.int64)


def test():
    # init

    F = FHE(scheme=BGV, number_of_primes=4)
    T = F.get_primes()
    a = 15600
    b = -1246
    ab_square = a * b

    a_decipher = Decipher(number=a, primes=T, N=F.N, name='a')
    b_decipher = Decipher(number=b, primes=T, N=F.N, name='b')
    ab_decipher = a * b
    # a_decipher = Decipher(number=a, primes=T, N=N)

    # encrypt
    Pa = F.encrypt_1D(v=a_decipher)
    Pb = F.encrypt_1D(v=b_decipher)
    Pab = Pa * Pb
    # Pa = F.encrypt_1D(v=a_decipher)
    # Pb = F.encrypt_1D(Vector=b_crt)

    # decrypt
    Ra = F.decrypt_1D(cipher=Pa)
    Rb = F.decrypt_1D(cipher=Pb)
    Rab = F.decrypt_1D(cipher=Pab)
    Rab.name = 'ab'

    Ra_num, Rb_num = Ra.to_num(), Rb.to_num()
    Rab_num = Rab.to_num()

    # validate encryption
    print("a encryption: {}".format("SUCCESS" if Ra_num == a else "FAIL"))
    print("b encryption: {}".format("SUCCESS" if Rb_num == b else "FAIL"))
    print("b encryption: {}".format("SUCCESS" if Rb_num == b else "FAIL"))

    # check addition
    P_add = Pa + Pb
    R_add = F.decrypt_1D(cipher=P_add)
    result_add = R_add.to_num()

    # check multiplication
    P_ab = F.encrypt_1D(v=Decipher(number=a * b, primes=T, N=F.N, name='ab'))
    P_ab_squared = P_ab * P_ab
    P_ab_squared.name = 'a^2 b^2 test'
    R_square_test = F.decrypt_1D(cipher=P_ab_squared, name='a^2 b^2 test')

    P_mul = Pa * Pb
    P_mul_squared = P_mul * P_mul
    R_mul = F.decrypt_1D(cipher=P_mul)
    R_mul_squared = F.decrypt_1D(cipher=P_mul_squared, name='a_2*b_2')

    # check subtraction
    P_sub = Pb - Pa
    R_sub = F.decrypt_1D(cipher=P_sub)
    result_sub = R_sub.to_num()

    print("------------Addition:-------------")
    print("{}".format("SUCCESS" if result_add == a + b else "FAILED"))
    print("result: {}, Truth: {}".format(result_add, a + b))
    print("CRT fields (result): {}".format(R_add))

    print("------------Subtraction:-------------")
    print("{}".format("SUCCESS" if result_sub == -(a - b) else "FAILED"))
    print("result: {}, Truth: {}".format(result_sub, b - a))
    print("CRT fields (result): {}".format(R_sub))

    print("------------Multiplication (1):-------------")
    print("{}".format("SUCCESS" if R_mul.to_num() == a * b else "FAILED"))
    print("result: {}, Truth: {}".format(R_mul.to_num(), a * b))
    print("CRT fields (result): {}".format(R_mul))

    print("------------Multiplication (2):-------------")
    print("{}, bit size: {}".format("SUCCESS" if R_mul_squared.to_num() == (a * b) ** 2 else "FAILED",
                                    bit_size(R_mul_squared.to_num())))
    print("result: {}, Truth: {}".format(R_mul_squared.to_num(), (a * b) ** 2))
    print("CRT fields (result): {}".format(R_mul_squared))


if __name__ == '__main__':
    print("FHE4")
    test()
