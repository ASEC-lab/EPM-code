from Pyfhel import PyCtxt, Pyfhel, PyPtxt
from typing import Dict, List
import numpy as np
from sympy.ntheory.modular import crt


T = np.asarray([3, 5, 17, 257, 65537], dtype=np.int64)

Default_Params = {
    'scheme': 'BFV',
    'n': 2 ** 14,
    't_bits': 20,
    'sec': 128,
}


def crt_to_num(moduli, dividers=T):
    return crt(dividers, moduli)[0]




class FHE:
    params = None
    label = None
    length = None

    def __init__(self, params: Dict[str, int] = None, label: int = 0):
        if params is None: params = Default_Params
        self.scheme = params['scheme']
        self.n = params['n']
        self.t = params['t']
        self.t_bits = params['t_bits']
        self.sec = params['sec']
        self.label = label

        self.HE = Pyfhel()
        self.HE.contextGen(**params)  # Generate context for bfv scheme
        self.HE.keyGen()  # Key Generation: generates a pair of public/secret keys
        self.HE.rotateKeyGen()  # Rotate key generation --> Allows rotation/shifting
        self.HE.relinKeyGen()  # Relinearization key generation

        self.length = None

    def encrypt_1D(self, Vector: np.ndarray[np.int64]) -> PyCtxt:
        return self.HE.encryptInt(Vector)

    def decrypt_1D(self, Cipher: PyCtxt, length: int = None) -> np.ndarray[np.int64]:
        vector = self.HE.decryptInt(Cipher)
        return vector[0]



class FHE2:
    params = None
    Primes = None
    length = None
    E = None

    def __init__(self, params: Dict[str, int] = None, T_: np.ndarray[np.int64] = T, label: int = 0):
        if params is None: params = Default_Params
        self.params = params
        self.Primes, self.length = T_, len(T_)
        self.E = []

        for prime in T_:
            params['t'] = prime
            F = FHE(params=params, label=prime)
            self.E.append(F)

    def encrypt_1D(self, Vector: np.ndarray[np.int64]) -> PyCtxt:
        return [self.E[i].encrypt_1D(Vector[i:i + 1]) for i in range(self.length)]

    def decrypt_1D(self, cipher: List[PyCtxt]) -> np.ndarray[np.int64]:
        return np.asarray([self.E[i].decrypt_1D(cipher[i]) for i in range(self.length)])

    def add_1D(self, cipher_1: List[PyCtxt], cipher_2: List[PyCtxt]) -> List[PyCtxt]:
        return [cipher_1[i] + cipher_2[i] for i in range(len(cipher_1))]

    def mul_1D(self, cipher_1: List[PyCtxt], cipher_2: List[PyCtxt]) -> List[PyCtxt]:
        # return [cipher_1[i] * cipher_2[i] for i in range(len(cipher_1))]
        return [self.E[i].HE.multiply(cipher_1[i], cipher_2[i]) for i in range(len(cipher_1))]



def f(a, b):
    return (a + b)*(b*b)


def main():


    a = 2103
    b = 824
    c = f(a, b)

    a_crt = np.mod(a, T)
    b_crt = np.mod(b, T)
    c_crt = np.mod((a_crt+b_crt), T)

    # init
    F = FHE2()
    # enc
    Pa = F.encrypt_1D(Vector=a_crt)
    Pb = F.encrypt_1D(Vector=b_crt)
    # dec
    Ra = F.decrypt_1D(cipher=Pa)
    Rb = F.decrypt_1D(cipher=Pb)
    # form C
    b_times_b = F.mul_1D(Pb, Pb)
    a_plus_b = F.add_1D(Pa, Pb)

    R_b2 = F.decrypt_1D(b_times_b)
    R_apb = F.decrypt_1D(a_plus_b)

    Pc = F.mul_1D(cipher_1=a_plus_b, cipher_2=b_times_b)
    C = F.decrypt_1D(Pc)

    print("c = (a+b)b^2")

    print("------------Before encryption:-------------")
    print("a real value: {}, CRT fields: {}".format(a, a_crt))
    print("b real value: {}, CRT fields: {}".format(b, b_crt))
    print("b^2: {}".format(b**2))
    print("a + b: {}".format(a+b))
    print("c real value: {}, CRT fields: {}".format(c, c_crt))


    print("------------After encryption:-------------")
    print("a real value: {}, CRT fields: {}".format(crt_to_num(Ra), Ra))
    print("b real value: {}, CRT fields: {}".format(crt_to_num(Rb), Rb))
    print("b^2: {}, CRT fields: {}".format(crt_to_num(R_b2), R_b2))
    print("a + b: {}, CRT fields: {}".format(crt_to_num(R_apb), R_apb))
    print("c real value: {}, CRT fields: {}".format(crt_to_num(C), C))


if __name__ == '__main__':
    main()
