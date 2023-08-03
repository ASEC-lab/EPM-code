import numpy as np
from CRT.CRT import CRT_to_num
import math


def product(array):
    return math.prod([int(x) for x in array])




class Decipher:
    def __init__(self, N: int, primes: np.ndarray[np.int64], number: int = 0, values: np.ndarray = None,
                 name: str = ""):
        self.N = N
        self.primes = primes
        self.name = name
        if values is None:
            if number >= 0:
                self.moduli = np.mod(number, primes)
            else:
                new_number = N + number
                self.moduli = np.mod(new_number, primes)
        else:
            self.moduli = values

        for i in range(len(self.moduli)):
            while self.moduli[i] < 0:
                self.moduli[i] += self.primes[i]

    def to_num(self) -> np.int64:
        num = CRT_to_num(values=self.moduli, div=self.primes)
        # if self.name == 'a^2 b^2':
        #     print(self.name+"  "+str(num))
        if num <= self.N / 2:
            return num
        else:
            return num - self.N

    def __add__(self, other, check: bool = True):
        moduli = np.add(self.moduli, other.moduli)
        moduli = np.mod(moduli, self.primes)
        return Decipher(N=self.N, primes=self.primes, values=moduli, name=self.name + '+' + other.name)

    def __sub__(self, other, check: bool = True):
        moduli = np.subtract(self.moduli, other.moduli)
        moduli = np.mod(moduli, self.primes)
        return Decipher(N=self.N, primes=self.primes, values=moduli, name=self.name + '-' + other.name)

    def __mul__(self, other, check: bool = True):
        moduli = np.multiply(self.moduli, other.moduli)
        moduli = np.mod(moduli, self.primes)
        return Decipher(N=self.N, primes=self.primes, values=moduli, name=self.name + '*' + other.name)






def test():
    pass
    P = np.load('primes.npy')
    T = P[:4, 0]
    N = product(T)
    a = 15600
    b = -1246
    ab_square = a * b

    Da = Decipher(number=a, primes=T, N=N, name='a')
    Db = Decipher(number=b, primes=T, N=N, name='b')

    Dab = Da * Db
    Dab_2 = Dab*Dab

    print("Da: {}, real a: {}".format(Da.to_num(), a))
    print("Db: {}, real b: {}".format(Db.to_num(), b))
    print("Dab: {}, real prod: {}".format(Dab.to_num(), a*b))
    print("Dab_2: {}, real prod^2: {}".format(Dab_2.to_num(), a*a*b*b))









if __name__ == '__main__':
    print("Decipher")
    test()