from Pyfhel import PyCtxt, Pyfhel, PyPtxt
from typing import Dict, List
import numpy as np
from CRT.CRT import CRT_to_num
from time import time
# from sympy.ntheory.modular import crt
import math


class Decipher:
    def __init__(self, N: int, primes: np.ndarray[np.int64], number: int = 0, values: np.ndarray = None,
                 name: str = None):
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
        if num <= self.N / 2:
            return num
        else:
            return num - self.N

    def __add__(self, other, check: bool = True):
        moduli = np.add(self.moduli, other.moduli)
        moduli = np.mod(moduli, self.primes)
        return Decipher(N=self.N, primes=self.primes, values=moduli)

    def __sub__(self, other, check: bool = True):
        moduli = np.subtract(self.moduli, other.moduli)
        moduli = np.mod(moduli, self.primes)
        return Decipher(N=self.N, primes=self.primes, values=moduli)

    def __mul__(self, other, check: bool = True):
        moduli = np.multiply(self.moduli, other.moduli)
        moduli = np.mod(moduli, self.primes)
        return Decipher(N=self.N, primes=self.primes, values=moduli)


class Cipher:
    values = None

    def __init__(self, ciphers: List[PyCtxt], name: str = None):
        self.values = ciphers
        self.length = len(ciphers)
        self.name = name

    def __add__(self, other):
        return Cipher(ciphers=[self.values[i] + other.values[i] for i in range(self.length)])

    def __mul__(self, other):
        return Cipher(ciphers=[self.values[i] * other.values[i] for i in range(self.length)])

    def __sub__(self, other):
        return Cipher(ciphers=[self.values[i] - other.values[i] for i in range(self.length)])
