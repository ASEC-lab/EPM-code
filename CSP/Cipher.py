from Pyfhel import PyCtxt, Pyfhel, PyPtxt
from typing import Dict, List
import numpy as np
from CRT.CRT import CRT_to_num
from time import time
import math




class Cipher:
    values = None

    def __init__(self, ciphers: List[PyCtxt], name: str = None):
        self.values = ciphers
        self.length = len(ciphers)
        self.name = name

    def __add__(self, other):
        return Cipher(ciphers=[self.values[i] + other.values[i] for i in range(self.length)],
                      name=self.name + '+' + other.name)

    def __mul__(self, other):
        return Cipher(ciphers=[~self.values[i] * ~other.values[i] for i in range(self.length)],
                      name=self.name + '*' + other.name)

    def __sub__(self, other):
        return Cipher(ciphers=[self.values[i] - other.values[i] for i in range(self.length)],
                      name=self.name + '-' + other.name)
