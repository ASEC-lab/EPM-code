from FHE import FHE as FHE
from time import time
import numpy as np


def f(a, b):
    return a + b * (b * b)


def test_bgv(F: FHE, low: int, high: int, attempts: int = 100):
    numbers = np.random.randint(low=low, high=high, size=attempts, dtype=np.int64)

    # addition
    start = time()
    sum = np.sum(numbers)
    moduli = np.asarray([np.mod(x, F.Primes) for x in numbers], dtype=object)
    P = F.encrypt_2D(moduli)
    Psum = np.sum(P, axis=0)
    success = F.validate_1D(Psum) == sum
    outcome = "PASSED" if success else "FAILED"
    print("Addition Test with {} attempts {} in {:.2f} seconds.".format(attempts, outcome, time() - start))

    # multiplication
    start = time()
    indices1 = np.random.permutation(attempts)[:int(0.1 * attempts)]
    indices2 = np.random.permutation(attempts)[:int(0.1 * attempts)]
    A = P[indices1]
    B = P[indices2]
    prod = A * B
    truth = numbers[indices1] * numbers[indices2]
    success = True

    for i in range(len(prod)):
        if F.validate_1D(prod[i]) != truth[i]:
            success = False
            break
    outcome = "PASSED" if success else "FAILED"
    print("Multiplication Test with {} attempts {} in {:.2f} seconds.".format(int(0.1 * attempts), outcome,
                                                                              time() - start))

    # subtraction
    start = time()
    Psum_real = F.encrypt_1D(np.mod(sum, F.Primes))
    for c in P:
        Psum_real = Psum_real - c

    success = F.validate_1D(Psum_real) == 0
    outcome = "PASSED" if success else "FAILED"
    print("Subtraction Test with {} attempts {} in {:.2f} seconds.".format(attempts, outcome, time() - start))


if __name__ == '__main__':
    print("Encryption Testing")
    F = FHE()
    test_bgv(low=100, high=1000, F=F)
