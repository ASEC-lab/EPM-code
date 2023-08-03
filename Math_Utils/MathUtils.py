import math


def is_prime(n):
    for i in range(2, int(math.sqrt(n))+1):
        if (n%i) == 0:
            return False

    return True


def read_primes_from_file(file_path):
    primes = []
    with open(file_path, 'r') as fp:
        for line in fp:
            x = line[:-1]
            primes.append(int(x))

    return primes


def find_primes(n, divisible_by, num_of_primes):
    prime_list = []
    i = 1
    while True:
        num = i * n + 1

        if (((num-1) % divisible_by) == 0) and is_prime(num):
            prime_list.append(num)
            if len(prime_list) == num_of_primes:
                return prime_list

        i += 1