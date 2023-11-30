import numpy as np
import math
from sympy.ntheory.modular import crt


class CrtUtils:
    def run_crt(self, primes, res_list_of_lists):
        values = []
        transposed = np.array(res_list_of_lists).T.tolist()

        for res_list in transposed:
            values.append(crt(primes, res_list)[0])

        return values

    def calc_final_ages_crt(self, primes, numerator_list, denom_list):
        final_ages = []
        prime_mult = math.prod(primes)
        half_prime_mult = prime_mult // 2
        ages = self.run_crt(primes, numerator_list)
        sum_ri_squared = self.run_crt(primes, denom_list)
        actual_age_num_vals = []

        # as we are dealing with very large numbers, we may exceed the maximum float value for python
        # which is given in sys.float_info.max
        # in this case, we need to replace this with an integer division
        for age_num in ages:
            if age_num > half_prime_mult:
                age_num = -(prime_mult - age_num)
            actual_age_num_vals.append(age_num)
            try:
                age = age_num / sum_ri_squared[0]
            except OverflowError:
                age = age_num // sum_ri_squared[0]
            final_ages.append(age)

        return final_ages, max(actual_age_num_vals)