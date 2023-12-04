import numpy as np
import math
from sympy.ntheory.modular import crt

'''
A class with various crt related utilities

Coded by Meir Goldenberg  
meirgold@hotmail.com
'''
class CrtUtils:
    def run_crt(self, primes, res_list_of_lists):
        '''
        Calculates values based on their division remainder using the chinese remainder theorem
        @param primes: A list of primes by which the values were divided
        @param res_list_of_lists: list of value lists
        @return: the list of values calculated by the CRT
        '''
        values = []
        # as the values are received
        transposed = np.array(res_list_of_lists).T.tolist()

        for res_list in transposed:
            values.append(crt(primes, res_list)[0])

        return values

    def calc_final_ages_crt(self, primes, numerator_list, denom_list):
        '''
        Calculate the age values using CRT
        @param primes: list of primes used for the calculation
        @param numerator_list: list of t_num value lists  per prime
        @param denom_list: list of t_denom values per prime
        @return: the age values (t_num/t_denom) and the largest t_num value (useful for amount of primes estimation)
        '''
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
