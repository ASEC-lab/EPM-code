from __future__ import annotations
from CRT.CrtSet import CrtSet


class CrtVector:
    '''
    Class for handling a list of CrtSets

    Coded by Meir Goldenberg
    meirgold@hotmail.com
    '''
    def __init__(self):
        self.__crt_vector = []

    def add_vector(self, crt_vector: CrtVector):
        '''
        append a vector to the existing vector
        @param crt_vector: the vector to append
        '''
        self.__crt_vector.extend(crt_vector.get_vector())

    def add(self, crt_set: CrtSet):
        '''
        Add a crt set to the crt vector
        @param crt_set: the crt set to add
        @return:
        '''
        self.__crt_vector.append(crt_set)

    def get(self, idx) -> CrtSet:
        '''
        return a crt set at a given index
        @param idx: the index to return from
        @return: the crt set
        '''
        return self.__crt_vector[idx]

    def get_vector(self):
        '''
        returns the crt vector
        @return: the crt vector
        '''
        return self.__crt_vector

    def copy_calc_results(self, crt_set: CrtSet):
        '''
        copy results from a crt_set to the crt vector
        useful when calculations are done on a specific set and later need to be added to the vector
        @param crt_set: the crt set to copy from
        '''
        idx_lst = [i for i, item in enumerate(self.__crt_vector) if item.prime_index == crt_set.prime_index]
        # index in vector should be unique
        assert len(idx_lst) > 0, "index {} not found in the crt_vector".format(crt_set.prime_index)
        assert len(idx_lst) == 1, "Non unique items {} found in crt_vector".format(crt_set.prime_index)
        idx = idx_lst[0]
        self.__crt_vector[idx].t_num = crt_set.t_num
        self.__crt_vector[idx].t_denom = crt_set.t_denom
        self.__crt_vector[idx].s0_vals = crt_set.s0_vals
        self.__crt_vector[idx].rates = crt_set.rates

    def __len__(self):
        '''

        @return: the length of the crt vector
        '''
        return len(self.__crt_vector)
