from __future__ import annotations
from CRT.CrtSet import CrtSet


class CrtVector:

    def __init__(self):
        self.__crt_vector = []

    def add_vector(self, crt_vector: CrtVector):
        self.__crt_vector.extend(crt_vector.get_vector())

    def add(self, crt_set: CrtSet):
        self.__crt_vector.append(crt_set)

    def get(self, idx) -> CrtSet:
        return self.__crt_vector[idx]

    def get_vector(self) :
        return self.__crt_vector

    def copy_calc_results(self, crt_set: CrtSet):
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
        return len(self.__crt_vector)



