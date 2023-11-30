class CrtSet:
    def __init__(self, prime_index, enc_ages, enc_meth_val_list, enc_transposed_meth_val_list):
        self.prime_index = prime_index
        self.enc_ages = enc_ages
        self.enc_meth_val_list = enc_meth_val_list
        self.enc_transposed_meth_val_list = enc_transposed_meth_val_list
        self.t_num = None
        self.t_denom = None
        self.s0_vals = None
        self.rates = None

