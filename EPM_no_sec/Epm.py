import numpy as np
# An implementation of the Epigenetic pacemaker algorithm
# used mainly for benchmark and validation
from CRT.CRT import *

TESTING = True
SIZE = 10


class EPM:
    age_vales = None
    meth_vals = None
    use_crt = None

    def __init__(self, meth_vals: np.ndarray, age_vals: np.ndarray, primes: np.ndarray = None, use_crt: bool = False):
        self.age_vals = age_vals
        self.meth_vals = meth_vals
        self.use_crt = use_crt

        self.age_vals_CRT = []
        self.meth_vals_CRT = []
        self.primes = primes

        if use_crt:
            self.age_vals_CRT = np.asarray([Crt(num=x, dividers=primes) for x in age_vals]).astype(object)
            for line in meth_vals:
                self.meth_vals_CRT.append([Crt(num=x, dividers=primes) for x in line])
            self.meth_vals_CRT = np.asarray(self.meth_vals_CRT).astype(object)

    def validate_CRT(self):
        # validate age_vals
        for i in range(len(self.age_vals)):
            if self.age_vals[i] != self.age_vals_CRT[i].Crt_to_num():
                print("age vals {} (value {}) crt invalid!".format(i, self.age_vals[i]))

        # validate meth vals
        for i in range(len(self.meth_vals)):
            for j in range(len(self.meth_vals[0])):
                if self.meth_vals[i][j] != self.meth_vals_CRT[i][j].Crt_to_num():
                    print("meth vals {},{} (value {}) crt invalid!".format(i, j, self.meth_vals[i][j]))

    def calc_X(self, ages, meth_vals):

        """
        Calculate the X matrix as described in the EPM algorithm
        @param ages: age values
        @param meth_vals: methylation values
        @return: The X matrix
        """
        # calculate the size of the X matrix
        # the size should be mn X 2n
        m = ages.shape[0]  # the number of individuals
        n = meth_vals.shape[0]  # the number of sites
        x_rows_num = m * n
        x_cols_num = 2 * n
        X = np.zeros((x_rows_num, x_cols_num))

        t = 0
        col_index = 0
        for i in range(x_rows_num):

            X[i][col_index] = ages[t]
            X[i][col_index + n] = 1
            t = t + 1
            if ((i + 1) % m) == 0:
                col_index = (col_index + 1)
                t = 0

        return X

    # below are the 3 methods discussed in the articles for calculating the value of beta
    # beta = (XtX)^-1*XtY
    # in the end only one needs to be used and this will be the result of the site step

    def calc_beta(self):
        """
        The standard method for calculating beta = (XtX)^-1*XtY
        @return: rates and s0 values
        """
        n = self.meth_vals.shape[0]  # the number of sites
        X = self.calc_X(self.age_vals, self.meth_vals)
        Xt = X.transpose()
        XtX = np.dot(Xt, X)
        invert_XtX = np.linalg.inv(XtX)
        self.A = np.dot(invert_XtX, XtX)  # this should be replaced by an I matrix
        invert_XtX_Xt = np.dot(invert_XtX, Xt)
        Y = self.meth_vals.flatten().transpose()
        result = np.dot(invert_XtX_Xt, Y)

        rates = result[0:n]
        s0 = result[n:]
        return rates, s0

    def calc_beta_lemma(self):
        """
        calc beta using lemma2 and lemma3
        from the article https://bmcgenomics.biomedcentral.com/articles/10.1186/s12864-020-6606-0
        @return: rates and s0 values
        """
        m = self.age_vals.shape[0]  # the number of individuals
        n = self.meth_vals.shape[0]  # the number of sites
        x_rows_num = 2 * n
        x_cols_num = m * n
        invert_XtX_Xt = np.zeros((x_rows_num, x_cols_num))
        sigma_t = np.sum(self.age_vals)
        sigma_t_square = np.sum(self.age_vals ** 2)
        gamma = 1 / (sigma_t ** 2 - m * sigma_t_square)

        # upper expanded diagonal matrix
        for k in range(n):
            l = k * m
            while l < ((k + 1) * m):
                t_index = l - (k * m)
                invert_XtX_Xt[k][l] = (-1 * m * self.age_vals[t_index] + sigma_t) * gamma
                l += 1

        # lower expanded diagonal matrix
        for k in range(n):
            l = k * m
            while l < ((k + 1) * m):
                t_index = l - (k * m)
                invert_XtX_Xt[k + n][l] = (self.age_vals[t_index] * sigma_t - sigma_t_square) * gamma
                l += 1

        Y = self.meth_vals.flatten().transpose()
        result = np.dot(invert_XtX_Xt, Y)

        rates = result[0:n]
        s0 = result[n:]
        return rates, s0

    def calc_beta_corollary1(self):
        """
        calc beta using corollary1
        from the article https://bmcgenomics.biomedcentral.com/articles/10.1186/s12864-020-6606-0
        @return: rates and s0 values
        """

        mult_val_list = []
        m = self.age_vals.shape[0]  # the number of individuals
        n = self.meth_vals.shape[0]  # the number of sites
        result = np.zeros(2 * n)  # here will be the results of the final calculation
        sigma_t = np.sum(self.age_vals)
        sigma_t_square = np.sum(self.age_vals ** 2)
        gamma = 1 / (sigma_t ** 2 - m * sigma_t_square)

        # prepare the multiply values
        for i in range(m):
            mult_val_list.append((-1 * m * self.age_vals[i] + sigma_t) * gamma)

        for i in range(m):
            mult_val_list.append((self.age_vals[i] * sigma_t - sigma_t_square) * gamma)

        list_len_div2 = int(len(mult_val_list) / 2)

        Y = self.meth_vals.flatten().transpose().astype(float)

        # as described in the corollary, we should be able to calculate (XtX)^-1XtY
        # without using heavy linear algebra calculations
        # due to the structure of the matrix (XtX)^-1Xt which is made up of 2 expanded-diagonal matrices
        # we can calculate the multiplication values for each of these matrices
        # and then multiply the respective values in Y.
        # for the upper part of the result: each i*m value in Y will be multiplied by the i'th value
        # for the lower part of the result: each i*m value in Y will be multiplied by the i+list_len_div2 value

        i = 0
        temp_result_upper = Y.copy()
        temp_result_lower = Y.copy()
        while (i < list_len_div2):
            temp_result_upper[i::list_len_div2] *= mult_val_list[i]
            temp_result_lower[i::list_len_div2] *= mult_val_list[i + list_len_div2]
            i += 1

        # for the final step, each m values from the upper and lower matrices will be summed
        # (in order to "emulate" matrix multiplication) and put into its respective location in the result
        for j in range(n):
            result[j] = np.sum(temp_result_upper[j * m:((j + 1) * m)])
            result[n + j] = np.sum(temp_result_lower[j * m:((j + 1) * m)])

        rates = result[0:n]
        s0 = result[n:]
        return rates, s0

    def both_steps_no_division(self, sigma_ri_square, Testing: bool = False):
        """ Remove once we know CRT is ok"""
        if Testing:
            self.age_vals = self.age_vals[:SIZE]
            self.meth_vals = self.meth_vals[:SIZE]
        """"""

        mult_val_list = []
        m = self.age_vals.shape[0]  # the number of individuals
        n = self.meth_vals.shape[0]  # the number of sites
        result = np.zeros(2 * n)  # here will be the results of the final calculation
        sigma_t = np.sum(self.age_vals)
        sigma_t_square = np.sum(self.age_vals ** 2)
        gamma = (sigma_t ** 2 - m * sigma_t_square)

        # prepare the multiply values
        for i in range(m):
            mult_val_list.append((-1 * m * self.age_vals[i] + sigma_t) * sigma_ri_square)

        for i in range(m):
            mult_val_list.append(self.age_vals[i] * sigma_t - sigma_t_square)

        list_len_div2 = int(len(mult_val_list) / 2)

        Y = self.meth_vals.flatten().transpose().astype(float)

        # as described in the corollary, we should be able to calculate (XtX)^-1XtY
        # without using heavy linear algebra calculations
        # due to the structure of the matrix (XtX)^-1Xt which is made up of 2 expanded-diagonal matrices
        # we can calculate the multiplication values for each of these matrices
        # and then multiply the respective values in Y.
        # for the upper part of the result: each i*m value in Y will be multiplied by the i'th value
        # for the lower part of the result: each i*m value in Y will be multiplied by the i+list_len_div2 value

        i = 0
        temp_result_upper = Y.copy()
        temp_result_lower = Y.copy()
        while (i < list_len_div2):
            temp_result_upper[i::list_len_div2] *= mult_val_list[i]
            temp_result_lower[i::list_len_div2] *= mult_val_list[i + list_len_div2]
            i += 1

        # for the final step, each m values from the upper and lower matrices will be summed
        # (in order to "emulate" matrix multiplication) and put into its respective location in the result
        for j in range(n):
            result[j] = np.sum(temp_result_upper[j * m:((j + 1) * m)])
            result[n + j] = np.sum(temp_result_lower[j * m:((j + 1) * m)])

        rates = result[0:n]
        s0_vals = result[n:]

        # calc the matrix  S = (S_ij - s0_i)
        meth_vals_gamma = self.meth_vals * gamma
        S = np.transpose(np.subtract(np.transpose(meth_vals_gamma), s0_vals))
        # calc Si * ri
        F = S * rates[:, np.newaxis]
        # calc sum(r_i^2)
        r_squared_sum = np.sum(rates ** 2)
        t = np.sum(F, axis=0)
        return t, r_squared_sum

    def both_steps_no_division_with_crt(self, sigma_ri_square: Crt, Testing: bool = False) -> (np.ndarray[Crt],Crt):
        """ Remove once we know CRT is ok"""
        if Testing:
            self.age_vals = self.age_vals[:SIZE]
            self.age_vals_CRT = self.age_vals_CRT[:SIZE]
            self.meth_vals = self.meth_vals[:SIZE]
            self.meth_vals_CRT = self.meth_vals_CRT[:SIZE]
        """"""

        # preliminaries
        mult_val_list = []
        m = self.age_vals.shape[0]  # the number of individuals
        n = self.meth_vals.shape[0]  # the number of sites
        m_crt = Crt(num=m, dividers=self.primes)

        minus_one = Crt(num=-1, dividers=self.primes)
        result = np.full(shape=(2 * n), fill_value=Crt(num=0, dividers=self.primes))
        sigma_t = np.sum(self.age_vals_CRT)
        sigma_t_square = np.sum(np.multiply(self.age_vals_CRT, self.age_vals_CRT))

        """check sigma_t"""
        # if sigma_t.Crt_to_num() != np.sum(self.age_vals):
        #     print("sigma_t bad")
        #     for i in range(len(self.age_vals)):
        #         if self.age_vals[i]!=self.age_vals_CRT[i].Crt_to_num():
        #             print(i, self.age_vals[i], self.age_vals_CRT[i].Crt_to_num())

        """check sigma_t_square"""
        # if sigma_t_square.Crt_to_num() != np.sum(self.age_vals**2):
        #     print("sigma_t_square bad")
        #     print(sigma_t_square.Crt_to_num(), np.sum(self.age_vals**2))

        gamma = (sigma_t * sigma_t) - (m_crt * sigma_t_square)

        """check gamma:"""
        # gamma_other = (np.sum(self.age_vals) ** 2 - (m * np.sum(self.age_vals ** 2)))
        # if gamma!=gamma_other:
        #     print("gamma bad")
        #     print(gamma.Crt_to_num(), gamma_other)

        # prepare the multiply values
        for i in range(m):
            mult_val_list.append((minus_one * m_crt * self.age_vals_CRT[i] + sigma_t) * sigma_ri_square)

        for i in range(m):
            mult_val_list.append(self.age_vals_CRT[i] * sigma_t - sigma_t_square)

        list_len_div2 = int(len(mult_val_list) / 2)

        Y = self.meth_vals_CRT.flatten().transpose()

        temp_result_upper = Y.copy()
        temp_result_lower = Y.copy()

        for i in range(list_len_div2):
            temp_result_upper[i::list_len_div2] *= mult_val_list[i]
            temp_result_lower[i::list_len_div2] *= mult_val_list[i + list_len_div2]

        # for the final step, each m values from the upper and lower matrices will be summed
        # (in order to "emulate" matrix multiplication) and put into its respective location in the result
        for j in range(n):
            result[j] = np.sum(temp_result_upper[j * m:((j + 1) * m)])
            result[n + j] = np.sum(temp_result_lower[j * m:((j + 1) * m)])

        rates = result[0:n]
        s0_vals = result[n:]

        # calc the matrix  S = (S_ij - s0_i)
        meth_vals_gamma = self.meth_vals_CRT * gamma
        S = np.transpose(np.subtract(np.transpose(meth_vals_gamma), s0_vals))
        # calc Si * ri
        F = S * rates[:, np.newaxis]
        # calc sum(r_i^2)
        r_squared_sum = np.sum(rates ** 2)
        t = np.sum(F, axis=0)
        return t, r_squared_sum


    def calc_rss(self, rates, s0_vals, ages):

        total_error = 0.0
        col_num = 0
        for meth_col in self.meth_vals.T:
            t_j = ages[col_num]
            r_t_j = rates * t_j
            s0_plus_r_t_j = s0_vals + r_t_j
            m_rss = ((meth_col - s0_plus_r_t_j) ** 2)
            total_error += np.sum(m_rss)
            col_num += 1

        return total_error

    def site_step(self):
        rates, s0 = self.calc_beta_corollary1()
        return rates, s0

    def time_step(self, rates: np.ndarray, s0_vals: np.ndarray):
        # calc the matrix  S = (S_ij - s0_i)
        S = np.transpose(np.subtract(np.transpose(self.meth_vals), s0_vals))
        # calc Si * ri
        F = S * rates[:, np.newaxis]
        # calc sum(r_i^2)
        r_squared_sum = np.sum(rates ** 2)
        t = np.sum(F, axis=0) / r_squared_sum
        return t

    def calc_model(self, iter_limit: int = 100, error_tolerance: float = .00001):

        prev_rss = 0
        iter = 0
        rss_diff = prev_rss - error_tolerance
        rates, s0 = self.site_step()

        while iter < iter_limit:
            rates, s0 = self.site_step()
            self.age_vals = self.time_step(rates, s0)
            rss = self.calc_rss(rates, s0, self.age_vals)
            # print("iter: {} rss: {} prev_rss: {}".format(iter, rss, prev_rss))
            if prev_rss > 0:  # don't check this on the first iteration
                assert rss < prev_rss, "New RSS {} is larger than previous {}".format(rss, prev_rss)
                rss_diff = prev_rss - rss
                if rss_diff < error_tolerance:
                    break
            prev_rss = rss

            iter += 1

        model_params = {
            'rss_err': rss_diff,
            'num_of_iterations': iter,
            's0': s0,
            'rates': rates,
            'ages': self.age_vals
        }

        return model_params

    def calc_model_new_method(self, max_iterations: int, Testing: bool = False) -> np.ndarray:
        i = 0
        sigma_ri_squared = 1
        ages = []

        while i < max_iterations:
            ages, sigma_ri_squared = self.both_steps_no_division(sigma_ri_squared, Testing=Testing)
            self.age_vals = ages
            i += 1

        ages = self.age_vals / sigma_ri_squared
        return ages


    def calc_model_new_method_crt(self, max_iterations: int, Testing: bool = False) -> (np.ndarray[Crt], Crt):
        i = 0
        sigma_ri_squared = Crt(num=1, dividers=self.primes)
        ages = []

        while i < max_iterations:
            ages, sigma_ri_squared = self.both_steps_no_division_with_crt(sigma_ri_squared, Testing=Testing)
            self.age_vals_CRT = ages
            i += 1

        ages = CRT_vec_to_num(self.age_vals_CRT) / sigma_ri_squared.Crt_to_num()
        return ages
