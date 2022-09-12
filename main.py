import numpy as np
from CSP.Csp import CSP
import datetime
from EPM_no_sec.Epm import EPM
from DataHandler.DataSets import DataSets
from DataHandler.DataFormat import pearson_correlation
from DO.Do import DO

def epm_orig():
    """
    EPM implementation without encryption. Used for benchmarking.
    @return: calculated model params
    """
    max_iterations = 100
    rss = 0.0000001
    data_sets = DataSets()
    # read training data
    full_train_data = data_sets.get_example_train_data()
    train_samples, train_cpg_sites, train_ages, train_methylation_values = full_train_data
    # run pearson correlation in order to reduce the amount of processed data
    abs_pcc_coefficients = abs(pearson_correlation(train_methylation_values, train_ages))
    correlated_meth_val_indices = np.where(abs_pcc_coefficients > .91)[0]
    correlated_meth_val = train_methylation_values[correlated_meth_val_indices, :]
    # run the algorithm
    epm = EPM(correlated_meth_val, train_ages)
    model = epm.calc_model(max_iterations, rss)
    return model
'''
def main_old():
    print("Calculating encrypted model")
    start_time = datetime.datetime.now()
    enc_epm_model = encrypted_epm()
    enc_end_time = datetime.datetime.now()
    print("Calculating non-encrypted model")
    orig_model = epm_orig()
    orig_model_end_time = datetime.datetime.now()

    with open("output.txt", 'w') as f:
        f.write("Encrypted model:\n")
        for key, value in enc_epm_model.items():
            f.write('%s:%s\n' % (key, value))

        f.write("Original model:\n")
        for key, value in orig_model.items():
            f.write('%s:%s\n' % (key, value))

        f.write("enc model calc start: {}\n".format(start_time))
        f.write("enc model calc end: {}\n".format(enc_end_time))
        f.write("non-enc model calc end: {}\n".format(orig_model_end_time))
    print(enc_epm_model)
    print(orig_model)
'''

def main():
    csp = CSP()
    do = DO(csp)
    do.calc_model()

if __name__ == '__main__':
    main()

