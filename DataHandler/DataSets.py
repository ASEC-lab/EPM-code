import gzip
import io
import os
import numpy as np


# utility functions for reading the input data

class DataSets:
    example_dir = os.path.dirname(os.path.realpath(__file__)) +'/'

    def convert_to_float(self, line):
        """
        converts a line of numeric values in text format to float
        @param line: the line to convert
        @return: the float values
        """
        converted_line = []
        for value in line:
            try:
                converted_line.append(float(value))
            except ValueError:
                converted_line.append(value)
        return converted_line

    def load_data_set(self, file_path):
        """
        Loads the data from .gz files
        @param file_path:
        @return: individual names, site names, ages and methylation values
        """
        formatted_data = []
        with io.BufferedReader(gzip.open(file_path, 'rb')) as data:
            for line in data:
                formatted_data.append(self.convert_to_float(line.decode('utf-8').strip().split('\t')))
        sample_names = formatted_data[0]
        cpg_sites = [line[0] for line in formatted_data[1:-1]]
        ages = np.array(formatted_data[-1][1:], dtype=np.longdouble)
        meth_vals = np.array([line[1:] for line in formatted_data[1:-1]])
        return sample_names, cpg_sites, ages, meth_vals

    def reduce_data_size(self, data: np.ndarray, rows: int, cols: int):
        """
        Return a given number of rows and columns from the input data
        used for running on a smaller subset of the data mainly for testing purposes
        @param data: individual names, site names, ages and methylation values
        @param rows: the number of rows to use
        @param cols: the number of columns to use
        @return: the reduced data sets
        """
        sample_names, cpg_sites, ages, meth_vals = data
        reduced_sample_names = sample_names[0:cols]
        reduced_cpg_sites = cpg_sites[0:rows]
        reduced_ages = ages[0:cols]
        reduced_meth_vals = meth_vals[0:rows, 0:cols]

        return reduced_sample_names, reduced_cpg_sites, reduced_ages, reduced_meth_vals

    def get_example_train_data(self):
        """
        Load the training data
        @return: training data set
        """
        #train_data = self.load_data_set(f'{self.example_dir}GSE74193_train.tsv.gz')
        train_data = self.load_data_set(f'{self.example_dir}custom_train_data.tsv.gz')

        return train_data

    def get_example_test_data(self):
        """
        Read the test data
        @return: The test data set
        """
        test_data = self.load_data_set(f'{self.example_dir}GSE74193_test.tsv.gz')
        return test_data
