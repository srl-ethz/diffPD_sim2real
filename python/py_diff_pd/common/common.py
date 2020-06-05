import numpy as np

def ndarray(val):
    return np.asarray(val, dtype=np.float64)

def print_error(*message):
    print('\033[91m', 'ERROR ', *message, '\033[0m')

def print_ok(*message):
    print('\033[92m', *message, '\033[0m')

def print_warning(*message):
    print('\033[93m', *message, '\033[0m')

def print_info(*message):
    print('\033[96m', *message, '\033[0m')

import shutil
import os
def create_folder(folder_name):
    if os.path.isdir(folder_name):
        shutil.rmtree(folder_name)
    os.makedirs(folder_name)

from py_diff_pd.core.py_diff_pd_core import StdRealVector
def to_std_real_vector(v):
    v = ndarray(v).ravel()
    n = v.size
    v_array = StdRealVector(n)
    for i in range(n):
        v_array[i] = v[i]
    return v_array