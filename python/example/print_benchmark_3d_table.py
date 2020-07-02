import sys
sys.path.append('../')

import pickle
from pathlib import Path
import numpy as np

from py_diff_pd.common.common import PrettyTabular
from py_diff_pd.common.common import print_info

if __name__ == '__main__':
    folder = Path('benchmark_3d')
    rel_tols, forward_times, backward_times, losses, grads = pickle.load(open(folder / 'table.bin', 'rb'))

    for idx, rel_tol in enumerate(rel_tols):
        print_info('rel_tol: {:3.3e}'.format(rel_tol))
        tabular = PrettyTabular({
            'method': '{:^30s}',
            'forward and backward (s)': '{:3.3f}',
            'forward only (s)': '{:3.3f}',
            'loss': '{:3.3f}',
            '|grad|': '{:3.3f}'
        })
        print_info(tabular.head_string())
        for method in forward_times:
            print(tabular.row_string({
                'method': method,
                'forward and backward (s)': forward_times[method][idx] + backward_times[method][idx],
                'forward only (s)': forward_times[method][idx],
                'loss': losses[method][idx],
                '|grad|': np.linalg.norm(grads[method][idx]) }))
