import sys
sys.path.append('../')

from pathlib import Path
import pickle
import matplotlib.pyplot as plt
import numpy as np

from py_diff_pd.common.common import print_info

if __name__ == '__main__':
    for thread_ct in [2, 4, 8, 12, 16,]:
        data_file = Path('cantilever_3d') / 'data_{:04d}_threads.bin'.format(thread_ct)
        if data_file.exists():
            print_info('Loading {}'.format(data_file))
            data = pickle.load(open(data_file, 'rb'))
            for method in ['newton_pcg', 'newton_cholesky', 'pd']:
                total_time = 0
                for d in data[method]:
                    print('loss: {:8.3f}, |grad|: {:8.3f}, E: {:8.3e}, nu: {:4.3f}, forward time: {:6.3f}s, backward time: {:6.3f}s'.format(
                        d['loss'], np.linalg.norm(d['grad']), d['E'], d['nu'], d['forward_time'], d['backward_time']))
                    total_time += d['forward_time'] + d['backward_time']
                print_info('Optimizing with {} finished in {:6.3f} seconds'.format(method, total_time))
