import sys
sys.path.append('../')

import pickle
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from py_diff_pd.common.common import PrettyTabular
from py_diff_pd.common.common import print_info

def transpose_list(l, row_num, col_num):
    assert len(l) == row_num * col_num
    l2 = []
    for j in range(col_num):
        for i in range(row_num):
            l2.append(l[j + i * col_num])
    return l2

if __name__ == '__main__':
    plt.rc('pdf', fonttype=42)
    plt.rc('font', size=16)             # Controls default text sizes.
    plt.rc('axes', titlesize=16)        # Fontsize of the axes title.
    plt.rc('axes', labelsize=16)        # Fontsize of the x and y labels.
    plt.rc('xtick', labelsize=16)       # Fontsize of the tick labels.
    plt.rc('ytick', labelsize=16)       # Fontsize of the tick labels.
    plt.rc('legend', fontsize=16)       # Legend fontsize.
    plt.rc('figure', titlesize=16)      # Fontsize of the figure title.

    folder = Path('benchmark_3d')
    rel_tols, forward_times, backward_times, losses, grads = pickle.load(open(folder / 'table.bin', 'rb'))

    # Parse thread cnt.
    thread_cts = []
    for method in forward_times:
        if 'pd' in method:
            # Extract thread number from pd_XXthreads.
            thread_cts.append(int(method[3:-7]))
    thread_cts = sorted(thread_cts)

    forward_backward_times = {}
    for method in forward_times:
        forward_backward_times[method] = np.zeros(len(rel_tols))

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
            forward_backward_times[method][idx] = forward_times[method][idx] + backward_times[method][idx]
            print(tabular.row_string({
                'method': method,
                'forward and backward (s)': forward_times[method][idx] + backward_times[method][idx],
                'forward only (s)': forward_times[method][idx],
                'loss': losses[method][idx],
                '|grad|': np.linalg.norm(grads[method][idx]) }))

    fig = plt.figure(figsize=(18, 7))
    ax_fb = fig.add_subplot(131)
    ax_f = fig.add_subplot(132)
    ax_b = fig.add_subplot(133)
    titles = ['forward + backward', 'forward', 'backward']
    ax_poses = [(0.07, 0.29, 0.25, 0.6),
        (0.39, 0.29, 0.25, 0.6),
        (0.71, 0.29, 0.25, 0.6)]
    dash_list =[(5, 0), (5, 2), (2, 5), (4, 10), (3, 3, 2, 2), (5, 2, 20, 2), (5, 5), (5, 2, 1, 2)]
    for ax_pos, title, ax, t in zip(ax_poses, titles, (ax_fb, ax_f, ax_b), (forward_backward_times, forward_times, backward_times)):
        ax.set_position(ax_pos)
        ax.set_xlabel('time (s)')
        ax.set_ylabel('relative error')
        ax.set_yscale('log')
        for method in ['newton_pcg', 'newton_cholesky', 'pd']:
            if 'pd' in method:
                color = 'tab:green'
            elif 'pcg' in method:
                color = 'tab:blue'
            elif 'cholesky' in method:
                color = 'tab:red'
            for thread_ct in thread_cts:
                idx = thread_cts.index(thread_ct)
                meth_thread_num = '{}_{}threads'.format(method, thread_ct)
                ax.plot(t[meth_thread_num], rel_tols, label=meth_thread_num,
                    color=color, dashes=dash_list[idx], linewidth=2)
        ax.grid(True)
        ax.set_title(title)
        handles, labels = ax.get_legend_handles_labels()

    # Share legends.
    row_num = 3
    col_num = len(thread_cts)
    fig.legend(transpose_list(handles, row_num, col_num), transpose_list(labels, row_num, col_num),
        loc='upper center', ncol=col_num, bbox_to_anchor=(0.5, 0.19))
    fig.savefig(folder / 'benchmark.pdf')
    fig.savefig(folder / 'benchmark.png')
    plt.show()