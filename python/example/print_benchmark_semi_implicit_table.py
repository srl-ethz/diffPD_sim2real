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

    folder = Path('benchmark_semi_implicit')
    dts, forward_times, single_frame_times = pickle.load(open(folder / 'forward_table.bin', 'rb'))

    for idx, dt in enumerate(dts):
        print_info('dt: {:3.3e}'.format(dt))
        tabular = PrettyTabular({
            'method': '{:^30s}',
            'forward only (s)': '{:3.3f}',
            'single frame average (s)': '{:3.3f}',
        })
        print_info(tabular.head_string())
        for method in forward_times:
            print(tabular.row_string({
                'method': method,
                'forward only (s)': forward_times[method][idx],
                'single frame average (s)': single_frame_times[method][idx]
                }))

    epsilons, loss_sensitivity, grad_sensitivity = pickle.load(open(folder / 'backward_table.bin', 'rb'))
    for idx, eps in enumerate(epsilons):
        print_info('epsilon: {:3.3e}'.format(eps))
        tabular = PrettyTabular({
            'method': '{:^30s}',
            'loss sensitivity': '{:3.3f}',
            'grad sensitivity': '{:3.3e}'
        })
        for method in grad_sensitivity:
            print(tabular.row_string({
                'method': method,
                'loss sensitivity': loss_sensitivity[method][idx],
                'grad sensitivity': grad_sensitivity[method][idx]
                }))

    import matplotlib.pyplot as plt
    fig2 = plt.figure(figsize=(18, 7))
    ax_l = fig2.add_subplot(121)
    ax_g = fig2.add_subplot(122)
    titles = ['loss', '|grad|']
    ax_poses = [(0.07, 0.29, 0.40, 0.6),
        (0.56, 0.29, 0.40, 0.6)]
    for ax_pos, title, ax, l in zip(ax_poses, titles, (ax_l, ax_g), (loss_sensitivity, grad_sensitivity)):
        ax.set_position(ax_pos)
        ax.set_xlabel('epsilon (/)')
        ax.set_xscale('log')
        if 'grad' in title:
            ax.set_ylabel('Normalized change in |gradient| (/)')
            ax.set_yscale('log')
        else:
            ax.set_ylabel('Magnitude (/)')
        for method in grad_sensitivity:
            if 'pd' in method:
                color = 'tab:green'
            else:
                color = 'tab:blue'
            ax.plot(epsilons,l[method], label=method,
                color=color, linewidth=2)
        ax.grid(True)
        ax.set_title(title)
        handles, labels = ax.get_legend_handles_labels()

    # Share legends.
    fig2.legend(transpose_list(handles, 2, 1), transpose_list(labels, 2, 1),
        loc='upper center', ncol=1, bbox_to_anchor=(0.5, 0.19))
    fig2.savefig(folder / 'loss_and_grad_si.pdf')
    fig2.savefig(folder / 'loss_and_grad_si.png')

    plt.show()
