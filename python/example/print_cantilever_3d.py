import sys
sys.path.append('../')

from pathlib import Path
import pickle
import matplotlib.pyplot as plt
import numpy as np

from py_diff_pd.common.common import print_info

if __name__ == '__main__':
    for thread_ct in [2, 4, 8]:
        data_file = Path('cantilever_3d') / 'data_{:04d}_threads.bin'.format(thread_ct)
        if data_file.exists():
            print_info('Loading {}'.format(data_file))
            data = pickle.load(open(data_file, 'rb'))
            for method in ['newton_pcg', 'newton_cholesky', 'pd']:
                total_time = 0
                avg_forward = 0
                average_backward = 0
                iter = 0
                for d in data[method]:
                    print('loss: {:8.3f}, |grad|: {:8.3f}, E: {:8.3e}, nu: {:4.3f}, forward time: {:6.3f}s, backward time: {:6.3f}s'.format(
                        d['loss'], np.linalg.norm(d['grad']), d['E'], d['nu'], d['forward_time'], d['backward_time']))
                    total_time += d['forward_time'] + d['backward_time']
                    average_backward += d['backward_time']
                    avg_forward += d['forward_time']
                    iter += 1
                avg_forward /= iter
                average_backward /= iter
                print_info('Optimizing with {} finished in {:6.3f} seconds. Average Backward time: {}, Average Forward Time = {}'.format(method, total_time, average_backward, avg_forward))

    plt.rc('pdf', fonttype=42)
    plt.rc('font', size=16)             # Controls default text sizes.
    plt.rc('axes', titlesize=16)        # Fontsize of the axes title.
    plt.rc('axes', labelsize=16)        # Fontsize of the x and y labels.
    plt.rc('xtick', labelsize=16)       # Fontsize of the tick labels.
    plt.rc('ytick', labelsize=16)       # Fontsize of the tick labels.
    plt.rc('legend', fontsize=16)       # Legend fontsize.
    plt.rc('figure', titlesize=16)      # Fontsize of the figure title.
    Es = {}
    nus = {}
    losses = {}
    for method in ['newton_pcg', 'newton_cholesky', 'pd']:
        Es[method] = [d['E'] for d in data[method]]
        nus[method] = [d['nu'] for d in data[method]]
        losses[method] = [d['loss'] for d in data[method]]
    fig = plt.figure(figsize=(18, 8))
    ax_E = fig.add_subplot(131)
    ax_E.plot([1e6 for _ in range(iter)], label='ground truth', linestyle='--', color='tab:orange', linewidth=2)
    ax_nu = fig.add_subplot(132)
    ax_nu.plot([0.45 for _ in range(iter)], label='ground truth', linestyle='--', color='tab:orange', linewidth=2)
    ax_loss = fig.add_subplot(133)
    ax_loss.plot([0 for _ in range(iter)], label='ground truth', linestyle='--', color ='tab:orange', linewidth=2)
    titles = ['Young\'s modulus estimate', 'Poisson\'s ratio estimate', 'Loss']
    for title, ax, y in zip(titles, (ax_E, ax_nu, ax_loss), (Es, nus, losses)):
        if 'modulus' in title:
            ax.set_ylabel("log(Young's modulus) (Pa)")
            ax.ticklabel_format(axis='y', style='sci')
            ax.set_yscale('log')
        elif 'Poisson' in title:
            ax.set_ylabel("log(Poisson's ratio) (/)")
            ax.set_yscale('log')
        else:
            ax.set_ylabel("Magnitude")
        ax.set_xlabel('iterations')
        for method, color in zip(['newton_pcg', 'newton_cholesky', 'pd'], ['tab:blue', 'tab:red', 'tab:green']):
            ax.plot(y[method], label=method, color=color, linewidth=2)
        ax.grid(True)
        ax.set_title(title)
        ax.legend()

    fig.tight_layout()

    fig.savefig(folder / 'parameter_estimation.pdf')
    fig.savefig(folder / 'parameter_estimation.png')
    plt.show()
