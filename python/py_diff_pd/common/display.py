import numpy as np
import matplotlib.pyplot as plt
from matplotlib import collections as mc
from py_diff_pd.core.py_diff_pd_core import Mesh2d
from py_diff_pd.common.common import ndarray

def display_quad_mesh(quad_mesh, xlim=None, ylim=None, title=None, file_name=None, show=True):
    vertex_num = quad_mesh.NumOfVertices()
    face_num = quad_mesh.NumOfElements()

    fig = plt.figure()
    ax = fig.add_subplot()
    lines = []
    for i in range(face_num):
        f = ndarray(quad_mesh.py_face(i))
        j01 = [(0, 1), (1, 3), (3, 2), (2, 0)]
        for j0, j1 in j01:
            j0 = int(f[j0])
            j1 = int(f[j1])
            v0 = ndarray(quad_mesh.py_vertex(j0))
            v1 = ndarray(quad_mesh.py_vertex(j1))
            lines.append((v0, v1))
    ax.add_collection(mc.LineCollection(lines, colors='tab:red', alpha=0.5))

    ax.set_aspect('equal')
    v = ndarray(lines)
    padding = 0.5
    x_min = np.min(v[:, :, 0]) - padding
    x_max = np.max(v[:, :, 0]) + padding
    y_min = np.min(v[:, :, 1]) - padding
    y_max = np.max(v[:, :, 1]) + padding
    if xlim is None:
        ax.set_xlim([x_min, x_max])
    else:
        ax.set_xlim(xlim)
    if ylim is None:
        ax.set_ylim([y_min, y_max])
    else:
        ax.set_ylim(ylim)
    ax.set_xticks([])
    ax.set_yticks([])
    if title is not None:
        ax.set_title(title)
    if file_name is not None:
        fig.savefig(file_name)
    if show:
        plt.show()
    plt.close()

def display_hex_mesh(quad_mesh, xlim=None, ylim=None, title=None, file_name=None, show=True):
    pass

import imageio
import os
def export_gif(folder_name, gif_name, fps, name_prefix=''):
    frame_names = [os.path.join(folder_name, f) for f in os.listdir(folder_name)
        if os.path.isfile(os.path.join(folder_name, f)) and f.startswith(name_prefix) and f.endswith('.png')]
    frame_names = sorted(frame_names)

    # Read images.
    images = [imageio.imread(f) for f in frame_names]
    if fps > 0:
        imageio.mimsave(gif_name, images, fps=fps)
    else:
        imageio.mimsave(gif_name, images)