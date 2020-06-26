import sys
sys.path.append('../')

import os
from pathlib import Path
import numpy as np
from PIL import Image, ImageChops
from contextlib import contextmanager, redirect_stderr,redirect_stdout
from py_diff_pd.common.common import create_folder
from py_diff_pd.common.common import print_info, print_ok, print_error
from py_diff_pd.common.display import render_hex_mesh
from py_diff_pd.common.mesh import generate_hex_mesh
from py_diff_pd.core.py_diff_pd_core import Mesh3d

def test_render_hex_mesh(verbose):
    render_ok = True

    folder = Path('render_hex_mesh')
    if not os.path.exists(folder):
        create_folder(folder)

    voxels = np.ones((10, 10, 10))
    bin_file_name = 'cube.bin'
    generate_hex_mesh(voxels, 0.1, (0, 0, 0), bin_file_name)
    mesh = Mesh3d()
    mesh.Initialize(bin_file_name)

    render_hex_mesh(mesh, folder / 'render_hex_mesh.png')

    # Test if first render matches
    render_template = Image.open(folder / "hex_master_1.png").convert('LA')
    render = Image.open(folder / "render_hex_mesh.png").convert('LA')
    # Calculate pixelwise absolute error
    render_tolerance = 0
    renders_same = compareImages(render_template, render, render_tolerance)
    if not renders_same:
        render_ok = False
        if verbose:
            print_error("The hex mesh was rendered incorrectly")
        else:
            return render_ok
    elif verbose:
        print_ok("Render Succesful")
        os.system('eog {}/render_hex_mesh.png'.format(folder))

    # More advanced options.
    resolution = (600, 600)
    sample_num = 16
    # Scale the cube by 0.5, rotate along the vertical axis by 30 degrees, and translate by (0.5, 0.5, 0).
    transforms = [('s', 0.5), ('r', (np.pi / 6, 0, 0, 1)), ('t', (0.5, 0.5, 0))]
    render_hex_mesh(mesh, folder / 'render_hex_mesh.png', resolution=resolution, sample=sample_num, transforms=transforms)

    # Test if second render matches
    render_template = Image.open(folder / "hex_master_2.png").convert('LA')
    render = Image.open(folder / "render_hex_mesh.png").convert('LA')

    renders_same = compareImages(render_template, render, render_tolerance)
    if not renders_same:
        render_ok = False
        if verbose:
            print_error("The hex mesh was rendered incorrectly")
        else:
            return render_ok
    elif verbose:
        print_ok("Render Succesful")
        os.system('eog {}/render_hex_mesh.png'.format(folder))

    os.remove(bin_file_name)

    return render_ok

def compareImages(im1, im2, rtol):
    assert (rtol >= 0 or rtol <= 1), "Invalid render tolerance."
    assert im1.size == im2.size, "Render image is sized incorrectly."
    differenceImage = ImageChops.difference(im1,im2)
    WIDTH, HEIGHT = differenceImage.size
    data = list(differenceImage.getdata())
    data = [data[offset:offset+WIDTH] for offset in range(0, WIDTH*HEIGHT, WIDTH)]

    abs_error = 0
    for y in range(WIDTH):
        for x in range(HEIGHT):
            abs_error += data[y][x][0]

    rel_error = abs_error / (WIDTH*HEIGHT*255)

    if rel_error > rtol:
        return False

    return True


if __name__ == '__main__':
    verbose = False
    if not verbose:
        print_info("Testing render hex mesh...")
        if test_render_hex_mesh(verbose):
            print_ok("Test completed with no errors")
            sys.exit(0)
        else:
            print_error("Errors found in render hex mesh")
            sys.exit(-1)
    else:
        if test_render_hex_mesh(verbose):
            sys.exit(0)
        else:
            sys.exit(-1)
