import os
import numpy as np

def generate_rectangle_obj(cell_nums, dx, origin, obj_file_name):
    nx, ny = cell_nums
    nx += 1
    ny += 1
    with open(obj_file_name, 'w') as f:
        for i in range(nx):
            for j in range(ny):
                f.write('v {} {} 0\n'.format(origin[0] + i * dx, origin[1] + j * dx))
        f.write('\n')
        for i in range(nx - 1):
            for j in range(ny - 1):
                f.write('f {} {} {} {}\n'.format(
                    i * ny + j + 1,
                    (i + 1) * ny + j + 1,
                    (i + 1) * ny + j + 2,
                    i * ny + j + 2,
                ))