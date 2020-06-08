import os
import struct
import numpy as np

def generate_rectangle_mesh(cell_nums, dx, origin, bin_file_name):
    nx, ny = cell_nums
    with open(bin_file_name, 'wb') as f:
        # Load.
        vertex_num = (nx + 1) * (ny + 1)
        face_num = nx * ny
        f.write(struct.pack('i', 2))
        f.write(struct.pack('i', 4))
        # Vertices.
        f.write(struct.pack('i', 2))
        f.write(struct.pack('i', vertex_num))
        for i in range(nx + 1):
            for j in range(ny + 1):
                vx = origin[0] + i * dx
                f.write(struct.pack('d', vx))
        for i in range(nx + 1):
            for j in range(ny + 1):
                vy = origin[1] + j * dx
                f.write(struct.pack('d', vy))

        # Faces.
        f.write(struct.pack('i', 4))
        f.write(struct.pack('i', face_num))
        for i in range(nx):
            for j in range(ny):
                f.write(struct.pack('i', i * (ny + 1) + j))
        for i in range(nx):
            for j in range(ny):
                f.write(struct.pack('i', (i + 1) * (ny + 1) + j))
        for i in range(nx):
            for j in range(ny):
                f.write(struct.pack('i', (i + 1) * (ny + 1) + j + 1))
        for i in range(nx):
            for j in range(ny):
                f.write(struct.pack('i', i * (ny + 1) + j + 1))