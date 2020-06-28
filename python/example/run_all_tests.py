import sys
sys.path.append('../')

from py_diff_pd.common.common import print_ok, print_error

# Display.
from render_hex_mesh import test_render_hex_mesh
from render_quad_mesh import test_render_quad_mesh

if __name__ == '__main__':
    tests = [
        (test_render_hex_mesh, 'render_hex_mesh'),
        (test_render_quad_mesh, 'render_quad_mesh'),
    ]
    for test_func, test_name in tests:
        if test_func(verbose=False):
            print_ok('[{}] PASSED.'.format(test_name))
        else:
            print_error('[{}] FAILED.'.format(test_name))
            sys.exit(-1)