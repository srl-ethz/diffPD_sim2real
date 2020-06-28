import sys
sys.path.append('../')

from py_diff_pd.common.common import print_ok, print_error

# Display.
from render_hex_mesh import test_render_hex_mesh

if __name__ == '__main__':
    if test_render_hex_mesh(verbose=False):
        print_ok('[render_hex_mesh] PASSED.')
    else:
        print_error('[render_hex_mesh] FAILED.')
        sys.exit(-1)