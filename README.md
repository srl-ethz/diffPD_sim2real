# Differentiable Projective Dynamics

## Recommended systems
- Ubuntu 18.04
- (Mini)conda 4.7.12 or higher

## Installation
```
git clone --recursive git@github.com:mit-gfx/diff_pd.git
cd diff_pd
conda env create -f environment.yml
conda activate diff_pd
./install.sh
```

## Examples
Navigate to the `python/example` path and run `python [example_name].py` where the `example_name` could be the following:

### Display
- `render_hex_mesh` explains how to use the external renderer (pbrt) to render a 3D hex mesh.
- `render_quad_mesh` explains how to use matplotlib to render a 2D quad mesh.

### Numerical check
- `elastic_energy` tests the implementation of `ElasticEnergy`, `ElasticForce`, and `ElasticForceDifferential`.

### Quasi-static solvers
- `deformable_quasi_static_3d` solves the quasi-static state of a 3D hex mesh. The hex mesh's bottom and top faces are fixed but the top face is twisted.
- `rotating_deformable_quasi_static_2d` solves the quasi-static state of a 2D square in a rotational frame with its four edges fixed to the frame.
- `rotating_deformable_quasi_static_3d` solves the quasi-static state of a 3D hex mesh but in a rotational frame. The frame spins around the vertical direction at a constant speed and one of the face is fixed to the frame.

### Dynamic solvers
- `rotating_deformable_2d` solves the dynamic motion of a 2D rectangle in a rotational frame.
- `rotating_deformable_3d` solves the dynamic motion of a 3D cube in a rotational frame.