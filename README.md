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
- `pd_forward` verifies the forward simulation of projective dynamics by comparing it to the solutions from Newton's method.
- `deformable_backward_2d` uses central differencing to numerically check the gradients in Newton-PCG, Newton-Cholesky, and PD methods. A 2D rectangle is simulated with some fixed boundary conditions and a random but constant external force for 1 second at 30 fps. The loss is defined as a weighted sum of the final position and velocity and the gradients are computed by back-propagation.
- `deformable_backward_3d` is teh same as `deformabled_backward_2d` but the test is done in 3D.

### Quasi-static solvers
- `deformable_quasi_static_3d` solves the quasi-static state of a 3D hex mesh. The hex mesh's bottom and top faces are fixed but the top face is twisted.
- `rotating_deformable_quasi_static_2d` solves the quasi-static state of a 2D square in a rotational frame with its four edges fixed to the frame.
- `rotating_deformable_quasi_static_3d` solves the quasi-static state of a 3D hex mesh but in a rotational frame. The frame spins around the vertical direction at a constant speed and one of the face is fixed to the frame.

### Dynamic solvers
- `rotating_deformable_2d` solves the dynamic motion of a 2D rectangle in a rotational frame.
- `rotating_deformable_3d` solves the dynamic motion of a 3D cube in a rotational frame.

### Demos (TODO)
- `profile_3d`
- `sticky_finger_3d`