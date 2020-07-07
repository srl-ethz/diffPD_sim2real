# Differentiable Projective Dynamics

[![Travis CI Status](https://travis-ci.com/mit-gfx/diff_pd.svg?token=2N8A1xT9VhnH3M7Rxu74&branch=master)](https://travis-ci.com/mit-gfx/diff_pd)

## Recommended systems
- Ubuntu 18.04
- (Mini)conda 4.7.12 or higher

## Installation
```
git clone --recursive https://github.com/mit-gfx/diff_pd.git
cd diff_pd
conda env create -f environment.yml
conda activate diff_pd
./install.sh
```
If you would like to enable multi-threading, set the thread_ct in the options object in the python script. The examples below all use a default of 4 threads for parallel processes. Using 1 will force the program to run sequentially.

## Examples
Navigate to the `python/example` path and run `python [example_name].py` where the `example_name` could be the following:

### Display
- `render_hex_mesh` explains how to use the external renderer (pbrt) to render a 3D hex mesh.
- `render_quad_mesh` explains how to use matplotlib to render a 2D quad mesh.

### Numerical check
- `elastic_energy_2d` and `elastic_energy_3d` test the implementation of `ElasticEnergy`, `ElasticForce`, and `ElasticForceDifferential`.
- `state_force_2d` and `state_force_3d` test the implementation of state-based forces (e.g., friction, hydrodynamic force, penalty force for collisions) and their gradients w.r.t. position and velocity states.
- `pd_energy_2d` and `pd_energy_3d` test the implementation of vertex-based and element-based projective dynamics energies.
- `actuation_2d` and `actuation_3d` test the implementation of the muscle model.
- `pd_forward` verifies the forward simulation of projective dynamics by comparing it to the solutions from Newton's method.
- `deformable_backward_2d` uses central differencing to numerically check the gradients in Newton-PCG, Newton-Cholesky, and PD methods. A 2D rectangle is simulated with some fixed boundary conditions and a random but constant external force for 1 second at 30 fps. The loss is defined as a weighted sum of the final position and velocity and the gradients are computed by back-propagation.
- `deformable_backward_3d` tests the forward simulation and back-propagation in 3D with three methods (Newton-PCG, Newton-Cholesky, and PD) and with dirichlet boundary conditions, gravity, and collisions. `deformable_backward_3d` also plots the loss and magnitude of the three methods against the relative tolerance that was used to compute them.

### Quasi-static solvers
- `deformable_quasi_static_3d` solves the quasi-static state of a 3D hex mesh. The hex mesh's bottom and top faces are fixed but the top face is twisted.
- `rotating_deformable_quasi_static_2d` solves the quasi-static state of a 2D square in a rotational frame with its four edges fixed to the frame.
- `rotating_deformable_quasi_static_3d` solves the quasi-static state of a 3D hex mesh but in a rotational frame. The frame spins around the vertical direction at a constant speed and one of the face is fixed to the frame.

### Dynamic solvers
- `rotating_deformable_2d` solves the dynamic motion of a 2D rectangle in a rotational frame.
- `rotating_deformable_3d` solves the dynamic motion of a 3D cube in a rotational frame.

### Collisions
- `collision_3d` shows a 3D cuboid hitting on the ground with an implicit penalty force. The simulation is done with Newton-PCG, Newton-Cholesky, and PD. In the end, it will play three videos of almost identical motions, which also cross-validates the implementations of these three simulation methods. With the implicit method, the jumper becomes damped and a bit sticky.

### Demos
- `benchmark_3d` compares and reports the time cost of one forward call and one backward call in Newton-PCG, Newton-Cholesky, and PD. Below is the time cost on a benchmark cantilever beam with 4131 DoFs. We simulated the example for 30 frames with dt = `0.03`. `forward` and `backward` indicates the time cost for forward simulation (30 frames in total) and back propagation respectively. This result was generated with `OMP_NUM_THREADS=4`.
![benchmark](python/example/benchmark_3d/benchmark.png)
- `tendon_routing_3d` implements a simple tendon routing example with forward and backward PD and two Newton baselines. The goal is to let the endpoint of the stick finger reach a target point in the 3D space. 
- `sticky_finger_3d` optimizes a constant force applied to the nodes of a cuboid whose bottom is fixed on the ground. The goal is to bend the cuboid so that the upper-right corner reaches a target 3D position with zero velocity after 1 second. Optimization results with Newton-PCG, Newton-Cholesky, and PD are reported on the terminal, and a video of the final solution will pop up and play in the end.

## Implementation details of the `Deformable` class

### The dynamics model
Let `q` and `v` be the positions and velocities of nodes at the beginning of a time step. Let `q_next` and `v_next` be the position and velocity information at the end of this time step. This `Deformable` class considers the following internal and external forces:
- External force `f_ext` explicitly provide by the user;
- Elastic force `f_ela`: internal, defined on `q` or `q_next` depending on whether explicit or implicit Euler time integration is applied. This force is defined by setting the material model in the `Initialize` function. Currently, only `corotated` and `linear` materials are supported. Note that this force is ignored in projective dynamics;
- State based force `f_state` that depends on `q` and `v`. Examples are gravity, fricition, and hydrodynamic forces;
- Force induced by projective dynamics energy `f_pd`, defined on `q_next`. The projective dynamics energy is defined either on elements or on nodes. For elements, we assume it takes the form of `\Psi(q) = \|F - proj(F)\|^2` where `F` is the deformation gradient and `proj` is a projection operator. Currently, we support two element energies: `corotated` projects `F` to the set of rotational matrices and `volume` projects `F` to matrices with determinant being `1`. We typically use both to mimic the behavior of a real corotated material. In other words, you can think of the energy density function as `mu \|F - R\|^2 + la / 2 \|F - D\|^2`. Element energies are assumed to be defined on all elements. For vertex energies, we assume they have the form `\Psi(q) = \|q_i - proj(q_i)\|^2` where `q_i` represents the position of node `i`. We typically use vertex energies to model a soft collision penalty force, in which case the projection operator simply projects the node to the collision surface.
- Actuation force `f_act`, defined on `q` or `q_next` depending on whether explicit or implicit Euler time integration is applied. We implemented the muscle model discussed in the SoftCon paper: `\Psi(q) = \|Fm - r(e)Fm\|` where `r(e) = e/|Fm|`. You can think of it as projecting `Fm` to a sphere of radius being `e`.
- Dirichlet boundary conditions: this simply sets certain nodes to a given position.

### Quasi-static solver
```
f_ext + f_ela(q_next) + f_state(q, v) + f_pd(q_next) + f_act(q_next, u) = 0
```
or equivalently,
```
f_ela(q_next) + f_pd(q_next) + f_act(q_next, u) = -f_ext - f_state(q, v)
```

### Semi-implicit method
The governing equations are easy to formulate and solve in this case:
```
q_next = q + h * v_next
v_next = v + h / m * (f_ext + f_ela(q) + f_state(q, v) + f_pd(q) + f_act(q, u))
```
or equivalently
```
q_next = T(q + h * v + h2m * (f_ext + f_ela(q) + f_state(q, v) + f_pd(q) + f_act(q, u)))
```

### Newton's method
An implicit time integration method (Newton-PCG, Newton-Cholesky, and PD) aims to solve the following governing equations:
```
q_next = q + h * v_next
v_next = v + h / m * (f_ext + f_ela(q_next) + f_state(q, v) + f_pd(q_next) + f_act(q_next, u))
```
where DoFs specified by the Dirichlet boundary conditions are explicitly set in q_next, and therefore the corresponding v_next is also determined by `(q_next - q) / h`. The equations corresponding to Dofs in the Dirichlet boundary conditions are removed from the governing equations accordingly. We further rewrite it into one single equation:
```
q_next = q + h * v + h * h / m * (f_ext + f_ela(q_next) + f_state(q, v) + f_pd(q_next) + f_act(q_next, u))
```

Let `n` be the number of DoFs and let `m` be the number of DoFs in the Dirichlet boundary conditions. `q_next` is an `n`-dimensional vector but `m` elements are known. Let's rearrange the equation so that the unknown variables are on the left-hand side (We define h2m := h * h / m for simplicity):
```
q_next - h2m * (f_ela(q_next) + f_pd(q_next) + f_act(q_next, u)) = q + h * v + h2m * f_ext + h2m * f_state(q, v)
```
To see the structure of the left-hand side more clearly, let `q_sol` be the `n - m`-dimensional unknown variables to be solved. Define a linear mapping `S` and its inverse `T` as follows: `S` fetches `n - m`-dimensional free DoFs from an `n`-dimensional vector, and `T` recovers an `n`-dimensional vector from an `n - m`-dimensional free DoFs by filling the remaining `m` dimensions with the Dirichlet boundary condition values. The system we are solving now becomes:
```
q_sol - h2m * S(f_ela(T(q_sol)) + f_pd(T(q_sol)) + f_act(T(q_sol), u)) = rhs
```
where `rhs := S(q + h * v + h2m * f_ext + h2m * f_state(q, v))` can be pre-computed at the beginning of a time step.

Define `f_imp` as a function of `q_next` and `u`:
```
f_imp(q_next, u) = f_ela(q_next) + f_pd(q_next) + f_act(q_next, u)
```
Note that `u` is given in both forward and backward simulation. We simply need `u` in the definition of `f_imp` because we need to derive gradients w.r.t. it.

The equation now becomes
```
q_sol - h2m * S(f_imp(T(q_sol), u)) = rhs
```

Let `q_i` be the current guess of the solution. Newton's method iteratively updates `q_i` by solving
```
q_i + dq - h2m * S(f_imp(T(q_i), u)) - h2m * S * J * J_T * dq = rhs
```
or equivalently
```
(I - h2m * S * J * J_T) * dq = rhs - q_i + h2m * S(f_imp(T(q_i), u))
```
The core of Newton's method is to derive the Jacobian of the left-hand side nonlinear functions. Here `J_T`, the Jacobian of `T`, is very similar to `T` except that it fills Dirichlet DoFs with `0` instead of values in the boundary condition. Now consider the following system:
```
(I - h2m * J') * dq_aug = J_T * (rhs - q_i + h2m * S(f_imp(T(q_i), u)))
```
where `J'` zeros out all the rows and columns whose DoFs are fixed by the Dirichlet boundary conditions. We argue the solution `dq_aug` must be `dq_aug = J_T dq` and here is why:
```
J_T * (rhs - q_i + h2m * S(f_imp(T(q_i), u)))
= J_T * (I - h2m * S * J * J_T) * dq
= J_T * dq - h2m * J_T * S * J * J_T * dq
```
Note that `J_T * S * J` essentially zeros out corresponding rows of `J`. Moreover, since `J_T * dq` has zeros on fixed DoFs, the values in the corresponding columns of `J_T * S * J` does not matter. Using these two facts, we have
```
J_T * dq - h2m * J_T * S * J * J_T * dq
= J_T * dq - h2m * J' * J_T * dq
= (I - h2m * J') * (J_T * dq)
```
This confirms that `J_T * dq` is one solution to the dual system. Since `I - h2m * J`' is symmetric positive definite, the solution is unique, finihsing our proof. This shows a (well known) less destructive method to update Newton's iterations.

### PD method
The governing equations are
```
q_next = q + h * v_next
v_next = v + h / m * (f_ext + f_ela(q_next) + f_state(q, v) + f_pd(q_next) + f_act(q_next, u))
```
Note that f_ela is not supported in PD, we have
```
q_next = q + h * v + h2m * f_ext + h2m * f_state(q, v) + h2m * f_pd(q_next) + h2m * f_act(q_next, u)
```
or equivalently,
```
q_next - h2m * f_pd(q_next) - h2m * f_act(q_next, u) = q + h * v + h2m * f_ext + h2m * f_state(q, v)
```
Let `rhs := q + h * v + h2m * f_ext + h2m * f_state(q, v)` and notice that
```
f_pd(q_next) = -w_i S'A'(ASq_next - proj(q_next))
```
for element energy, and
```
f_pd(q_next)_i = -w_i (q_i - proj(q_i))
```
for vertex energy. For muscles, we have
```
f_act(q_next, u)_i = -w_i S'A'M'(MASq_next - proj(q_next))
```
As a result, the governing equations become:
```
q_next + h2m * w_i * S'A'AS q_next + h2m * w_i * q_i + h2m * w_i * S'A'M'MASq_next
= rhs + h2m * w_i * S'A' * proj(q_next) + h2m * w_i * proj(q_i) + h2m * w_i * S'A'M' proj(q_next)
```
Therefore, the PD-style updates are:
```
(I + h2m * w_i * S'A'AS + h2m * w_i + h2m * w_i * S'A'M'MAS)q_next = rhs + ...
```
To incorporate Dirichlet boundary conditions, we zero out rows and cols in the matrix after the first plus on the left-hand side.