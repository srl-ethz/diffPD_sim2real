# Differentiable Projective Dynamics

The work and examples in this project are based on the original work of https://github.com/mit-gfx/diff_pd. This is the code accompanying our paper "**Sim-to-Real  for  Differentiable  Projective  Dynamics  on  Soft  Robots: Meshing,  Damping,  and  Actuation**" to run the simulation experiments that are shown.


## Recommended systems
- Ubuntu 20.04
- (Mini)conda 4.7.12 or higher
- GCC 7.5 (Other versions might work but we tested the codebase with 7.5 only)

## Installation
```
git clone --recursive git@github.com:srl-ethz/diffPD_sim2real.git
cd diffPD_sim2real
conda env create -f environment.yml
conda activate diffPD_sim2real
./install.sh
```

## Examples
Navigate to `python/[scenario]` where `scenario` is one of the following:

### realbeam_experiments
Here you'll find the examples for the Clamped Beam under External Force, run `python [example_name]` where `example_name` is one of the following:
- `clamped_beam_Case_A-1.py`: Generates the results for the first case from TABLE 1 in our paper. Similarly, the other cases can also be run. Running these examples with the `--video` flag will create a video of the simulation as well, however, this will require much more time to run.

### 1segment_arm
Here you'll find the examples for the Soft Robotic Arm, run `python [example_name].py` where `example_name` is one of the following:
- `soft1arm.py`: Run the simulation for accurate, high resolution single segment robotic arm mesh.
- `muscles_AC1.py`: Run the optimization for the single segment robotic arm with muscle model AC1 from our paper. 
- `muscles_AC2.py`: Run the optimization for the single segment robotic arm with muscle model AC2 from our paper. 
- `muscles_AC2_multiActuation.py`: Run the optimization for the single segment robotic arm with muscle model AC2 from our paper using multiple actuations, every 5 frames (0.05s), now matching the dynamic behavior of the real pressure actuation better.

### soft_fish
Here you'll find the examples for the Soft Fish Tail, run `python [example_name].py` where `example_name` is one of the following:
- `soft_fish_2muscles.py`: Run the optimization for the soft fish tail with 2 muscles, one extending and one contracting, where the contraction is always half of the extension. We optimize a single muscle actuation for the whole motion of pressure actuation.
- `soft_fish_2muscles_multiActuation.py`: Optimization for multiple actuations, every 20 frames (0.2s), now matching the dynamic behavior of the real pressure actuation better.

