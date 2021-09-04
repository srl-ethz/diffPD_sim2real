# Differentiable Projective Dynamics

The work and examples in this project are based on the original work of https://github.com/mit-gfx/diff_pd. This is the code accompanying our paper "Sim-to-Real  for  Differentiable  Projective  Dynamics  on  Soft  Robots:Meshing,  Damping,  and  Actuation" to run the simulation experiments that are shown.


## Recommended systems
- Ubuntu 20.04
- (Mini)conda 4.7.12 or higher
- GCC 7.5 (Other versions might work but we tested the codebase with 7.5 only)

## Installation
```
git clone --recursive git@github.com:srl-ethz/diffPD_sim2real.git
cd diff_pd
conda env create -f environment.yml
conda activate diff_pd
./install.sh
```

## Examples
Navigate to `python/[scenario]` where `scenario` is one of the following:

### realbeam_experiments
Here you'll find the examples for the Clamped Beam under External Force, run `python [example_name]` where `example_name` is one of the following:
- `clamped_beam_Case_A-1.py` 

### 1segment_arm
Here you'll find the examples for the Soft Robotic Arm, run `python [example_name].py` where `example_name` is one of the following:
- `muscles_AC1.py` 

### soft_fish
Here you'll find the examples for the Soft Fish Tail, run `python [example_name].py` where `example_name` is one of the following:
- `soft_fish.py` 

