# Differentiable Projective Dynamics

### Recommended systems
- Ubuntu 18.04
- (Mini)conda 4.7.12 or higher

### Installation
```
git clone --recursive git@github.com:mit-gfx/diff_pd.git
cd diff_pd
conda env create -f environment.yml
conda activate diff_pd
./install.sh
```

### Examples
```
cd python
python open_loop_demo_2d.py
eog open_loop_demo_2d/final.gif
```