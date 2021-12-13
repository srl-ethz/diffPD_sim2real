# ------------------------------------------------------------------------------
# Numerical Damping Compensation - Case A-1
# ------------------------------------------------------------------------------
### Import some useful functions
import sys
sys.path.append('../')

from pathlib import Path
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Utility functions 
from utils import read_measurement_data

from Environments.beam_env_damp_comp import BeamEnv




def envelope (x, a, b, c):
    return a * np.exp(-b * x) + c


### MAIN
if __name__ == '__main__':
    seed = 42
    np.random.seed(seed)
    folder = Path('Numerical_Damping_Compensation_Case_A-1')
    
    ### Motion Markers data
    qs_real = read_measurement_data(65,228,'Measurement_data/beam_gravity_V2_b.c3d')
    
    # Material parameters: Dragon Skin 10 
    youngs_modulus = 263824 # Optimized value
    poissons_ratio = 0.499
    density = 1.07e3

    # Gravity
    state_force = [0, 0, -9.80709]

    # Create simulation scene
    hex_params = {
        'density': density,
        'youngs_modulus': youngs_modulus,
        'poissons_ratio': poissons_ratio,
        'state_force_parameters': state_force,
        'mesh_type': 'hex',
        'refinement': 2.35 
    }

    hex_env = BeamEnv(seed, folder, hex_params,0,'A-1', 0.01)
    hex_deformable = hex_env.deformable()

    ### Optimize for the best frame
    R, t = hex_env.fit_realframe(qs_real[0])
    qs_real = qs_real @ R.T + t
    qs_real = qs_real[:, 1, 2]
        
        
        
    ### Get upper and lower envelopes
    u_x = [0,]
    u_y = [qs_real[0],]

    l_x = []
    l_y = []
        
    for k in range(1,len(qs_real)-1):
        if (np.sign(qs_real[k]-qs_real[k-1])==1) and (np.sign(qs_real[k]-qs_real[k+1])==1):
            u_x.append(k*0.01)
            u_y.append(qs_real[k])

        if (np.sign(qs_real[k]-qs_real[k-1])==-1) and ((np.sign(qs_real[k]-qs_real[k+1]))==-1):
            l_x.append(k*0.01)
            l_y.append(qs_real[k])

    ### Fit curve
    upper, _ = curve_fit(envelope, u_x, u_y)    
    lower, _ = curve_fit(envelope, l_x, l_y)
    print(f"Upper: {upper}")
    print(f"Lower: {lower}")
    
    x = np.linspace(0, 0.01*len(qs_real), 1000)
    y_up = envelope(x, *upper)
    y_low = envelope(x, *lower)
    
    
    fig, ax = plt.subplots(figsize=(12,8))
    ax.plot(np.array([i*0.01 for i in range(len(qs_real))]), qs_real, marker='o', markersize=4, label='Real Data')
    ax.plot(x, y_up, linewidth=1.5, c='k', linestyle='dashed', label=f'Fitted Upper Envelope: {upper[0]:.3f} * e^{upper[1]:.3f} + {upper[2]:.3f}')
    ax.plot(x, y_low, linewidth=1.5, c='k', linestyle='dashed', label=f'Fitted Lower Envelope: {lower[0]:.3f} * e^{lower[1]:.3f} + {lower[2]:.3f}')
    ax.scatter(u_x, u_y, c='r')
    ax.scatter(l_x, l_y, c='r')

    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)

    #ax.grid(which='minor', alpha=0.1)
    #ax.grid(which='major', alpha=0.5)   

    ax.set_title("Damping Envelope Fit", fontsize=28)
    ax.set_xlabel("Time [s]", fontsize=24)
    ax.set_ylabel("z-position [m]", fontsize=24)
    ax.title.set_position([.5, 1.05])
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.55), ncol=1, prop={'size': 24})

    fig.savefig(f"envelope.png", bbox_inches='tight', dpi=300)
    plt.close()


