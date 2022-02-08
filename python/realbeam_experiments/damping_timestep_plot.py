import numpy as np 
import matplotlib.pyplot as plt


timesteps = np.flip(np.arange(2e-2, 0.24e-2, -0.25e-2))
damping = np.flip([
    0.005577240672069684,
    0.00493906996465535,
    0.004144966473112475,
    0.0033642089853526333,
    0.0025690735199912895,
    0.001727949571555505,
    0.0008479616826417235,
    -1.241069279772321e-05
])

critical_damping = np.flip([
    0.007382461764075759,
    0.006454885556833581,
    0.005544007592831606,
    0.00461652523239477,
    0.0036922887972658937,
    0.0027715548895692935,
    0.0018469585456226357,
    0.0009222746589705064
])


fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(timesteps, damping, marker='o', facecolor='none', edgecolors='k', s=120, label='Optimized Damping Parameters')
ax.scatter(timesteps, critical_damping, marker='x', c='r', s=120, label='Critical Damping Parameters')
#ax.plot(x_cont, y_cont, linewidth=1.5, c='k', linestyle='dashed', label=f"Continuous Actuation: \n {curve_string}")
#ax.scatter(X_test, y_pred, marker='x', c='r', s=120, label='COMSOL')

plt.xticks(fontsize=22)
plt.yticks(fontsize=22)

ax.grid(which='minor', alpha=0.1)
ax.grid(which='major', alpha=0.5)   

ax.set_title("Damping Parameter for different Timesteps", fontsize=28)
ax.set_xlabel("dt [s]", fontsize=24)
ax.set_ylabel("Damping Parameter", fontsize=24)
ax.title.set_position([.5, 1.05])
ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.35), ncol=1, prop={'size': 24})

fig.savefig(f"damping_parameter.png", bbox_inches='tight', dpi=300)
plt.close()



