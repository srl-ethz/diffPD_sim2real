import numpy as np 
import matplotlib.pyplot as plt

from numpy.polynomial import Polynomial


arr = np.load("Measurement_data/optimized_actuations.npy", allow_pickle=True)[()]
pressures = arr['pressures']
actuations = arr['actuations']


### Fitting curve to our "training data"
X_train, y_train = pressures[1:-1:2], actuations[1:-1:2]
curve = Polynomial.fit(X_train, y_train, 4)

x_cont = np.linspace(0, 450, 2000)
y_cont = curve(x_cont)
curve_string = ""
for i, c in enumerate(curve):
    if i==0:
        curve_string += f"{c:.3f}"
        continue
    curve_string += f" + {c:.3f} x**{i}"


### Generalization error on other data
X_test, y_test = pressures[0::2], actuations[0::2]
y_pred = curve(X_test)
errors = (y_pred - y_test)**2
print(f"Generalization Error: Mean {errors.mean():.2e} - Std {errors.std():.2e}")


fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(pressures[1:-1:2], actuations[1:-1:2], marker='o', facecolor='none', edgecolors='k', s=120, label='Single Optimized Actuation')
ax.plot(x_cont, y_cont, linewidth=1.5, c='k', linestyle='dashed', label=f"Continuous Actuation: \n {curve_string}")
#ax.scatter(X_test, y_pred, marker='x', c='k', s=120, label='Predicted Actuation')

plt.xticks(fontsize=22)
plt.yticks(fontsize=22)

ax.grid(which='minor', alpha=0.1)
ax.grid(which='major', alpha=0.5)   

ax.set_title("Pressure to Actuation Mapping", fontsize=28)
ax.set_xlabel("Pressure [mbar]", fontsize=24)
ax.set_ylabel("Muscle Actuation", fontsize=24)
ax.title.set_position([.5, 1.05])
ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.55), ncol=1, prop={'size': 24})

fig.savefig(f"pressure_to_actuation.png", bbox_inches='tight', dpi=300)
fig.savefig(f"pressure_to_actuation.pdf", bbox_inches='tight')
plt.close()

np.savetxt("AC2optim_allpoints.csv", np.stack([pressures, actuations], axis=1), delimiter=',', header="pressures, actuations")
np.savetxt("AC2optim_trainingpoints.csv", np.stack([X_train, y_train], axis=1), delimiter=',', header="pressures, actuations")
np.savetxt("AC2optim_testpoints.csv", np.stack([X_test, y_pred], axis=1), delimiter=',', header="pressures, actuations")
np.savetxt("AC2optim_contpoints.csv", np.stack([x_cont, y_cont], axis=1), delimiter=',', header="pressures, actuations")


