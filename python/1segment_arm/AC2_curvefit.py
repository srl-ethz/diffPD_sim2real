from matplotlib import markers
import numpy as np 
import matplotlib.pyplot as plt

from numpy.polynomial import Polynomial


arr = np.load("Measurement_data/optimized_actuations.npy", allow_pickle=True)[()]
pressures = arr['pressures']
actuations = arr['actuations']


def ridge_regression (X_input, y, alpha):
    # Computes the least squares regression problem with regularization term alpha. Loss is ||y - W.T * X||^2 + alpha * W.T * W
    # Fourth order term
    order = 4
    X = np.stack([X_input**i for i in range(order+1)], axis=-1)
    
    # Normalize
    #X_max = np.max(X)
    X_max = 1
    normed_X = X / X_max
    inv_term = np.linalg.inv(np.matmul(normed_X.T, normed_X) + alpha * np.eye(order+1))
    weights = np.matmul(np.matmul(inv_term, normed_X.T), y)
    
    # Renormalize
    weights = weights / X_max
    
    return weights


### Fitting curve to our "training data"
X_train, y_train = pressures[1:-1:2], actuations[1:-1:2]
curve = Polynomial.fit(X_train, y_train, 4)
# Curve with regularization
# coeffs = ridge_regression(X_train, y_train, alpha=1e1)
# curve = Polynomial(coeffs)
# print(coeffs)

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
ax.scatter(pressures, actuations, marker='o', facecolor='none', edgecolors='k', s=120, label='Single Optimized Actuation')
ax.plot(x_cont, y_cont, linewidth=1.5, c='k', linestyle='dashed', label=f"Continuous Actuation: \n {curve_string}")
ax.scatter(X_test, y_pred, marker='x', c='k', s=120, label='Predicted Actuation')

plt.xticks(fontsize=22)
plt.yticks(fontsize=22)

#ax.grid(which='minor', alpha=0.1)
#ax.grid(which='major', alpha=0.5)   

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




### Perform k-fold cross validation on the input data
# k = 6
# # best even split
# rest = np.ceil(len(pressures) / k) - len(pressures) / k
# testset_size = int(np.ceil(len(pressures) / k)) if ((k-1) * rest) < len(pressures) / k else len(pressures) // k
                   
# for i in range(k):
#     X_train = np.concatenate([pressures[:i*testset_size], pressures[(i+1)*testset_size:]], axis=0)
#     y_train = np.concatenate([actuations[:i*testset_size], actuations[(i+1)*testset_size:]], axis=0)
    
#     X_test = pressures[i*testset_size:(i+1)*testset_size]
#     y_test = actuations[i*testset_size:(i+1)*testset_size]
    
    



