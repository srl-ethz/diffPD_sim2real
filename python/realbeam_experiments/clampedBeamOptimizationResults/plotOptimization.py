import numpy as np
import matplotlib.pyplot as plt


losses1 = [
    1.7194e-1,
    7.1626e-2,
    6.6190e-3,
    5.0569e-3,
    4.7024e-3,
    4.6915e-3,
    4.6913e-3,
    4.6912e-3,
    4.6912e-3,
]

modulus1 = [
    100000.00000000001,
    141966.2356878344,
    252568.7270164775,
    274281.9245125423,
    290278.04843427247,
    293115.6676742373,
    293519.99698872596,
    293504.6033937859,
    293506.21331244265,
]

# Change to kPa
modulus1 = [m/1000 for m in modulus1]




losses2 = [
    4.4981e-02,
    4.3850e-02,
    4.2665e-02,
    3.7746e-02,
    1.6117e-02,
    4.3502e-01,
    9.5301e-03,
    2.1847e+01,
    3.1954e-01,
    6.1422e-03,
    5.8172e-03,
    4.7613e-03,
    4.6934e-03,
    4.6912e-03,
    4.6912e-03,
    4.6912e-03,
    4.6913e-03,
    4.6912e-03
]

modulus2 = [
    999999.9999999995,
    967090.0585259976,
    934629.9431331559,
    815325.0344609075,
    472166.450281134,
    53107.091301696724,
    390146.51350248576,
    999.9999999999998,
    69015.784189512,
    340145.3078131055,
    333910.37672163895,
    284722.5689532715,
    295058.47774240444,
    293555.8929310556,
    293505.71711584035,
    293552.7875639384,
    293552.7543170655,
    293552.7875639384
]

# Change to kPa
modulus2 = [m/1000 for m in modulus2]



# Match lengths, as modulus1 is shorter
for i in range(len(modulus2)-len(modulus1)):
    modulus1.append(293552/1000)
    losses1.append(4.6912e-03)


# Log mapping
modulus1 = [np.log(v) for v in modulus1]
modulus2 = [np.log(v) for v in modulus2]
losses1 = [np.log(v) for v in losses1]
losses2 = [np.log(v) for v in losses2]



fig, ax = plt.subplots(figsize=(12,8))
ax.plot(modulus1, marker='o', markersize=4, label='Initialization 100kPa')
ax.plot(modulus2, marker='o', markersize=4, label='Initialization 1MPa')
#ax.scatter(modulus, marker='x', s=80)


#major_ticks = np.arange(50, 500, 50)
#minor_ticks = np.arange(50, 500, 25)
#ax.set_xticks(major_ticks)
#ax.set_xticks(minor_ticks, minor=True)

plt.xticks(fontsize=22)
plt.yticks(fontsize=22)

ax.grid(which='minor', alpha=0.1)
ax.grid(which='major', alpha=0.5)   

ax.set_title("Young's Modulus Optimization", fontsize=28)
ax.set_xlabel("Optimization Iteration", fontsize=24)
ax.set_ylabel("Logarithmic Young's Modulus [kPa]", fontsize=24)
ax.title.set_position([.5, 1.03])
ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.35), ncol= 2, prop={'size': 24})

fig.savefig(f"clampedBeamCaseE_parameterOptimization.pdf", bbox_inches='tight')
fig.savefig(f"clampedBeamCaseE_parameterOptimization.png", bbox_inches='tight', dpi=300)
plt.close()



fig, ax = plt.subplots(figsize=(12,8))
ax.plot(losses1, marker='o', markersize=4, label='Initialization 100kPa')
ax.plot(losses2, marker='o', markersize=4, label='Initialization 1MPa')

plt.xticks(fontsize=22)
plt.yticks(fontsize=22)

ax.grid(which='minor', alpha=0.1)
ax.grid(which='major', alpha=0.5)   

ax.set_title("Young's Modulus Optimization", fontsize=28)
ax.set_xlabel("Optimization Iteration", fontsize=24)
ax.set_ylabel("Logarithmic MSE Loss", fontsize=24)
ax.title.set_position([.5, 1.03])
ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.35), ncol= 2, prop={'size': 24})

fig.savefig(f"clampedBeamCaseE_parameterOptimization_loss.png", bbox_inches='tight', dpi=300)
plt.close()

