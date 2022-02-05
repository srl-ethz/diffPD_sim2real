import numpy as np
import matplotlib.pyplot as plt


def plotActuation (actuation, dt):
	frame_num = len(actuation)
	times = []

	for i in range(frame_num):
	    times.append(i*dt)

	fig, ax = plt.subplots(figsize=(12,8))
	ax.plot(times, actuation, marker='o', markersize=4)
	   

	major_ticks = np.arange(0, frame_num*dt+0.05* frame_num*dt, 0.2)
	minor_ticks = np.arange(0, frame_num*dt+0.05* frame_num*dt, 0.04)
	ax.set_xticks(major_ticks)
	ax.set_xticks(minor_ticks, minor=True)

	major_ticks_y = np.arange(0.5, 3, 0.4)
	minor_ticks_y = np.arange(0.5, 3, 0.08)
	ax.set_yticks(major_ticks_y)
	ax.set_yticks(minor_ticks_y, minor=True)
	plt.xticks(fontsize=22)
	plt.yticks(fontsize=22)

	ax.grid(which='minor', alpha=0.2)
	ax.grid(which='major', alpha=0.5)

	ax.set_title("Multiactuation over Time", fontsize=28)
	ax.set_xlabel("Time [s]", fontsize=24)
	ax.set_ylabel("Muscle Actuation", fontsize=24)
	ax.title.set_position([.5, 1.03])
	#ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.35), ncol= 2, prop={'size': 24})

	fig.savefig(f"multiActuation.png", bbox_inches='tight')
	plt.close()
	


act = np.array([0.18081628, 0.33521603, 0.44212562, 0.51563384, 0.62284055, 0.66023927,
  0.73072006, 0.77621254, 0.78634246, 0.82055789, 0.83328386, 0.84924543,
  0.86258292, 0.86891084, 0.88203647, 0.88770833, 0.89842741, 0.90506757,
  0.91052297, 0.91650377, 0.92072227, 0.92739767, 0.93398631, 0.9412044,
  0.94791222, 0.95362008, 0.96341529, 0.97076898, 0.97668869, 0.98281742,
  0.9895724,  0.99575399, 1.00156311, 1.00691611, 1.01258991, 1.01726384,
  1.02175402, 1.02852799, 1.03442112, 1.03503342, 1.03848881, 1.04892388,
  1.05282903, 1.04973751, 1.05467312, 1.06006326, 1.06511551, 1.06606376,
  1.10619253, 0.94857512])
  
act += 1

plotActuation(act, 0.02)

