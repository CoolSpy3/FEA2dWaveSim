
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# Grid Params
x_vals = np.arange(0, 20, 0.05)
y_vals = np.arange(0, 20, 0.05)

# Animation Params
t_vals = np.arange(0, 60, 0.1)
t_step = t_vals[1] - t_vals[0]

animation_speed = 1

# Wave Params
k = 1
w = 1
A = 1

fig, ax = plt.subplots()
ax.set_xlim(min(x_vals), max(x_vals))
ax.set_ylim(min(y_vals), max(y_vals))

data_matrix = np.zeros((len(t_vals), len(y_vals), len(x_vals)))

# Generate Data
max_y = max(y_vals)
print("Computing Simulation Data")
for n, t in enumerate(tqdm(t_vals)):
	for y_idx, y in enumerate(tqdm(y_vals, leave=False, delay=1)):
		for x_idx, x in enumerate(x_vals):
			r = np.sqrt(x**2 + (y - (max_y / 2))**2)
			data_matrix[n][y_idx][x_idx] = A*np.sin(k*r - w*t) / (np.sqrt(r) or 1)

map_data = ax.imshow(
	np.zeros((len(y_vals), len(x_vals))),
	vmin=np.min(data_matrix), vmax=np.max(data_matrix),
	extent=(min(x_vals), max(x_vals), min(y_vals), max(y_vals)))

def frame(t):
	map_data.set_data(data_matrix[int(t / t_step)])

	return (map_data,)

ani = FuncAnimation(fig, frame, t_vals, interval=t_step*animation_speed, blit=False)
plt.show()
