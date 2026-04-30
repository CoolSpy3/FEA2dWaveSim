
#region Imports
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from tqdm import tqdm

#endregion

#region Params

# Grid Params
x_vals = np.linspace(0, 10, 100)
y_vals = np.linspace(0, 10, 100)

# Animation Params
t_vals = np.linspace(0, 80, 200) #step size should be >  seconds
t_step = t_vals[1] - t_vals[0]

animation_speed = 10

# Wave Params
w = 1    # Driving angular frequency
A = 20   # Driving amplitude

propagation_mode = "fea"  # Valid options: "exact", "fea"

## exact mode only
wave_number = 1  # exact mode only

## fea mode only
k = 2  # Spring constant (depending on simulation mode) Increases proportionatly to wavelength
B = 0.00005  # Damping constant
no_sponge = False

# Precalculate some stuff
n_x_vals = len(x_vals)
n_y_vals = len(y_vals)
y_step = y_vals[1] - y_vals[0]

# Boundary conditions
driving_range_size = 0.01
"""num_driving_points = max(1, driving_range_size // y_step)
driving_range_min = (n_y_vals - num_driving_points) // 2
driving_range_max = driving_range_min + num_driving_points"""

def sponge(x, y):
	if no_sponge:
		return False

	if (x-50)**2 + (y-50)**2 < 250:  # circle or radius 5
		return True

	# layer around the edge
	if 0 <= x <= 2:
		return True
	if n_x_vals-3 <= x <= n_x_vals-1:
		return True
	if 0 <= y <= 2:
		return True
	if n_y_vals-3 <= y <= n_y_vals-1:
		return True

def source(x, y):
	if 3 < x <= 5 and 20 <= y <= 25:
		return True
	#elif x <= 15 and 75 <= y <= 80:
		#return True
	
def outOfBounds(x, y):
	in_box = 0 <= x < n_x_vals and 0 <= y < n_y_vals

	if not in_box:
		return True
	return False

# end region

# Define possible simulators
def exact_solution(mat):
	max_y = max(y_vals)
	for n, t in enumerate(tqdm(t_vals)):
		for y_idx, y in enumerate(tqdm(y_vals, leave=False, delay=1)):
			for x_idx, x in enumerate(x_vals):
				r = np.sqrt(x**2 + (y - (max_y / 2))**2)
				# Should this have a 1/sqrt(k) or something similar?
				mat[n][y_idx][x_idx] = A*np.sin(wave_number*r - w*t) / (np.sqrt(r) or 1)


def fea(mat):

	# matrix[0=pos, 1=vel][y][x]
	working_size = (2, n_y_vals, n_x_vals)  # Size that we actually want use
	y_vec_size = (2*n_x_vals*n_y_vals,)     # Size fed to scipy (flattened version of ^)

	def ode_problem(t, v, progress_bar):
		progress_bar.update(round(t-progress_bar.n))

		vals = v.reshape(working_size)
		derivs = np.zeros(working_size)  # New array to store output

		# dx/dt = v
		derivs[0] = vals[1]

		# Compute driving force
		driving_pos = A * np.sin(w * t)
		driving_vel = A * w * np.cos(w * t)

		driving_info = (driving_pos, driving_vel)

		# Calculate acceleration for each point
		for y in range(n_y_vals):
			for x in range(n_x_vals):

				current_pos = vals[0][y][x]
				current_vel = vals[1][y][x]

				a = 0

				# For the position and velocity of each neighbor
				# (use current pos/vel if neighbor doesn't exist
				#  because then taking the difference -> 0)
				
				neighbors = [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]

				for nei_x, nei_y in neighbors:
					if source(nei_x, nei_y):
						pos, vel = driving_info
					elif not outOfBounds(nei_x, nei_y):
						pos, vel = (vals[0][nei_y][nei_x], vals[1][nei_y][nei_x])
					else:
						continue

					# Spring and damper!
					if sponge(x, y):
						a += 0 * (pos - current_pos) + B*1000 * (vel - current_vel)
					else:
						a += k * (pos - current_pos) + B * (vel - current_vel)

				# dv/dt = a
				derivs[1][y][x] = a

		return derivs.reshape(y_vec_size)

	with tqdm(total=max(t_vals), disable=False) as progress_bar:
		values = solve_ivp(
			ode_problem,
			[min(t_vals), max(t_vals)],
			np.zeros(y_vec_size),
			t_eval=t_vals,
			args=(progress_bar,)
		)

	# Copy the results to the output matrix
	for (n, frame) in enumerate(np.transpose(values.y)):
		mat[n] = frame.reshape(working_size)[0]

#endregion

#region Setup and Run

# Choose one and run the simulation
simulation_func = \
	exact_solution if propagation_mode == "exact" else \
	fea if propagation_mode == "fea" else \
	None

if simulation_func is None:
	raise ValueError(f"Invalid simulator: {propagation_mode}")

print("Running Simulation...")

data_matrix = np.zeros((len(t_vals), len(y_vals), len(x_vals)))
simulation_func(data_matrix)

print(f"Done! Computed amplitudes fall in the range [{np.min(data_matrix)}, {np.max(data_matrix)}]")

#endregion

#region Show Output

# Show the results!

fig, (wave_sim, sensor) = plt.subplots(2)
wave_sim.set_xlim(min(x_vals), max(x_vals))
wave_sim.set_ylim(min(y_vals), max(y_vals))

map_data = wave_sim.imshow(
	np.zeros((len(y_vals), len(x_vals))),
	vmin=np.min(data_matrix), vmax=np.max(data_matrix),
	extent=(min(x_vals), max(x_vals), min(y_vals), max(y_vals))
)

def frame(n):
	map_data.set_data(data_matrix[n])

	return (map_data,)

ani = FuncAnimation(fig, frame, range(len(t_vals)), interval=1000*t_step/animation_speed, blit=False)
plt.show()

#endregion
