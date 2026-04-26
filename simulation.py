
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
t_vals = np.linspace(0, 120, 1000)
t_step = t_vals[1] - t_vals[0]

animation_speed = 2

# Wave Params
w = 1    # Driving angular frequency
A = 20   # Driving amplitude

## exact mode only
wave_number = 1  # exact mode only

## fea mode only
k = 0.5  # Spring constant (depending on simulation mode)
B = 0.1  # Damping constant
driving_range_size = 0.1

propagation_mode = "fea"  # Valid options: "exact", "fea"

#endregion

#region Simulators

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
	# Precalculate some stuff
	n_x_vals = len(x_vals)
	n_y_vals = len(y_vals)
	y_step = y_vals[1] - y_vals[0]
	num_driving_points = max(1, driving_range_size // y_step)
	driving_range_min = (n_y_vals - num_driving_points) // 2
	driving_range_max = driving_range_min + num_driving_points

	# matrix[0=pos, 1=vel][y][x]
	working_size = (2, n_y_vals, n_x_vals)  # Size that we actually want use
	y_vec_size = (2*n_x_vals*n_y_vals,)     # Size fed to scipy (flattened version of ^)

	def ode_problem(t, v, progress_bar):
		progress_bar.update(t-progress_bar.n)

		vals = v.reshape(working_size)
		derivs = np.zeros(working_size)  # New array to store output

		# dx/dt = v
		derivs[0] = vals[1]

		# Compute driving force
		driving_pos = A * np.sin(w * t)
		driving_vel = A * w * np.cos(w * t)

		# Calculate acceleration for each point
		for y in range(n_y_vals):
			for x in range(n_x_vals):
				current_pos = vals[0][y][x]
				current_vel = vals[1][y][x]

				a = 0

				# For the position and velocity of each neighbor
				# (use current pos/vel if neighbor doesn't exist
				#  because then taking the difference -> 0)
				for (pos, vel) in (
					(
						vals[0][y][x-1] if x > 0 else (driving_pos if driving_range_min <= y <= driving_range_max else current_pos),
						vals[1][y][x-1] if x > 0 else (driving_vel if driving_range_min <= y <= driving_range_max else current_vel),
	  				),
					(
						vals[0][y][x+1] if x < n_x_vals-1 else current_pos,
						vals[1][y][x+1] if x < n_x_vals-1 else current_vel,
	  				),
					(
						vals[0][y-1][x] if y > 0 else current_pos,
						vals[1][y-1][x] if y > 0 else current_vel,
	  				),
					(
						vals[0][y+1][x] if y < n_y_vals-1 else current_pos,
						vals[1][y+1][x] if y < n_y_vals-1 else current_vel,
	  				)
				):
					# Spring and damper!
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

# Initialize figure and output matrix
fig, ax = plt.subplots()
ax.set_xlim(min(x_vals), max(x_vals))
ax.set_ylim(min(y_vals), max(y_vals))

data_matrix = np.zeros((len(t_vals), len(y_vals), len(x_vals)))

# Choose one and run the simulation
simulation_func = \
	exact_solution if propagation_mode == "exact" else \
	fea if propagation_mode == "fea" else \
	None

if simulation_func is None:
	raise ValueError(f"Invalid simulator: {propagation_mode}")

print("Running Simulation...")

simulation_func(data_matrix)

print(f"Done! Computed amplitudes fall in the range [{np.min(data_matrix)}, {np.max(data_matrix)}]")

#endregion

#region Show Output

# Show the results!
map_data = ax.imshow(
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
