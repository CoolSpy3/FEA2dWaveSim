#region Imports
from geometry import *
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
from odrpack import odr_fit
from scipy.fft import fft, ifft, fftfreq
from scipy.integrate import solve_ivp
from scipy.sparse import csr_matrix
from tqdm import tqdm

#endregion

#region Params

# Grid Params
x_origin = 0
y_origin = 0
sim_width = 10
sim_height = 10
n_x_vals = 100
n_y_vals = 100

# Animation Params
t_vals = np.linspace(0, 120, 500)
animation_speed = 10

# Wave Params
w = 1    # Driving angular frequency
A = 20   # Driving amplitude

propagation_mode = "fea"  # Valid options: "exact", "fea"

## exact mode only
wave_number = 1

## fea mode only
k = 2  # Spring constant (Increases proportionally to wavelength)
B = 0  # Damping constant (removed in latest model iteration)

# Absorptive Boundary Condition (drag sponge field)
sponge = True
sponge_thickness = 1.5 # VERY IMPORTANT. This needs to be uncomfrotably large unfortunatly. It needs to be larger for larger wavelengths
gamma_max = 2  # Experiment with

# Precalculate some stuff
x_vals = np.linspace(x_origin, x_origin + sim_width, n_x_vals)
y_vals = np.linspace(y_origin, y_origin + sim_height, n_y_vals)
x_step = abs(x_vals[1] - x_vals[0])
y_step = abs(y_vals[1] - y_vals[0])
t_step = abs(t_vals[1] - t_vals[0])

# Sensor params
sensor_x_idx = len(x_vals) // 4
sensor_y_idx = len(y_vals) // 2

# Place stuff on the grid!

# Format: [(Geom, Amplitude, Angular Freq)]
sources = [
	(Point(2, 5), A, w)
]

def sponge_func(border, x, y):
	# Max distance into sponge in one direction
	d_into_sponge = border.thickness - min(
		abs(x - border.outer_rect.x_min), abs(x - border.outer_rect.x_max),
		abs(y - border.outer_rect.y_min), abs(y - border.outer_rect.y_max)
		) 

	if 0 < d_into_sponge < border.thickness:
		return gamma_max * np.sin(np.pi/2 * d_into_sponge/border.thickness) ** 3
	else:
		return 0

# Format: [(Geom, < (geom,point)->sponge_factor > or < None > if hard boundary)]
obstacles = [
	# There must be a hard border around the simulation or else it'll try to calculate out of bounds points
	(
		Border(
			# Set it to be all the points just outside of the valid grid space, minus
			x_origin, y_origin,
			n_x_vals, n_y_vals,
			# thickness = 2 to give another cell of extra padding just-in-case
			2,
			False,  # Place around the grid; not in it
			True    # Above values are in grid-coordinates
		), None  # Hard boundary
	),
	# Conditional sponge border
	(
		Border(
			x_origin, y_origin,
			sim_width, sim_height,
			sponge_thickness
		), lambda border, x, y: sponge_func(border, x * x_step, y * y_step)
	) if sponge else None
]

# Allows us to use None as a null-obstacle.
while None in obstacles:
	obstacles.remove(None)

#endregion

#region Simulation

# Define possible simulators
# Note that many of these matrices are in [y, x] form because that's how imshow parses them

def exact_solution(mat):
	print("Running Simulation...")
	max_y = max(y_vals)
	for n, t in enumerate(tqdm(t_vals)):
		for y_idx, y in enumerate(tqdm(y_vals, leave=False, delay=1)):
			for x_idx, x in enumerate(x_vals):
				r = np.sqrt(x**2 + (y - (max_y / 2))**2)
				# Should this have a 1/sqrt(k) or something similar?
				# Doesn't really matter, this is just for a rough comparison
				mat[n][y_idx][x_idx] = A*np.sin(wave_number*r - w*t) / (np.sqrt(r) or 1)


def fea(mat):
	print("Preparing Grid...")

	# Format: matrix[0=pos, 1=vel][y][x]
	working_size = (2, n_y_vals, n_x_vals)  # Size that we actually want use

	y_vec_len = 2*n_x_vals*n_y_vals         # Length fed to/from scipy (flattened version of ^)

	# Why figure out how reshape works, when we can just reshape a [0, 1, 2, ...] vector to our preferred size?
	# Then, we can index it however we want and the value will be the correct index in the flattened vector!
	idx_lookup_table = np.arange(y_vec_len).reshape(working_size)
	# Define a shorthand for indexing this
	def idx(xv, y, x):
		return idx_lookup_table[xv,y,x]

	# Think of these as being indexed in the form [to, from] such that
	# the value at <to> in the source vector will be scaled by mat[to, from]
	# and placed at <from> in the result vector
	deriv_mat = np.zeros((y_vec_len, y_vec_len))      # Affects from cells in the grid
	source_mat = np.zeros((y_vec_len, len(sources)))  # Affects from sources

	source_freqs = np.array([w for _, _, w in sources])

	# Now, populate the matrices for all x,y pairs!
	for y in tqdm(range(n_y_vals)):
		for x in range(n_x_vals):
			# Cache the index of (x, y) in the scipy vector
			pos_loc_idx = idx(0, y, x)
			vel_loc_idx = idx(1, y, x)

			# dx/dt = 1*v
			deriv_mat[pos_loc_idx, vel_loc_idx] = 1

			# dv/dt is sum of effects from neighbors (and sponges)
			neighbors = [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]
			for nei_x, nei_y in neighbors:
				is_source = False
				for source_idx, (source_geom, A, _) in enumerate(sources):
					if source_geom.contains_raw_point(nei_x, nei_y, x_step, y_step):
						is_source = True
						# Include source <source_idx> with amplitude A
						# Will be scaled by (k * source_pos + B * source_vel)
						source_mat[vel_loc_idx, source_idx] = A
						# Subtract k * pos + B * vel so that the overall algebra becomes
						# k * (source_pos - pos) + B * (source_vel - vel)
						deriv_mat[vel_loc_idx, pos_loc_idx] -= k
						deriv_mat[vel_loc_idx, vel_loc_idx] -= B

				# If this point is a source, it's values are determined completely by the source behavior.
				# Whatever values the simulation calculates for it must be discarded!
				if is_source:
					continue

				for (obstacle_geom, sponge_behavior) in obstacles:
					if obstacle_geom.contains_raw_point(nei_x, nei_y, x_step, y_step) and \
							sponge_behavior is None:  # Hard boundary
						break  # Causes else to be skipped
				else:  # For-else: If for loop not broken (and, thus, not at a boundary),
					# Apply neighboring effects (see source math).
					# This is the same, but all the data's already in the state vec,
					# so we can do this with just one matrix
					deriv_mat[vel_loc_idx, idx(0, nei_y, nei_x)] += k
					deriv_mat[vel_loc_idx, pos_loc_idx] -= k
					deriv_mat[vel_loc_idx, idx(1, nei_y, nei_x)] += B
					deriv_mat[vel_loc_idx, vel_loc_idx] -= B

			# Sponges create drag at a point, regardless of neighboring behavior
			for (obstacle_geom, sponge_behavior) in obstacles:
				if obstacle_geom.contains_raw_point(nei_x, nei_y, x_step, y_step) and \
						sponge_behavior is not None:  # Sponge boundary
					# dv/dt += sponge_behavior * (-v)
					deriv_mat[vel_loc_idx, vel_loc_idx] -= sponge_behavior(obstacle_geom, x, y)

	print("Computing efficient representation...")

	# Most of the values in these matrices are 0 because every point is only
	# influenced by its neighbors (And every point only has a few of those!)
	# Thus, we can convert these to sparse csr matrices for speedy matrix-vector multiplications
	deriv_mat = csr_matrix(deriv_mat)
	source_mat = csr_matrix(source_mat)

	print("Running simulation...")

	# Now that we've computed all that, the actual evolution can be expressed fairly simply
	# (And done quickly by native numpy routines)
	def ode_problem(t, v, progress_bar):
		# Neat trick to get tqdm to show an approximation of how much time we've simulated.
		progress_bar.update(round(t, 3)-progress_bar.n)

		# Compute driving forces (amplitudes are already stored in source_mat)
		source_positions = np.sin(source_freqs * t)
		# Derivative of ^
		source_velocities = source_freqs * np.cos(source_freqs * t)  # Here * performs an element-wise product, so this is correct

		# Multiply the right things to make the math discussed above work and then add all the effects together
		return deriv_mat.dot(v) + source_mat.dot(k * source_positions + B * source_velocities)

	# Alright, we're all set!
	# I henceforth call unto the deep magic!
	# By the Numpy imported by me (as np),
	# Scipy, I hear-by summon ye to fulfill your oath
	# as defined in your reference documentation!
	# Grant my request to peak beyond the veil of time,
	# and show unto me how the ODE will unfold at t_vals!
	with tqdm(total=max(t_vals)) as progress_bar:
		values = solve_ivp(
			ode_problem,
			[min(t_vals), max(t_vals)],
			np.zeros(y_vec_len),
			t_eval=t_vals,
			args=(progress_bar,)
		)

	# Copy the results to the output matrix
	for (n, frame) in enumerate(np.transpose(values.y)):
		# We only care about the positions, so discard the velocity matrix
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

data_matrix = np.zeros((len(t_vals), len(y_vals), len(x_vals)))
simulation_func(data_matrix)

print(f"Done! Computed amplitudes fall in the range [{np.min(data_matrix)}, {np.max(data_matrix)}]")

#endregion

#region Show Output

# Display the results!

fig, (wave_sim, sensor) = plt.subplots(2)
wave_sim.set_xlim(min(x_vals), max(x_vals))
wave_sim.set_ylim(min(y_vals), max(y_vals))

map_data = wave_sim.imshow(
	np.zeros((len(y_vals), len(x_vals))),
	vmin=np.min(data_matrix), vmax=np.max(data_matrix),
	extent=(min(x_vals), max(x_vals), min(y_vals), max(y_vals))
)

# WIP Sensor code. TODO: Document

sensor_data = data_matrix[:,sensor_y_idx,sensor_x_idx]
sensor.plot(t_vals, sensor_data)

# clip_pane_start = [v > A/11 for v in sensor_data].index(True)
clip_pane_start = 0
clipped_sensor_data = sensor_data[clip_pane_start:]
sensor.axvline(t_vals[clip_pane_start])
current_time = sensor.axvline(0)

def fit_func(x, p):
	a, b, c, d, e, f = p
	# return a * np.exp(b * x) * np.sin(c * x + d)
	return a * np.sin(b * x + c) + d * np.sin(e * x + f)

sensor_data_clipped = (t_vals[clip_pane_start:], sensor_data[clip_pane_start:])
fit = odr_fit(fit_func, *sensor_data_clipped, [1, 1, 1, 1, 2, 1])

print(fit.beta)

sensor.plot(sensor_data_clipped[0], fit_func(sensor_data_clipped[0], fit.beta))

# Now, animate it!

def frame(n):
	map_data.set_data(data_matrix[n])
	current_time.set_xdata([n*t_step]*2)

	return (map_data, current_time)

ani = FuncAnimation(fig, frame, range(len(t_vals)), interval=1000*t_step/animation_speed, blit=False)
plt.show()

#endregion
