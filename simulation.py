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
wave_number = 1  # exact mode only

## fea mode only
k = 2  # Spring constant (depending on simulation mode) Increases proportionatly to wavelength
B = 0  # Damping constant (removed in latest model iteration)

# Absorptive Boundary Condition (drag sponge field)
sponge = True
sponge_thickness = 0.5
gamma_max = 2  # has to be less than 2

# Precalculate some stuff
x_vals = np.linspace(x_origin, x_origin + sim_width, n_x_vals)
y_vals = np.linspace(y_origin, y_origin + sim_height, n_y_vals)
x_step = abs(x_vals[1] - x_vals[0])
y_step = abs(y_vals[1] - y_vals[0])
t_step = abs(t_vals[1] - t_vals[0])

# [(Geom, Amplitude, Angular Freq)]
sources = [
	(Point(2, 5, atol=max(x_step, y_step)), A, w)
]

def sponge_func(dist, max_dist):
	return gamma_max * (np.sin((np.pi/2) * (dist/max_dist)) ** 2)

# [(Geom, (geom,point)->sponge factor or None if hard bound)]
null_obstacle = (Point(-10000, -10000), None)  # Because I'm too lazy to handle this properly
obstacles = [
	# There must be a hard border around the simulation or else we'll try to calculate out of bounds points
	(
		Border(
			x_origin, y_origin,
			n_x_vals, n_y_vals,
			2, False, True  # Over-sized outside border
		), None
	),
	# Conditional sponge
	(
		Border(
			x_origin, y_origin,
			sim_width, sim_height,
			sponge_thickness
		), lambda border, x, y: sponge_func(
			border.outer_rect.dist_to_border(x*x_step, y*y_step),
			border.thickness * np.sqrt(2)  # Because the corners extend a distance of sqrt(2)*<thickness>
		)
	) if sponge else null_obstacle
]

# Sensor params
sensor_x_idx = len(x_vals) // 4
sensor_y_idx = len(y_vals) // 2

#endregion

#region Simulation

# Define possible simulators
def exact_solution(mat):
	print("Running Simulation...")
	max_y = max(y_vals)
	for n, t in enumerate(tqdm(t_vals)):
		for y_idx, y in enumerate(tqdm(y_vals, leave=False, delay=1)):
			for x_idx, x in enumerate(x_vals):
				r = np.sqrt(x**2 + (y - (max_y / 2))**2)
				# Should this have a 1/sqrt(k) or something similar?
				mat[n][y_idx][x_idx] = A*np.sin(wave_number*r - w*t) / (np.sqrt(r) or 1)


def fea(mat):
	print("Preparing Grid...")

	# matrix[0=pos, 1=vel][y][x]
	working_size = (2, n_y_vals, n_x_vals)  # Size that we actually want use
	y_vec_len = 2*n_x_vals*n_y_vals         # Length fed to scipy (flattened version of ^)

	idx_lookup_table = np.arange(y_vec_len).reshape(working_size)
	def idx(xv, y, x):
		return idx_lookup_table[xv,y,x]

	# [to, from]
	deriv_mat = np.zeros((y_vec_len, y_vec_len))
	source_mat = np.zeros((y_vec_len, len(sources)))

	source_freqs = np.array([w for _, _, w in sources])

	for y in tqdm(range(n_y_vals)):
		for x in range(n_x_vals):
			pos_loc_idx = idx(0, y, x)
			vel_loc_idx = idx(1, y, x)
			deriv_mat[pos_loc_idx, vel_loc_idx] = 1

			neighbors = [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]

			for nei_x, nei_y in neighbors:
				is_source = False
				for source_idx, (source_geom, A, _) in enumerate(sources):
					if source_geom.contains_raw_point(nei_x, nei_y, x_step, y_step):
						is_source = True
						source_mat[vel_loc_idx, source_idx] = A
						deriv_mat[vel_loc_idx, pos_loc_idx] -= k
						deriv_mat[vel_loc_idx, vel_loc_idx] -= B

				if is_source:
					continue

				is_out_of_bounds = False
				for (obstacle_geom, sponge_behavior) in obstacles:
					if obstacle_geom.contains_raw_point(nei_x, nei_y, x_step, y_step) and \
							sponge_behavior is None:  # Hard boundary
						is_out_of_bounds = True
						break

				if not is_out_of_bounds:
					deriv_mat[vel_loc_idx, idx(0, nei_y, nei_x)] += k
					deriv_mat[vel_loc_idx, pos_loc_idx] -= k
					deriv_mat[vel_loc_idx, idx(1, nei_y, nei_x)] += B
					deriv_mat[vel_loc_idx, vel_loc_idx] -= B

			# Drag into the sponge
			for (obstacle_geom, sponge_behavior) in obstacles:
				if obstacle_geom.contains_raw_point(nei_x, nei_y, x_step, y_step) and \
						sponge_behavior is not None:  # Sponge boundary
					deriv_mat[vel_loc_idx, vel_loc_idx] -= sponge_behavior(obstacle_geom, x, y)

	print("Computing efficient representation...")

	# Convert to csr matrix for speedy matrix-vector multiplications
	deriv_mat = csr_matrix(deriv_mat)
	source_mat = csr_matrix(source_mat)

	print("Running simulation...")

	def ode_problem(t, v, progress_bar):
		progress_bar.update(round(t, 3)-progress_bar.n)

		# Compute driving forces
		source_positions = np.sin(source_freqs * t)
		source_velocities = source_freqs * np.cos(source_freqs * t)  # All of this just broadcasts correctly *trust*

		return deriv_mat.dot(v) + source_mat.dot(k * source_positions + B * source_velocities)

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

# Show the results!

fig, (wave_sim, sensor) = plt.subplots(2)
wave_sim.set_xlim(min(x_vals), max(x_vals))
wave_sim.set_ylim(min(y_vals), max(y_vals))

map_data = wave_sim.imshow(
	np.zeros((len(y_vals), len(x_vals))),
	vmin=np.min(data_matrix), vmax=np.max(data_matrix),
	extent=(min(x_vals), max(x_vals), min(y_vals), max(y_vals))
)

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

def frame(n):
	map_data.set_data(data_matrix[n])
	current_time.set_xdata([n*t_step]*2)

	return (map_data,)

ani = FuncAnimation(fig, frame, range(len(t_vals)), interval=1000*t_step/animation_speed, blit=False)
plt.show()

#endregion
