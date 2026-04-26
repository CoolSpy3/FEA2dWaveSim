
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots()
ax.set_xlim(0,100)
ax.set_ylim(0,100)

map_data = ax.imshow(np.zeros((100, 100)), vmin=-1, vmax=1)

def frame(n:int):
	grid = np.zeros((100, 100))

	for x in range(100):
		for y in range(100):
			r = np.sqrt(x**2 + (y - 50)**2)
			grid[y][x] = np.sin(r - n) / np.sqrt(r)

	map_data.set_data(grid)

	return (map_data,)

ani = FuncAnimation(fig, frame, range(1, 80, 2), interval=100, blit=False)
plt.show()
