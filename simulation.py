
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots()
ax.set_xlim(0,100)
ax.set_ylim(0,100)

map_data = ax.imshow(np.zeros((100, 100)))

def frame(n:int):
	grid = np.zeros((100, 100))

	for x in range(100):
		for y in range(100):
			grid[x][y] = np.sin(np.sqrt((x**2 + y**2)) + n)

	map_data.set_data(grid)

	return map_data

ani = FuncAnimation(fig, frame, range(1, 80, 2), interval=100, blit=False)
plt.show()
