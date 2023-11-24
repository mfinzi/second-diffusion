import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Example NumPy array (replace with your data)
data = np.random.rand(10, 10)  # 100 frames, each with 10 data points

# Set up the initial plot
fig, ax = plt.subplots()
line, = ax.plot(data[0])  # Start with first frame of data

# Define the update function


def update(frame):
    line.set_ydata(data[frame])  # Update the data for the next frame
    return line,


# Create the animation
ani = FuncAnimation(fig, update, frames=range(len(data)), blit=True)

# To display the animation
plt.show()

# To save the animation as a GIF or video
# ani.save('animation.gif', writer='imagemagick')
# ani.save('animation.mp4', writer='ffmpeg')
