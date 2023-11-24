import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# Example data
data = np.random.rand(100, 10)  # 100 frames, each with 10 data points

# Set up the initial plot
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.25)  # Make room for the slider
line, = ax.plot(data[0])  # Start with the first frame of data

# Create the slider
ax_slider = plt.axes([0.1, 0.05, 0.8, 0.05], facecolor='lightgoldenrodyellow')
slider = Slider(ax_slider, 'Frame', 0, len(data) - 1, valinit=0, valfmt='%0.0f')

# Update function for the slider


def update(val):
    frame = int(slider.val)
    line.set_ydata(data[frame])
    fig.canvas.draw_idle()


# Register the update function with the slider
slider.on_changed(update)

# Show the plot
plt.show()
