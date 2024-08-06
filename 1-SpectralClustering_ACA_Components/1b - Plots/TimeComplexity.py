import matplotlib.pyplot as plt
import numpy as np

# Given data points
sizes = np.array([500, 1000, 5000, 10000, 20000, 24000, 40000, 60000])
times = np.array([0.4185938835144043, 0.743741512298584, 2.327543258666992, 5.571155548095703, 10.51760482788086, 12.9204, 22.9204, 37.3806])

# Plotting the actual times
plt.figure(figsize=(10, 6))
plt.plot(sizes, times, 'o-', color='blue', label='Algoritme tijd', linewidth=2, markersize=8)

# Enhance the plot with a title and labels
plt.xlabel('Aantal tijdreeksen (n)', fontsize=14)
plt.ylabel('Tijd (seconde)', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)

# Set the x and y-axis limits for better visualization
plt.xlim(0, 60000)
plt.ylim(0, max(times) * 1.1)  # Give a bit more space on y-axis

# Save the plot as a high-resolution PNG file
plt.savefig('lanczos_time_complexity_professional.png', dpi=300)

# Show the plot
plt.show()
