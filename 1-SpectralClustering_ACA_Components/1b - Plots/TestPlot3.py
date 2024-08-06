import matplotlib.pyplot as plt

# Data points for the ARI scores
x = [0.9034, 0.609585138, 0.8644, 0.64, 0.775, 0.9332]
y = [0.9154, 0.601585138, 0.8433, 0.64, 0.783, 0.9212]
labels = [
    "CBF", 
    "ItalyPowerDemand", 
    "Mallat", 
    "StarLightCurves", 
    "Symbols", 
    "TwoPatterns"
]

plt.figure(figsize=(10, 6))

# Plot the data points
plt.scatter(x, y, color='blue')

# Add labels to the points
for i, label in enumerate(labels):
    plt.text(x[i], y[i], label, fontsize=12, ha='right')

# Add the diagonal reference line
plt.plot([0, 1], [0, 1], 'r--')

# Add labels and title
plt.xlabel(r'$\mathrm{ARI}_{\mathrm{benadering}}$', fontsize=12)
plt.ylabel(r'$\mathrm{ARI}_{\mathrm{eigen}}$', fontsize=12)

# Set the limits of the axes
plt.xlim(0, 1)
plt.ylim(0, 1)

# Add grid
plt.grid(True)

# Display the plot
plt.show()
