import matplotlib.pyplot as plt
import numpy as np

# Data
eigenvalues = [1.0067475e-09, 8.6322069e-01, 8.7776899e-01, 9.5024055e-01, 9.7428906e-01,
               9.8527324e-01, 9.8771304e-01, 9.9123877e-01, 9.9199015e-01, 9.9341708e-01]
indices = np.arange(len(eigenvalues))

# Plot
fig, ax = plt.subplots()

# Remove the baseline by setting basefmt to ' ' (an empty string)
ax.stem(indices, eigenvalues, linefmt='green', markerfmt='go', basefmt=' ')

# Annotate the large gap with a red line
ax.plot([2, 3], [8.7776899e-01, 9.5024055e-01], color='red', lw=2)
ax.text(1.2, 0.94, 'Eigengap', fontsize=12, color='red')

# Labels and title
ax.set_xlabel('Index van eigenwaarde')
ax.set_ylabel('Waarde')
ax.set_title('')

# # Adding values on top of the stems
# for i, value in enumerate(eigenvalues):
#     ax.text(i, value + 0.02, f'{value:.7f}', ha='center', fontsize=8)

plt.show()
