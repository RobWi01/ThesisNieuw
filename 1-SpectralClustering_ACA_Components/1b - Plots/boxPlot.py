import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Updated data
data_updated = {
    "Dataset": [
        "CBF", "TwoPatterns", "Mallat", "ItalyPowerDemand", "Symbols",
        "StarLightCurves"
    ],
    "Average ARI Score": [
        0.785848796, 0.865953838, 0.786994835, 0.427783759, 0.422744157,
        0.765352661, 0.534365337, 0.339923917, 0.345507302
    ],
    "Standard Deviation": [
        0.049271619, 0.043303819, 0.034906805, 0.039010159, 0.046072285,
        0.060137681, 0.028732833, 0.032662462, 0.036303698
    ],
    "IQR Score": [
        0.06737334, 0.04552378, 0.053320175, 0.049869366, 0.06188584,
        0.047012568, 0.010984291, 0.041162562, 0.054421589
    ]
}

df_updated = pd.DataFrame(data_updated)

# Generating synthetic data based on the updated values for standard deviation and IQR
data_for_boxplot_updated = [
    np.random.normal(loc=score, scale=iqr/4, size=100) for score, iqr in zip(df_updated['Average ARI Score'], df_updated['IQR Score'])
]
data_std_dev_updated = [
    np.random.normal(loc=score, scale=std, size=100) for score, std in zip(df_updated['Average ARI Score'], df_updated['Standard Deviation'])
]

# Plotting without outliers using the updated data
fig, ax = plt.subplots(figsize=(14, 8))

# IQR Box Plots without outliers
box_iqr_updated = ax.boxplot(data_for_boxplot_updated, positions=np.array(range(len(df_updated['Dataset'])))*2.0-0.4, widths=0.6, patch_artist=True, showfliers=False, boxprops=dict(facecolor="yellow", color="yellow"), medianprops=dict(color="red", linewidth=2))
# Std Dev Box Plots without outliers
box_std_dev_updated = ax.boxplot(data_std_dev_updated, positions=np.array(range(len(df_updated['Dataset'])))*2.0+0.4, widths=0.6, patch_artist=True, showfliers=False, boxprops=dict(facecolor="blue", color="blue"), medianprops=dict(color="black", linewidth=2))

ax.set_xticks(range(0, len(df_updated['Dataset'])*2, 2))
ax.set_xticklabels(df_updated['Dataset'])
ax.set_xlabel('Dataset')
ax.set_ylabel('ARI Score')
ax.set_title('Quality of Clustering Using Spectral Clustering ACA Components with Statistical Measures')
ax.legend([box_iqr_updated["boxes"][0], box_std_dev_updated["boxes"][0]], ['IQR', 'Standard Deviation'], loc='upper right')

plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()
