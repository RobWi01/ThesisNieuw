import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Data based on provided image
data = {
    "Dataset": ["CBF"] * 20 + ["Coffee"] * 20 + ["Fungi"] * 20 + ["GunPointOldVersusYoung"] * 20 + ["HouseTwenty"] * 20,
    "kmax": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200] * 5,
    "Average Score Recon": [0.852740601, 0.885512526, 0.881081752, 0.888845885, 0.886955804, 0.886535155, 0.89000436, 0.901020387, 0.898322567, 0.894771294, 0.893319367, 0.893341617, 0.896493543, 0.895989423, 0.895645525, 0.889467012, 0.893823393, 0.890481679, 0.894806352, 0.894154133,
                          0.793515548, 0.812903961, 0.803078993, 0.824936726, 0.830909532, 0.811431885, 0.821906851, 0.786723565, 0.810608987, 0.803167617, 0.804547155, 0.800872433, 0.81290499, 0.789574569, 0.777720884, 0.793431884, 0.798756122, 0.815200173, 0.802429204, 0.786723565,
                          0.331179485, 0.741825521, 0.727808063, 0.723862886, 0.719491502, 0.710445036, 0.708810325, 0.701414374, 0.697936345, 0.689458686, 0.701771251, 0.70827427, 0.697830377, 0.696928539, 0.691762008, 0.696087373, 0.688827703, 0.701256943, 0.682520799, 0.698611028,
                          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                          0.497290929, 0.559087657, 0.60145002, 0.603892319, 0.609804167, 0.618392285, 0.621725124, 0.623720405, 0.626374124, 0.621682055, 0.627690043, 0.625683733, 0.622339883, 0.630365124, 0.633708974, 0.635715284, 0.635715284, 0.635715284, 0.635715284, 0.635715284],
    "Std Deviation Recon": [0.039636457, 0.021529352, 0.022158766, 0.01981702, 0.019996985, 0.015969775, 0.023429202, 0.018600068, 0.015507997, 0.022416552, 0.020689271, 0.017314743, 0.020439446, 0.01943952, 0.019632101, 0.021799732, 0.02081442, 0.023043306, 0.022319893, 0.017187542,
                           2.22045E-16, 0.077106412, 0.093584624, 0.054332476, 0.085758403, 0.04236294, 0.082042816, 0.105874392, 0.074614556, 0.095248188, 0.024667424, 0.093002193, 0.077107223, 0.0837283, 0.098946059, 0.109587843, 0.09405316, 0.079455015, 0.028643264, 0.105874392,
                           0.072685369, 0.047693053, 0.034770698, 0.038025817, 0.03842518, 0.039042489, 0.04548448, 0.038492378, 0.034892154, 0.044784645, 0.033210428, 0.038108459, 0.041886323, 0.052501449, 0.043460642, 0.038103384, 0.038853097, 0.043135682, 0.0356541, 0.039429783,
                           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                           0.105547638, 0.05946331, 0.027537245, 0.017228746, 0.016320388, 0.014260321, 0.014694371, 0.014196653, 0.01235062, 0.010526469, 0.009828873, 0.010031551, 0.009457837, 0.008872238, 0.006018931, 0, 0, 0, 0, 0]
}

# Creating a DataFrame
df = pd.DataFrame(data)

# Set the aesthetic style of the plots
sns.set(style="whitegrid")

# Plotting ARI scores for different datasets with varying percentage of sampled data
plt.figure(figsize=(16, 9))
sns.lineplot(data=df, x="kmax", y="Average Score Recon", hue="Dataset", marker="o")
plt.xlabel('kmax', fontsize=14)
plt.ylabel('ARI Score', fontsize=14)
plt.title('ARI Scores voor verschillende datasets met variërend percentage gesamplede data', fontsize=16)
plt.legend(title='Dataset', fontsize=12)
plt.tight_layout()
plt.show()

# Plotting standard deviation of the scores for different values of kmax
plt.figure(figsize=(16, 9))
sns.lineplot(data=df, x="kmax", y="Std Deviation Recon", hue="Dataset", marker="o")
plt.xlabel('kmax', fontsize=14)
plt.ylabel('Standaardafwijking', fontsize=14)
plt.title('Standaardafwijking van de scores voor verschillende waarden van kmax', fontsize=16)
plt.legend(title='Dataset', fontsize=12)
plt.tight_layout()
plt.show()

# Sample pivot method data
pivot_data = {
    "Pivot Method": ["Method A"] * 20 + ["Method B"] * 20,
    "kmax": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200] * 2,
    "Average Score": [0.85, 0.86, 0.87, 0.88, 0.89, 0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1.00, 1.01, 1.02, 1.03, 1.04,
                      0.80, 0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.88, 0.89, 0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]
}

pivot_df = pd.DataFrame(pivot_data)

# Plotting comparison of different pivot methods on clustering quality
plt.figure(figsize=(16, 9))
sns.lineplot(data=pivot_df, x="kmax", y="Average Score", hue="Pivot Method", marker="o")
