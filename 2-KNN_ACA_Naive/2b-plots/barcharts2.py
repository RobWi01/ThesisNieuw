import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Dataset from the provided image
data = {
    "Dataset Name": ["BeetleFly", "BirdChicken", "BME", "Car", "CBF", "Chinatown", "Coffee", "CricketX", "CricketZ", "CinCECGTorso",
                     "DiatomSizeReduction", "DistalPhalanxOutlineAgeGroup", "DistalPhalanxOutlineCorrect", "DistalPhalanxTW", "Computers",
                     "Earthquakes", "ECG200", "ECG5000", "FaceAll", "FacesUCR", "Fish", "FordA", "Fungi", "FreezerRegularTrain",
                     "FreezerSmallTrain", "GunPoint", "GunPointMaleVersusFemale", "GunPointOldVersusYoung", "GunPointAgeSpan", "HandOutlines",
                     "HouseTwenty", "InsectEPGSmallTrain", "InsectEPGRegularTrain", "ItalyPowerDemand", "LargeKitchenAppliances", "Lightning2",
                     "Mallat", "Meat", "MedicalImages", "MiddlePhalanxOutlineCorrect", "MiddlePhalanxTW", "MixedShapesRegularTrain",
                     "MixedShapesSmallTrain", "MoteStrain", "NonInvasiveFetalECGThorax1", "NonInvasiveFetalECGThorax2", "OliveOil",
                     "PhalangesOutlinesCorrect", "Plane", "PowerCons", "ProximalPhalanxOutlineAgeGroup"],
    "Accuraatheid KNN Update methode": [0.75, 0.75, 0.8933333333333333, 0.5666666666666667, 1, 0.9737609329446064, 1, 0.4230769230769231, 0.5,
                                        0.6420289855072464, 0.9673202614379085, 0.7050359712230215, 0.7572463768115942, 0.6330935251798561, 0.608,
                                        0.762589928, 0.81, 0.9277777777777778, 0.5497041420118344, 0.7073170731707317, 0.46285714285714286,
                                        0.5772727272727273, 0.7580645161290323, 0.8659649122807017, 0.763859649, 0.8333333333333334, 0.8291139240506329,
                                        1, 0.9715189873417721, 0.882352941, 1, 1, 1, 0.9494655, 0.714666667, 0.885245902, 0.935181237, 0.933333333,
                                        0.639473684, 0.694158076, 0.454545455, 0.694020619, 0.704329897, 0.837859425, 0.140966921, 0.444274809,
                                        0.833333333, 0.656177156, 1, 0.627777778, 0.863414634],
    "Accuraatheid KNN exacte methode (benchmark)": [0.75, 0.75, 0.893333333, 0.733333333, 1, 0.973760933, 1, 0.753846154, 0.769230769,
                                                    0.650724638, 0.967320261, 0.769784173, 0.75, 0.64028777, 0.74, 0.769784173, 0.8,
                                                    0.940444444, 0.827218935, 0.904878049, 0.822857143, 0.593939394, 0.795698925, 0.898947368,
                                                    0.767368421, 0.906666667, 0.984177215, 1, 0.993670886, 0.882352941, 1, 1, 1, 0.952380952,
                                                    0.8, 0.868852459, 0.933901919, 0.933333333, 0.736842105, 0.742268041, 0.545454545,
                                                    0.841649485, 0.779793814, 0.834664537, 0.800508906, 0.865139949, 0.833333333, 0.77039627,
                                                    1, 0.922222222, 0.829268293]
}

# Creating a DataFrame
df = pd.DataFrame(data)

# Setting the aesthetic style of the plots
sns.set(style="whitegrid")

# Plotting the bar chart
plt.figure(figsize=(16, 9))

# Select a subset of datasets for better visibility
selected_datasets = df.sort_values(by='Accuraatheid KNN Update methode', ascending=False).head(10)  # Selecting top 10 datasets

bar_width = 0.35
index = range(len(selected_datasets))

# Bar chart for update method with a lighter blue
bars1 = plt.bar(index, selected_datasets['Accuraatheid KNN Update methode'], bar_width, label='Update Methode', color='skyblue', edgecolor='black')

# Bar chart for exact method with a darker blue
bars2 = plt.bar([i + bar_width for i in index], selected_datasets['Accuraatheid KNN exacte methode (benchmark)'], bar_width, label='Exacte Methode (Benchmark)', color='royalblue', edgecolor='black')

# Adding titles and labels
plt.xlabel('Dataset Name', fontsize=14)
plt.ylabel('Accuraatheid', fontsize=14)
plt.xticks([i + bar_width/2 for i in index], selected_datasets['Dataset Name'], rotation=60, fontsize=12, style='italic')
plt.yticks(fontsize=12)
plt.legend(fontsize=12)
plt.tight_layout()

# Adding value labels on the bars
for bar in bars1:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, round(yval, 2), ha='center', va='bottom', fontsize=10, color='black')

for bar in bars2:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, round(yval, 2), ha='center', va='bottom', fontsize=10, color='black')

# Show the plot
plt.show()
