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
                     "PhalangesOutlinesCorrect", "Plane", "PowerCons", "ProximalPhalanxOutlineAgeGroup", "ProximalPhalanxOutlineCorrect", 
                     "ProximalPhalanxTW", "SemgHandGenderCh2", "SemgHandMovementCh2", "SemgHandSubjectCh2", "ShapesAll", 
                     "SmallKitchenAppliances", "SmoothSubspace", "SonyAIBORobotSurface2", "StarLightCurves", "Strawberry", "SwedishLeaf", 
                     "Symbols", "SyntheticControl", "ToeSegmentation1", "ToeSegmentation2", "Trace", "TwoLeadECG", "TwoPatterns", "Wafer", 
                     "Yoga"],
    "Accuraatheid KNN naïeve methode": [0.4, 0.45, 0.36, 0.216666667, 0.166666667, 0.259475219, 0.178571429, 0.023076923, 0.025641026, 
                                        0.386231884, 0.013071895, 0.208633094, 0.416666667, 0, 0.544, 0.748201439, 0.5, 0.046444444, 
                                        0.013609467, 0.031707317, 0.074285714, 0.516666667, 0, 0.368070175, 0.497894737, 0.493333333, 
                                        0.525316456, 0.250793651, 0.490506329, 0.359459459, 0.512605042, 0, 0, 0.45675413, 0.336, 
                                        0.524590164, 0, 0.066666667, 0.317105263, 0.570446735, 0, 0.094845361, 0.056494845, 0.361022364, 
                                        0.029516539, 0.021374046, 0.066666667, 0.386946387, 0, 0.377777778, 0.487804878, 0.350515464, 
                                        0.048780488, 0.643333333, 0.16, 0.191111111, 0.023333333, 0.336, 0.08, 0.312696747, 0.002064109, 
                                        0.643243243, 0.0176, 0.001005025, 0, 0.495614035, 0.238461538, 0, 0.500438982, 0.03425, 0.971771577, 
                                        0.562333333],
    "Accuraatheid KNN exacte methode (benchmark)": [0.75, 0.75, 0.893333333, 0.733333333, 1, 0.973760933, 1, 0.753846154, 0.769230769, 
                                                    0.650724638, 0.967320261, 0.769784173, 0.75, 0.64028777, 0.74, 0.769784173, 0.8, 
                                                    0.940444444, 0.827218935, 0.904878049, 0.822857143, 0.593939394, 0.795698925, 0.898947368, 
                                                    0.767368421, 0.906666667, 0.984177215, 1, 0.993670886, 0.9, 0.882352941, 1, 1, 0.952380952, 
                                                    0.8, 0.868852459, 0.933901919, 0.933333333, 0.736842105, 0.742268041, 0.545454545, 
                                                    0.841649485, 0.779793814, 0.834664537, 0.800508906, 0.865139949, 0.833333333, 0.77039627, 
                                                    1, 0.922222222, 0.829268293, 0.83161512, 0.795121951, 0.928333333, 0.786666667, 0.873333333, 
                                                    0.768333333, 0.696, 0.886666667, 0.831059811, 0.914400194, 0.940540541, 0.792, 0.949748744, 
                                                    0.993333333, 0.771929825, 0.846153846, 1, 0.904302019, 1, 0.979883193, 0.836333333]
}

# Creating a DataFrame
df = pd.DataFrame(data)

# Setting the aesthetic style of the plots
sns.set(style="whitegrid")

# Plotting the bar chart
plt.figure(figsize=(16, 9))

# Select a subset of datasets for better visibility
selected_datasets = df.head(10)  # Selecting first 10 datasets for demonstration

bar_width = 0.35
index = range(len(selected_datasets))

# Bar chart for naïve method with a lighter blue
bars1 = plt.bar(index, selected_datasets['Accuraatheid KNN naïeve methode'], bar_width, label='Naïeve Methode', color='skyblue', edgecolor='black')

# Bar chart for exact method with a darker blue
bars2 = plt.bar([i + bar_width for i in index], selected_datasets['Accuraatheid KNN exacte methode (benchmark)'], bar_width, label='Exacte Methode (Benchmark)', color='royalblue', edgecolor='black')

# Adding titles and labels
plt.xlabel('Dataset Name', fontsize=14)
plt.ylabel('Accuraatheid', fontsize=14)
plt.xticks([i + bar_width/2 for i in index], selected_datasets['Dataset Name'], rotation=60, fontsize=12)
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
