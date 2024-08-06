import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Dataset names and accuracy results
dataset_names = [
    "CBF", 
    "ItalyPowerDemand", 
    "Mallat", 
    "StarLightCurves", 
    "Symbols", 
    "TwoPatterns"
]

accuracy_results = [
    0.900016981,
    0.596584225,
    0.793595901,
    0.546058715,
    0.75436762,
    0.942848779
]

# Creating a DataFrame
df = pd.DataFrame({
    "Dataset Name": dataset_names,
    "Accuraatheid KNN Update methode": accuracy_results
})

# Setting the aesthetic style of the plots
sns.set(style="whitegrid")

# Plotting the bar chart
plt.figure(figsize=(12, 7))

# Bar chart for update method with a smaller width
bars = plt.bar(df['Dataset Name'], df['Accuraatheid KNN Update methode'], width=0.3, color='skyblue', edgecolor='black')

# Adding titles and labels
plt.xlabel('Dataset', fontsize=14)
plt.ylabel('Gemiddelde ARI score', fontsize=14)
plt.xticks(rotation=60, fontsize=12, style='italic')
plt.yticks(fontsize=12)
plt.tight_layout()

# Adding value labels on the bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, round(yval, 2), ha='center', va='bottom', fontsize=10, color='black')

# Show the plot
plt.show()
