import sys
import os

# Path where the module should be located
# Note that this build is also architecture depedent
# This path is also machine dependent 
module_path = 'C:\\Users\\robwi\\Documents\\ThesisFinal\\1-SpectralClustering_ACA_Components\\build\\Debug'
sys.path.append(module_path)

# print("Checking directory:", module_path)
# print("Contents:", os.listdir(module_path))
# print("Current sys.path:", sys.path)

try:
    import fastMultModule
    print("Module imported successfully!")
except ImportError as e:
    print("Failed to import module:", e)