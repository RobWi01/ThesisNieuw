#!/usr/bin/env python3

"""
This script copies the content of the NPY file
specified by data_name in the clipboard in a CSV.
This is intended to be pasted into a spreadsheet program.
"""

import sys
import io
import os
import numpy as np
import pyperclip
import csv

print("Python version:", sys.version)
print("Python interpreter:", sys.executable)

if len(sys.argv) != 2:
    print("Usage: python script.py <data_name>")
    sys.exit(1)

data_name = sys.argv[1]

base_dir = "C:/Users/robwi/Documents/ThesisFinal/Matrices/"
file_path = os.path.join(base_dir, "Distance_matrices", f"{data_name}_dtw.npy")

if not os.path.isfile(file_path):
    print(f"File {file_path} does not exist.")
    sys.exit(1)

s = io.StringIO()
csvWriter = csv.writer(s)

A = np.load(file_path)
if A.ndim == 1:
    A = A.reshape([A.size, 1])
csvWriter.writerows(A)

pyperclip.copy(s.getvalue())

print(f"Content of {file_path} copied to clipboard in CSV format.")

if sys.platform.startswith('linux'):
    input("Hit Enter to quit ...")
