# ==================================================================================================================== #
# Contributors: Tri Le + Phuoc Nguyen
# Description: Naive Bayes
# Filename: naive_bayes.py
# ==================================================================================================================== #

# Dependencies
import math
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix

# PATH FILES
DATA_PATH = "./processed_data.csv"


def readfile(filename):
    try:
        # Read raw dataset
        raw_data = np.loadtxt(filename, delimiter=",")
        return raw_data

    # File not found
    except FileNotFoundError:
        print("Cannot read files. Please check your dataset!")
        return None


data = readfile(DATA_PATH)
print(data.shape)
