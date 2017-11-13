'''
    The goal of this script is to perform multiple sets of analysis on the
heterogeneous activity recognition data set to provide us a better
understanding about the distribution of data, and what might be the
best possible sets to use for testing our transfer learning methods.
Since the data sets have so many kinds of heterogeneous data domains, all
with similar activity label sets, there are many ways in which we can
partition the data to test our transfer learning methods.


Kinds of analysis:
*Total number of each activity kind.
*Number of each kind of activity label per user.

*Histograms of the length of time for each activity label.

#UD
*Histograms of the length of time for activity labels on user by user basis.

*Ratios of each activity label to every other activity label.
*Ratios of each activity label for each user to every other user.
    -We can use this to determine which two users, and which two data sets
    we might compare.

'''

import numpy as np
import pandas as pd
import cvxpy as cvx

# Load all phone accelerometer data.
phoneAccelData = pd.read_csv("../Activity recognition exp/Phones_accelerometer.csv")
# Load a single set of time for data.
dat1Entry = phoneAccelData[0:1362520]



def main():
    print("Inductive SVM Code.")




if __name__ == "__main__":
    main()









