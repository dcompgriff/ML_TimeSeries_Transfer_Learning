'''
    The goal of this script is to perform analysis of the labeled
    training data sets. This includes:
    *Histograms of the number of each kind of labeled example.
    *Plots of each time series activity for visual inspection of regularity.
    *Feature transformation and selection
    ---Possible kernel smoothing
    ---Possible sub sampling
    ---Possible construction of alternate feature sets


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
import matplotlib.pyplot as plt
import time as time
import argparse






def generateClassBarChart(data, window):
    # For each window offset, store the label of the first data example.
    barChartDict = {}
    for i in range(0, int(data.shape[0]/window)):
        try:
            barChartDict[str(data.iloc[i*window]['gt'])] += 1
        except:
            barChartDict[str(data.iloc[i*window]['gt'])] = 1

    # Make the matplotlib bar chart.
    keys = list(barChartDict.keys())
    values = []
    for key in barChartDict:
        values.append(barChartDict[key])
    plt.bar(np.arange(len(keys)), values)
    plt.xticks(np.arange(len(keys)), keys)
    plt.title('Class Label Count Bar chart')
    plt.show()



def main(args):
    # Load all phone accelerometer data.
    print("Loading data...")
    data = pd.read_csv(args.input_file)
    print("Done!")
    # print("Sorting data by arrival time...")
    # data = data.sort_values(by=['Arrival_Time'])
    # print("Done!")


    if args.p:
        generateClassBarChart(data, args.t)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process sensor data, and generate histograms and training data.')
    parser.add_argument('input_file', help='Input .csv file to read sensor data from.')
    parser.add_argument('-p', action='store_true', help='Generate bar chart for the number of each kind of example.')
    parser.add_argument('-t', type=int, default=35000, help='Window width size used to generate the training example.')
    args = parser.parse_args()

    main(args)




























