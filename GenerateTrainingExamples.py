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
import matplotlib.pyplot as plt
import time as time
import argparse

'''
This function iteratively scans the entire accelerometer dataframe in order
of creation time, and breaks the data frame into contiguous chunks based
on class label. It then creates a histogram of the "width" of time for
each class type, over each contiguous window. This should provide us with
an idea of average size the "width" of time windows for each class type.
This in turn may provide us an idea about how to break down all class types
into fixed size chunks. These smaller chunks of accelerometer data can
then become possible training instances.

Note: Use df.get_value(i, "col") and df.set_value(i, "col") instead of iloc. As, it
speeds up the iteration by a factor of 20 over .iterrows(), and a factor ov over 10
for iloc[].
'''
def histogramContiguousTimesPerClass(phoneAccelData):
    histogram = {}
    count = 0
    startTime = time.time()
    pos = 0

    print("Generating histogram of contiguous times per class...")
    previousClassLabel = str(phoneAccelData.get_value(phoneAccelData.index[0], 'gt'))
    for i in phoneAccelData.index:
        if pos%100000 == 0:
            endItime = time.time()
            print("%f percent completed."%((pos / phoneAccelData.shape[0])*100))
            startItime = time.time()
            print("Total Runtime: %f seconds."%(time.time() - startTime))


        if previousClassLabel == str(phoneAccelData.get_value(i, 'gt')):
            count += 1
        else:
            #The next class label has been found, so update the histogram counts.
            if previousClassLabel in histogram.keys():
                histogram[previousClassLabel].append(count)
                previousClassLabel = str(phoneAccelData.get_value(i, 'gt'))
                count = 1
            else:
                histogram[previousClassLabel] = [count]
                previousClassLabel = str(phoneAccelData.get_value(i, 'gt'))
                count = 1

        pos += 1

    print("Done!")

    print("Creating multiple plots of histograms...")
    plt.figure(1)
    i = 1
    for key in histogram.keys():
        plt.subplot(240 + i)
        plt.hist(histogram[key])
        plt.ylabel('Count')
        plt.xlabel('Width')
        plt.title(str(key))
        i += 1
    print("Done!")

    plt.show()


'''
This function is used to break accelerometer data into smaller windows, and
create equal width time series that can be used for classification.

Phone Accelerometer Threshold = 35,000

@:param threshold The fixed size of the window to break data down into.
'''
def generateTrainingExamples(threshold, output, phoneAccelData):
    count = 0
    startTime = time.time()
    pos = 0
    previousPos = 0
    windowSliceTupleList = []

    print("Generating window slice tuple list...")
    previousClassLabel = str(phoneAccelData.get_value(phoneAccelData.index[0], 'gt'))
    for i in phoneAccelData.index:
        if pos%100000 == 0:
            endItime = time.time()
            print("%f percent completed."%((pos / phoneAccelData.shape[0])*100))
            startItime = time.time()
            print("Total Runtime: %f seconds."%(time.time() - startTime))


        if previousClassLabel == str(phoneAccelData.get_value(i, 'gt')):
            count += 1
        else:
            # Make windows of width size "threshold", and label these divisions by their label.
            div = int((pos-previousPos) / threshold)
            for partition in range(0, div):
                windowSliceTupleList.append((previousPos + (threshold*partition), previousPos + (threshold*(partition+1)), previousClassLabel))

            previousPos = pos
            previousClassLabel = str(phoneAccelData.get_value(i, 'gt'))
            count = 1

        pos += 1
    print("Done!")

    print("Creating single data frame with equal width time series...")
    fixedWidthWindowList = []
    for tuple in windowSliceTupleList:
        # Append each fixed slice window of time series from the original data frame.
        fixedWidthWindowList.append(phoneAccelData[tuple[0]:tuple[1]])
    print("Done!")

    print("Concatenating training example dataframes...")
    fixedWidthPhoneAccelData = pd.concat(fixedWidthWindowList)
    print("Done!")
    print("Writing training set dataframe to output file...")
    fixedWidthPhoneAccelData.to_csv(output,index=False)
    print("Done!")

def main(args):
    print("Accelerometer Analysis Code")

    # Load all phone accelerometer data.
    print("Loading data...")
    phoneAccelData = pd.read_csv(args.input_file)
    print("Done!")
    print("Sorting data by arrival time...")
    phoneAccelData = phoneAccelData.sort_values(by=['Arrival_Time'])
    print("Done!")

    # If histogram plot option is enabled, plot the histogram.
    if args.p:
        histogramContiguousTimesPerClass(phoneAccelData)

    # If an output file was specified, generate the training data and output to the csv file.
    if not (args.output is None):
        generateTrainingExamples(args.t, args.output, phoneAccelData)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process sensor data, and generate histograms and training data.')
    parser.add_argument('input_file', help='Input .csv file to read sensor data from.')
    parser.add_argument('-p', action='store_true', help='Generate visual histogram plot of widths of each activity in the data set.')
    parser.add_argument('-t', type=int, default=500, help='Window width size to generate for training example data set. '
                                            '\nBase thresholds:\n'
                                            'Phone Accelerometer/Gyroscope: 35000\n'
                                                  'Watch Accelerometer/Gyroscope: 10000')
    parser.add_argument('--output', help='Output file to place the generated training examples into.')
    args = parser.parse_args()

    main(args)









