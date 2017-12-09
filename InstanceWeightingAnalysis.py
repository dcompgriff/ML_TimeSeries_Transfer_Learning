import numpy as np
import pandas as pd
import cvxpy as cvx
from sklearn.datasets import load_iris
from sklearn.svm import LinearSVC
import sklearn.svm
from sklearn.svm.classes import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import random
import time as time
import argparse

import GenerativeBayesNet


def calculateTotalAbsoluteError(yPredicted, yOriginal):
    count = 0
    for i in range(0, yOriginal.shape[0]):
        if yPredicted[i] != yOriginal[i]:
            count += 1
    return count

def main(args):
    print("Loading data...")
    # data = pd.read_csv("./Train_Phone-Acc-nexus4_1-a.csv")
    # data2 = pd.read_csv("./Train_Phone-Acc-nexus4_1-b.csv")
    sourceData = pd.read_csv(args.source_domain_file)
    targetData = pd.read_csv(args.target_domain_file)
    print("Done!")

    # Parse data and make bike vs not-biking classification using an SVM.
    # Note: I'm assuming a window width of 500
    print("Finding time series windows indexes for each class kind...")
    previousClassLabel = str(sourceData.get_value(sourceData.index[0], 'gt'))
    pos = 0
    y = []
    X = []
    yt = []
    Xt = []
    window = 500
    while pos < sourceData.shape[0]:
        # Make y label.
        if str(sourceData.iloc[pos]['gt']) == args.stype:
            y.append(1)
        else:
            y.append(-1)

        # Make X row.
        X.append(sourceData.iloc[pos:pos + window]['y'])

        # Move to the next window
        pos += window

    pos = 0
    window = 500
    while pos < targetData.shape[0]:
        # Make yt label.
        if str(targetData.iloc[pos]['gt']) == args.ttype:
            yt.append(1)
        else:
            yt.append(-1)

        # Make Xt row.
        Xt.append(targetData.iloc[pos:pos + window]['y'])

        # Move to the next window
        pos += window
    print("Done!")

    # Build and fit the Bayes Net.
    # X is nx500, where each row represents a window
    X = np.array(X)
    # Y is nx1, belonging either to class -1 or +1
    y = np.array(y)

    Xt = np.array(Xt)
    # Y is nx1, belonging either to class -1 or +1
    yt = np.array(yt)


    weightedErrorList = []
    unweightedErrorList = []
    weightedFscoreList = []
    unweightedFscoreList = []
    weightedMeanAccuracyList = []
    unweightedMeanAccuracyList = []
    for k in range(0, 100):
        if k % 1 == 0:
            print("Iteration %d"%(k))

        Xt_train, Xt_test, yt_train, yt_test = train_test_split(Xt, yt, stratify=yt, test_size=0.1)  # , random_state = 0

        # Subsample the window to a width of 50.
        #X = X[:, list(range(0, 500, 50))]
        GenX_train = X[:, list(range(0, 500, 50))]
        GenXt_train = Xt_train[:, list(range(0, 500, 50))]

        # Build the target bayes net.
        #print("Building bayes net...")
        sourceBayesNet = GenerativeBayesNet.BayesNet()
        sourceBayesNet.learn(GenX_train, y)
        targetBayesNet = GenerativeBayesNet.BayesNet()
        targetBayesNet.learn(GenXt_train, yt_train)
        #print("Done!")

        # Build the weights list.
        #print("Building weight list...")
        weights = []
        #for i in range(0, Xt_train.shape[0]):
        #   weights.append(10)
        for i in range(0, X.shape[0]):
            #pSource = sourceBayesNet.probability(Xt_test[i,:].reshape((1, Xt_test[i,:].shape[0])), yt_test[i])
            pTarget = targetBayesNet.probability(GenX_train[i,:].reshape((1, GenX_train[i,:].shape[0])), y[i])
            weights.append(pTarget*1000000)
            #weights.append(random.uniform(0, 0.5))
        #print("Done!")

        # Build the weighted SVM.
        #print("Building weighted SVM...")
        Xsource_and_target = np.vstack((Xt_train, X))
        Ysource_and_target = np.vstack((yt_train.reshape((len(yt_train), 1)), y.reshape((len(y), 1))))
        clfsc = SVC()
        clfsc.fit(Xsource_and_target, Ysource_and_target, sample_weight=weights)
        yhatCombined = clfsc.predict(Xt_test)
        #print("Done!")

        # Build the regular SVM.
        #print("Building regular SVM...")
        clfs = SVC()
        clfs.fit(Xt_train, yt_train)
        yhatTargetOnly = clfs.predict(Xt_test)
        #print("Done!")

        # Evaluate both SVMs.
        weightedErrorList.append(calculateTotalAbsoluteError(yhatCombined, yt_test)/len(yt_test))
        weightedFscoreList.append(f1_score(yt_test, yhatCombined, pos_label=1, average='binary'))
        unweightedErrorList.append(calculateTotalAbsoluteError(yhatTargetOnly, yt_test)/len(yt_test))
        unweightedFscoreList.append(f1_score(yt_test, yhatTargetOnly, pos_label=1, average='binary'))
        weightedMeanAccuracyList.append(clfsc.score(Xt_test, yt_test))
        unweightedMeanAccuracyList.append(clfs.score(Xt_test, yt_test))

        # print()
        # print("Evaluating SVMs on test set...")
        # print("Weighted SVM Abs Error = %f"%(calculateTotalAbsoluteError(yhatCombined, yt_test)/len(yt_test)))
        # print("Weighted SVM mean accuracy score: %f"%clfsc.score(Xt_test, yt_test))
        # f1 = f1_score(yt_test, yhatCombined, pos_label=1, average='binary')
        # print("Weighted SVM f1 score: %f" % (f1))
        #
        # print()
        # print("SVM Abs Error = %f"%(calculateTotalAbsoluteError(yhatTargetOnly, yt_test)/len(yt_test)))
        # print("SVM mean accuracy score: %f"%clfs.score(Xt_test, yt_test))
        # f1 = f1_score(yt_test, yhatTargetOnly, pos_label=1, average='binary')
        # print("SVM f1 score: %f" % (f1))

    print()
    print("Weighted SVM Average Error = %f"%(np.average(weightedErrorList)))
    print("Weighted SVM Average fscore = %f" % (np.average(weightedFscoreList)))
    print("Weighted SVM Average mean Accuracy = %f"%(np.average(weightedMeanAccuracyList)))
    print()
    print("Weighted SVM Std. Dev = %f" % (np.std(weightedErrorList)))
    print("Weighted SVM Std. Dev fscore = %f" % (np.std(weightedFscoreList)))
    print("Weighted SVM Std. Dev mean Accuracy = %f" % (np.std(weightedMeanAccuracyList)))

    print()
    print("SVM Average Error = %f" % (np.average(unweightedErrorList)))
    print("SVM Average fscore = %f" % (np.average(unweightedFscoreList)))
    print("SVM Average mean Accuracy = %f"%(np.average(unweightedMeanAccuracyList)))
    print()
    print("SVM Std. Dev = %f" % (np.std(unweightedErrorList)))
    print("SVM Std. Dev fscore = %f" % (np.std(unweightedFscoreList)))
    print("SVM Std. Dev mean Accuracy = %f" % (np.std(unweightedMeanAccuracyList)))


    n = 100
    pStd = ( ( ((n-1)*np.std(weightedMeanAccuracyList)) + ((n-1)*np.std(unweightedMeanAccuracyList))) / (2*n - 2) )**.5
    t = ( (np.mean(weightedMeanAccuracyList) - (np.mean(unweightedMeanAccuracyList))) / (pStd*((2/n)**.5)) )

    print(pStd)
    print(t)










if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process sensor data, and generate histograms and training data.')
    parser.add_argument('source_domain_file', help='Training .csv file for the source domain.')
    parser.add_argument('target_domain_file', help='Training .csv file for the target domain.')
    parser.add_argument('--stype', type=str, default='sit', help='Task label type for source domain.')
    parser.add_argument('--ttype', type=str, default='walk', help='Task label type for target domain.')
    args = parser.parse_args()

    main(args)



