'''
This code implements a hybrid, discrete continuous generative bayes net.
This code assumes the bayes net structure is nearly markovian, with directed
links between nodes representing each time point, and a directed link
between the class node, and all data nodes. This network can be thought of
in many ways including a special "Tree-Augmented" bayes net, or a special
kind of Hidden Markov Model with the restriction that there is only one
unobserved state (The class node), which is tied to all of the observation
data.

'''
import numpy as np
import pandas as pd
import cvxpy as cvx
from sklearn.datasets import load_iris
from sklearn.svm import LinearSVC
import sklearn.svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import random

'''

'''
class BayesNet:

    def __init__(self):
        # List of dicts {-1: [aj, bj, ssj], 1: [aj, bj, ssj]}
        self._nodeList = None
        # Prior probability of class y.
        self._py = None

    '''
    Note: This function expects classes to be binary, and labeled -1 or 1.
    '''
    def learn(self, X, y):
        # Build and fit the Bayes Net.
        # X is nx500, where each row represents a window
        X = np.array(X)
        # Y is nx1, belonging either to class -1 or +1
        y = np.array(y)

        '''
        Steps for MLE estimates for each node xj:
        1) Filter rows in X to be only those that have the desired class, -1 or +1.
        2) Calculate the prior probability for x0 as a gaussian/uniform distribution for each class.
        3) Calculate xj's sample variance ssj.
        4) Calculate xj's MLE estimate for aj using the hand derived equation.
        5) Calculate xj's MLE estimate for bj by plugging the calculated aj into the derived equation.
        6) Given the aj's, bj's, and ssj's for each class value, build the Bayes Net Structure.
        '''

        # 1)
        X1 = X[y == 1]
        X_1 = X[y == -1]

        # 2)
        x1mean = np.mean(X1[:, 0], dtype=np.float64)
        x1ss = np.var(X1[:, 0], dtype=np.float64)
        x_1mean = np.mean(X_1[:, 0], dtype=np.float64)
        x_1ss = np.var(X_1[:, 0], dtype=np.float64)

        # 3)
        ss1 = np.var(X1, axis=0, dtype=np.float64)
        ss_1 = np.var(X_1, axis=0, dtype=np.float64)

        # 4)
        n1 = X1.shape[0]
        n_1 = X_1.shape[0]
        a1 = np.zeros((X1.shape[1], 1))
        a_1 = np.zeros((X_1.shape[1], 1))
        # Calculate ai for class 1
        for j in range(1, X1.shape[1]):
            sumXj = np.sum(X1[:, j])
            sumXj_1 = np.sum(X1[:, j - 1])
            M = -1 * np.sum(X1[:, j] * X1[:, j - 1])
            K = -1 * np.sum(X1[:, j - 1] * X1[:, j - 1])
            L = (-1 / n1) * sumXj * sumXj_1
            P = (-1 / n1) * sumXj_1 * sumXj_1
            a1[j] = (M + L) / (K + P)
        # Calculate ai for class -1
        for j in range(1, X_1.shape[1]):
            sumXj = np.sum(X_1[:, j])
            sumXj_1 = np.sum(X_1[:, j - 1])
            M = -1 * np.sum(X_1[:, j] * X_1[:, j - 1])
            K = -1 * np.sum(X_1[:, j - 1] * X_1[:, j - 1])
            L = (-1 / n1) * sumXj * sumXj_1
            P = (-1 / n1) * sumXj_1 * sumXj_1
            a_1[j] = (M + L) / (K + P)

        # 5)
        b1 = np.zeros((X1.shape[1], 1))
        b_1 = np.zeros((X_1.shape[1], 1))
        # Calculate bi for class 1
        for j in range(1, X1.shape[1]):
            b1[j] = (-1 / n1) * (np.sum(X1[:, j])) * (-1 * a1[j] * np.sum(X1[:, j - 1]))
        # Calculate bi for class -1
        for j in range(1, X_1.shape[1]):
            b_1[j] = (-1 / n1) * (np.sum(X_1[:, j])) * (-1 * a_1[j] * np.sum(X_1[:, j - 1]))

        # 6)
        # Base probability of class y
        py = {-1: X[y == -1].shape[0] / X.shape[0], 1: X[y == 1].shape[0] / X.shape[0]}
        # List of dicts {-1: [aj, bj, ssj], 1: [aj, bj, ssj]}
        nodeList = [{-1: [x_1mean, x_1ss], 1: [x1mean, x1ss]}]
        for j in range(1, X.shape[1]):
            nodeList.append({-1: [a_1[j], b_1[j], ss_1[j]], 1: [a1[j], b1[j], ss1[j]]})

        # Set the class variables
        self._nodeList = nodeList
        self._py = py

    '''
    This function accepts a time window x, and class label y, and calculates the probability
    of that instance.

    @:param x a single ROW (d, 1) shaped vector with width (number of columns) equal
    to the width of the original X matrix used to train this BayesNet.
    @:param y a single class label, -1 or 1
    '''
    def probability(self, x, y):
        # Initialize to prior probability of y
        probability = self._py[y]
        # Multiply gaussian prior probability variable x[0]
        const = (1/((2*np.pi*self._nodeList[0][y][1])**.5))
        probability *= const*np.exp(-.5* (((x[0] - self._nodeList[0][y][0])**2)/self._nodeList[0][y][1]) )
        # Multiply out all other prior probability variables.
        for j in range(1, x.shape[1]):
            const = (1/((2*np.pi*self._nodeList[j][y][2])**.5))
            linearMeanTerm = self._nodeList[j][y][0]*x[j-1] + self._nodeList[j][y][1]
            probability *= const*np.exp(-.5*(  (x[j] - ((linearMeanTerm)**2)/self._nodeList[j][y][2])  ))

        return probability


'''
This function reads in a data file, and parses out the data set X,
and class label set Y. It the builds a conditional gaussian
distributed, markovian based bayes net, where the class label
represents a discrete valued node in the graph, and each time
point represents a continuous, gaussian distributed node that
only depends on (Aka has parents) the previous value in time,
and the current class label value.
'''
def buildBayesNet():
    print("Loading data...")
    data = pd.read_csv("./Train_Phone-Acc-nexus4_1-a.csv")
    print("Done!")

    # Parse data and make bike vs not-biking classification using an SVM.
    # Note: I'm assuming a window width of 500
    print("Finding time series windows indexes for each class kind...")
    previousClassLabel = str(data.get_value(data.index[0], 'gt'))
    pos = 0
    y = []
    X = []
    window = 500
    while pos < data.shape[0]:
        # Make y label.
        if str(data.iloc[pos]['gt']) == 'sit':
            y.append(1)
        else:
            y.append(-1)

        # Make X row.
        X.append(data.iloc[pos:pos + window]['y'])

        # Move to the next window
        pos += window
    print("Done!")

    # # Build and fit the Bayes Net.
    # # X is nx500, where each row represents a window
    # X = np.array(X)
    # # Y is nx1, belonging either to class -1 or +1
    # y = np.array(y)
    #
    # '''
    # Steps for MLE estimates for each node xj:
    # 1) Filter rows in X to be only those that have the desired class, -1 or +1.
    # 2) Calculate the prior probability for x0 as a gaussian/uniform distribution for each class.
    # 3) Calculate xj's sample variance ssj.
    # 4) Calculate xj's MLE estimate for aj using the hand derived equation.
    # 5) Calculate xj's MLE estimate for bj by plugging the calculated aj into the derived equation.
    # 6) Given the aj's, bj's, and ssj's for each class value, build the Bayes Net Structure.
    # '''
    #
    # # 1)
    # X1 = X[y == 1]
    # X_1 = X[y == -1]
    #
    # # 2)
    # x1mean = np.mean(X1[:, 0], dtype=np.float64)
    # x1ss = np.var(X1[:, 0], dtype=np.float64)
    # x_1mean = np.mean(X_1[:, 0], dtype=np.float64)
    # x_1ss = np.var(X_1[:, 0], dtype=np.float64)
    #
    # # 3)
    # ss1 = np.var(X1, axis=0, dtype=np.float64)
    # ss_1 = np.var(X_1, axis=0, dtype=np.float64)
    #
    # # 4)
    # n1 = X1.shape[0]
    # n_1 = X_1.shape[0]
    # a1 = np.zeros((X1.shape[1], 1))
    # a_1 = np.zeros((X_1.shape[1], 1))
    # # Calculate ai for class 1
    # for j in range(1, X1.shape[1]):
    #     sumXj = np.sum(X1[:,j])
    #     sumXj_1 = np.sum(X1[:, j-1])
    #     M = -1*np.sum(X1[:, j]*X1[:, j-1])
    #     K = -1*np.sum(X1[:, j-1]*X1[:, j-1])
    #     L = (-1/n1)*sumXj*sumXj_1
    #     P = (-1/n1)*sumXj_1*sumXj_1
    #     a1[j] = (M + L) / (K + P)
    # # Calculate ai for class -1
    # for j in range(1, X_1.shape[1]):
    #     sumXj = np.sum(X_1[:, j])
    #     sumXj_1 = np.sum(X_1[:, j - 1])
    #     M = -1 * np.sum(X_1[:, j] * X_1[:, j - 1])
    #     K = -1 * np.sum(X_1[:, j - 1] * X_1[:, j - 1])
    #     L = (-1 / n1) * sumXj * sumXj_1
    #     P = (-1 / n1) * sumXj_1 * sumXj_1
    #     a_1[j] = (M + L) / (K + P)
    #
    # # 5)
    # b1 = np.zeros((X1.shape[1], 1))
    # b_1 = np.zeros((X_1.shape[1], 1))
    # # Calculate bi for class 1
    # for j in range(1, X1.shape[1]):
    #     b1[j] = (-1/n1)*(np.sum(X1[:, j]))*(-1*a1[j]*np.sum(X1[:, j-1]))
    # # Calculate bi for class -1
    # for j in range(1, X_1.shape[1]):
    #     b_1[j] = (-1 / n1) * (np.sum(X_1[:, j])) * (-1 * a_1[j] * np.sum(X_1[:, j - 1]))
    #
    # # 6)
    # # Base probability of class y
    # py = {-1: X[y == -1].shape[0] / X.shape[0], 1: X[y == 1].shape[0] / X.shape[0]}
    # # List of dicts {-1: [aj, bj, ssj], 1: [aj, bj, ssj]}
    # nodeList = [{-1: [x_1mean, x_1ss], 1: [x1mean, x1ss]}]
    # for j in range(1, X.shape[1]):
    #     nodeList.append({-1: [a_1[j], b_1[j], ss_1[j]], 1: [a1[j], b1[j], ss1[j]]})



def main():
    print("Loading data...")
    data = pd.read_csv("./Train_Phone-Acc-nexus4_1-a.csv")
    print("Done!")

    # Parse data and make bike vs not-biking classification using an SVM.
    # Note: I'm assuming a window width of 500
    print("Finding time series windows indexes for each class kind...")
    previousClassLabel = str(data.get_value(data.index[0], 'gt'))
    pos = 0
    y = []
    X = []
    window = 500
    while pos < data.shape[0]:
        # Make y label.
        if str(data.iloc[pos]['gt']) == 'sit':
            y.append(1)
        else:
            y.append(-1)

        # Make X row.
        X.append(data.iloc[pos:pos + window]['y'])

        # Move to the next window
        pos += window
    print("Done!")

    # Build and fit the Bayes Net.
    # X is nx500, where each row represents a window
    X = np.array(X)
    # Y is nx1, belonging either to class -1 or +1
    y = np.array(y)

    sourceBayesNet = BayesNet()
    sourceBayesNet.learn(X, y)
    for i in range(0, X.shape[0]):
        print("Probability for example %d is: %f"%(i, sourceBayesNet.probability(X[i,:].reshape((X[i,:].shape[0], 1)), y[i])))








if __name__ == "__main__":
    main()







































